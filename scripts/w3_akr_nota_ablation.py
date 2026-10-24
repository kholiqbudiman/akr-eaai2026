"""
W3: AKR-no-TA ablation — route ALL non-skipped samples to base teacher.
Reuses cached logits from k-ablation v3 (same as factorial ablation).

Full AKR:    hard→base_teacher, std→TA, easy→skip, adaptive alpha
AKR-no-TA:  hard→base_teacher, std→base_teacher, easy→skip, adaptive alpha

2 students × 4 K-values × 10 seeds = 80 runs.
"""

import os, sys, json, random, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(Path(os.path.expanduser("~/KD/scripts"))))

from train_kd_framework import (
    WasteNetMicro, WasteNetSmall, count_params, get_transforms,
)
from speed_utils_v3 import (
    IndexedImageFolder, kd_loss, compute_ece,
)

SEEDS = [5, 42, 99, 123, 456, 777, 789, 1024, 2026, 2048]
EPOCHS = 100
LR = 1e-3
WD = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 4
TEMPERATURE = 4.0
ALPHA = 0.5
PATIENCE = 15

STUDENTS = {
    "WasteNet-Micro": {"factory": lambda nc: WasteNetMicro(nc), "input_size": 96},
    "WasteNet-Small": {"factory": lambda nc: WasteNetSmall(nc), "input_size": 96},
}

K_CONFIGS = {
    28: {"data_dir": "~/KD/wasterecycle/results_teacher_yolo11/warp_c_crops",
         "teacher_dir": "teachers_warpc_k28"},
    7:  {"data_dir": "~/KD/results_k_ablation/warpc_k7",
         "teacher_dir": "teachers_k7"},
    6:  {"data_dir": "~/KD/wasterecycle/data/trashnet",
         "teacher_dir": "teachers_trashnet6"},
    5:  {"data_dir": "~/KD/results_k_ablation/trashnet_k5",
         "teacher_dir": "teachers_trashnet5"},
}

OUTPUT_ROOT = Path(os.path.expanduser("~/KD/results_k_ablation"))
RESULTS_DIR = Path(os.path.expanduser("~/KD/results_w3_nota"))


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1]
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_logits.append(outputs.cpu())
    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    logits_t = torch.cat(all_logits, dim=0)
    acc = 100.0 * np.mean(preds_np == labels_np)
    f1 = round(f1_score(labels_np, preds_np, average="macro", zero_division=0) * 100, 4)
    ece = compute_ece(logits_t, labels_np)
    return {"accuracy": round(acc, 4), "f1_macro": f1, "ece": ece}


def _difficulty_from_logits(logits):
    probs = F.softmax(logits, dim=1)
    top2 = probs.topk(2, dim=1).values
    return 1.0 - (top2[:, 0] - top2[:, 1])


def train_akr_nota(student, dataloader, optimizer, criterion,
                   device, scaler, cached_base_logits):
    """Full AKR but standard→base_teacher instead of TA.
    hard→base, std→base, easy→skip, adaptive alpha."""
    student.train()
    rloss = correct = total = 0
    for images, labels, indices in dataloader:
        bs = images.size(0)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            s_out = student(images)
            bt_logits = cached_base_logits[indices].to(device)
            d = _difficulty_from_logits(bt_logits)
            et, ht = d.quantile(0.33).item(), d.quantile(0.67).item()
            easy_mask = d < et
            hard_mask = d >= ht
            std_mask = ~easy_mask & ~hard_mask

            loss_ce = criterion(s_out, labels)
            alphas = d.clamp(0.1, 0.7)
            loss_kd = torch.tensor(0.0, device=device)

            if hard_mask.any():
                loss_kd = loss_kd + alphas[hard_mask].mean() * kd_loss(
                    s_out[hard_mask], bt_logits[hard_mask], TEMPERATURE)
            if std_mask.any():
                # KEY DIFFERENCE: use base teacher for standard samples (not TA)
                loss_kd = loss_kd + alphas[std_mask].mean() * kd_loss(
                    s_out[std_mask], bt_logits[std_mask], TEMPERATURE)

            # Easy → skip (no KD, only CE)
            n_distilled = hard_mask.sum() + std_mask.sum()
            if n_distilled > 0:
                kd_weight = ALPHA * (n_distilled.float() / bs)
                loss = kd_weight * loss_kd + (1.0 - kd_weight) * loss_ce
            else:
                loss = loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        rloss += loss.item()
        _, p = s_out.max(1)
        correct += p.eq(labels).sum().item()
        total += labels.size(0)
    return rloss / len(dataloader), 100.0 * correct / total


def run_single(student_name, k, seed, device,
               train_loader, test_loader, num_classes,
               cached_base_logits):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = STUDENTS[student_name]["factory"](num_classes).to(device)
    params = count_params(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    warmup_epochs = min(5, EPOCHS // 10)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=EPOCHS - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()

    best_acc = 0
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        loss, tacc = train_akr_nota(model, train_loader, optimizer,
                                    criterion, device, scaler,
                                    cached_base_logits)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            val = evaluate_model(model, test_loader, device)
            vacc = val["accuracy"]
            if vacc > best_acc:
                best_acc = vacc
                best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:
                print(f"    Epoch {epoch+1}/{EPOCHS} -- Loss: {loss:.4f}, "
                      f"Train: {tacc:.2f}%, Val: {vacc:.2f}% (best: {best_acc:.2f}%)")
        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state:
        model.load_state_dict(best_state)
    final = evaluate_model(model, test_loader, device)
    print(f"  Final: Acc={final['accuracy']:.2f}%, "
          f"F1={final['f1_macro']:.2f}%, ECE={final['ece']:.4f}")

    return {
        "method": "akr_nota",
        "student": student_name, "k": k, "seed": seed,
        "params": params, "accuracy": final["accuracy"],
        "f1_macro": final["f1_macro"], "ece": final["ece"],
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k-values", nargs="+", type=int, default=[28, 7, 6, 5])
    parser.add_argument("--students", nargs="+",
                        default=["WasteNet-Micro", "WasteNet-Small"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--tag", type=str, default="")
    args = parser.parse_args()

    seeds = args.seeds if args.seeds else SEEDS

    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.benchmark = True
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""
    results_file = RESULTS_DIR / f"akr_nota_results{suffix}.json"
    all_results = json.load(open(results_file)) if results_file.exists() else []

    def is_done(s, kv, sd):
        return any(r["student"] == s and r["k"] == kv and r["seed"] == sd
                   for r in all_results)

    total = len(args.students) * len(args.k_values) * len(seeds)
    done = sum(1 for s in args.students for kv in args.k_values
               for sd in seeds if is_done(s, kv, sd))
    print(f"\n{'='*70}")
    print(f"  W3: AKR-no-TA Ablation — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Students: {args.students}, K-values: {args.k_values}")
    print(f"  Seeds: {seeds}")
    print(f"  Total: {total}, Done: {done}, Remaining: {total - done}")
    print(f"  GPU: {args.gpu}")
    print(f"{'='*70}\n")

    for kv in args.k_values:
        cfg = K_CONFIGS[kv]
        data_dir = Path(os.path.expanduser(cfg["data_dir"]))
        teacher_dir = OUTPUT_ROOT / cfg["teacher_dir"]
        cache_dir = teacher_dir / "cache"
        train_dir = data_dir / "train"
        num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
        print(f"\n  K={kv}: {num_classes} classes, data={data_dir}")

        train_tf = get_transforms(96, is_train=True)
        test_tf = get_transforms(96, is_train=False)
        train_ds = IndexedImageFolder(str(train_dir), transform=train_tf)
        test_ds = IndexedImageFolder(str(data_dir / "test"), transform=test_tf)

        targets = train_ds.targets
        class_counts = [0] * len(train_ds.classes)
        for t in targets:
            class_counts[t] += 1
        weights = [1.0 / class_counts[t] for t in targets]
        sampler = WeightedRandomSampler(weights, len(targets), replacement=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=NUM_WORKERS, pin_memory=True)

        base_cache = cache_dir / "base_train_logits.pt"
        print(f"  Loading cached base logits from {cache_dir}")
        cached_base = torch.load(base_cache, map_location="cpu", weights_only=True)
        print(f"  Base: {cached_base.shape}")

        for student_name in args.students:
            for seed in seeds:
                if is_done(student_name, kv, seed):
                    print(f"  [SKIP] akr_nota/{student_name}/K={kv}/s={seed}")
                    continue
                print(f"\n  {'='*60}")
                print(f"  akr_nota | {student_name} | K={kv} | seed={seed}")
                print(f"  {'='*60}")
                result = run_single(student_name, kv, seed, device,
                                    train_loader, test_loader, num_classes,
                                    cached_base)
                all_results.append(result)
                with open(results_file, "w") as f:
                    json.dump(all_results, f, indent=2)

        del cached_base
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  DONE! {len(all_results)} results in {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
