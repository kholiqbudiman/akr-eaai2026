"""
=============================================================================
AKR Factorial Ablation: 2x2 Design (Adaptive alpha x Easy-Skip)
=============================================================================
Config matched to run_k_ablation_v3_a100.py:
  - AdamW lr=1e-3, wd=1e-3
  - Warmup 5 epochs + CosineAnnealingLR
  - CrossEntropyLoss(label_smoothing=0.1)
  - Early stop: 15 epochs (no_improve += 5, check every 5 epochs)
  - Batch size 64, WeightedRandomSampler
  - AMP (GradScaler)
  - T=4.0, alpha=0.5

4 variants:
  1. vanilla     — No adaptive alpha, No easy-skip
  2. akr_noskip  — Adaptive alpha + routing, No easy-skip
  3. skip_only   — No adaptive alpha, WITH easy-skip (NEW)
  4. akr         — Full AKR

Students: WasteNet-Micro (21.6K), WasteNet-Small (43K)
Datasets: WaRP-K28 (IR=59.6), WaRP-K7 (IR=27.5), TrashNet-K6 (IR=4.4),
          TrashNet-K5 (IR=1.5)
Seeds: 5 (42, 123, 456, 789, 2024)
Total: 4 x 2 x 4 x 5 = 160 runs

Usage:  python run_akr_factorial_ablation.py --gpu 0
=============================================================================
"""

import os, sys, json, copy, random, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_kd_framework import (
    WasteNetMicro, WasteNetSmall, count_params, get_transforms,
)
from speed_utils_v3 import (
    resize_for_teacher, kd_loss, compute_ece,
)

# ===================================================================
# Config (matched to k-ablation v3)
# ===================================================================
SEEDS = [5, 42, 99, 123, 456, 777, 789, 1024, 2026, 2048]
EPOCHS = 100
LR = 1e-3
WD = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 4
TEMPERATURE = 4.0
ALPHA = 0.5
PATIENCE = 15  # no_improve += 5 per eval, so 3 bad evals = stop

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
ABLATION_DIR = Path(os.path.expanduser("~/KD/results_akr_factorial"))


# ===================================================================
# Evaluation (handles both 2-tuple and 3-tuple batches)
# ===================================================================
def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in dataloader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
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


# ===================================================================
# Training functions (all use same teacher forward, no cached logits)
# ===================================================================

def _get_teacher_logits(teacher, images, teacher_input_size=224):
    t_images = resize_for_teacher(images, teacher_input_size)
    with torch.no_grad():
        return teacher(t_images)

def _difficulty(t_logits):
    t_probs = F.softmax(t_logits, dim=1)
    top2 = t_probs.topk(2, dim=1).values
    return 1.0 - (top2[:, 0] - top2[:, 1])


def train_vanilla_kd(student, teacher, dataloader, optimizer, criterion,
                     device, scaler):
    student.train(); teacher.eval()
    rloss = correct = total = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            s_out = student(images)
            t_logits = _get_teacher_logits(teacher, images)
            loss = ALPHA * kd_loss(s_out, t_logits, TEMPERATURE) + \
                   (1 - ALPHA) * criterion(s_out, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        rloss += loss.item()
        _, p = s_out.max(1); correct += p.eq(labels).sum().item(); total += labels.size(0)
    return rloss / len(dataloader), 100.0 * correct / total


def train_skip_only(student, teacher, dataloader, optimizer, criterion,
                    device, scaler):
    """Easy=CE only; standard+hard=vanilla KD with FIXED alpha. Single teacher."""
    student.train(); teacher.eval()
    rloss = correct = total = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        bs = images.size(0)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            s_out = student(images)
            t_logits = _get_teacher_logits(teacher, images)
            d = _difficulty(t_logits)
            easy_mask = d < d.quantile(0.33).item()
            distill_mask = ~easy_mask
            loss_ce = criterion(s_out, labels)
            if distill_mask.any():
                lkd = kd_loss(s_out[distill_mask], t_logits[distill_mask], TEMPERATURE)
                frac = distill_mask.sum().float() / bs
                loss = ALPHA * frac * lkd + (1.0 - ALPHA * frac) * loss_ce
            else:
                loss = loss_ce
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        rloss += loss.item()
        _, p = s_out.max(1); correct += p.eq(labels).sum().item(); total += labels.size(0)
    return rloss / len(dataloader), 100.0 * correct / total


def train_akr_noskip(student, base_teacher, ta_teacher, dataloader, optimizer,
                     criterion, device, scaler):
    """Routing + adaptive alpha, but ALL samples get KD (no easy-skip)."""
    student.train(); base_teacher.eval(); ta_teacher.eval()
    rloss = correct = total = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            s_out = student(images)
            t_images = resize_for_teacher(images, 224)
            with torch.no_grad():
                bt_logits = base_teacher(t_images)
            d = _difficulty(bt_logits)
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
                with torch.no_grad():
                    ta_logits = ta_teacher(t_images[std_mask])
                loss_kd = loss_kd + alphas[std_mask].mean() * kd_loss(
                    s_out[std_mask], ta_logits, TEMPERATURE)
            if easy_mask.any():
                with torch.no_grad():
                    ta_easy = ta_teacher(t_images[easy_mask])
                loss_kd = loss_kd + alphas[easy_mask].mean() * kd_loss(
                    s_out[easy_mask], ta_easy, TEMPERATURE)
            loss = ALPHA * loss_kd + (1 - ALPHA) * loss_ce
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        rloss += loss.item()
        _, p = s_out.max(1); correct += p.eq(labels).sum().item(); total += labels.size(0)
    return rloss / len(dataloader), 100.0 * correct / total


def train_akr_full(student, base_teacher, ta_teacher, dataloader, optimizer,
                   criterion, device, scaler):
    """Full AKR: routing + adaptive alpha + easy-skip."""
    student.train(); base_teacher.eval(); ta_teacher.eval()
    rloss = correct = total = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        bs = images.size(0)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            s_out = student(images)
            t_images = resize_for_teacher(images, 224)
            with torch.no_grad():
                bt_logits = base_teacher(t_images)
            d = _difficulty(bt_logits)
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
                with torch.no_grad():
                    ta_logits = ta_teacher(t_images[std_mask])
                loss_kd = loss_kd + alphas[std_mask].mean() * kd_loss(
                    s_out[std_mask], ta_logits, TEMPERATURE)
            n_dist = hard_mask.sum() + std_mask.sum()
            if n_dist > 0:
                kd_w = ALPHA * (n_dist.float() / bs)
                loss = kd_w * loss_kd + (1.0 - kd_w) * loss_ce
            else:
                loss = loss_ce
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        rloss += loss.item()
        _, p = s_out.max(1); correct += p.eq(labels).sum().item(); total += labels.size(0)
    return rloss / len(dataloader), 100.0 * correct / total


# ===================================================================
# Teacher loader
# ===================================================================
def load_teacher(teacher_dir, name, num_classes, device):
    if name == "convnext-tiny":
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
    elif name == "efficientnet-b0":
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    w = torch.load(teacher_dir / f"teacher_{name}.pt",
                    map_location=device, weights_only=True)
    model.load_state_dict(w)
    return model.to(device).eval()


# ===================================================================
# Single experiment runner
# ===================================================================
def run_single(method, student_name, k, seed, device,
               base_teacher, ta_teacher, train_loader, test_loader,
               num_classes):
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

    best_acc = 0; best_state = None; no_improve = 0

    for epoch in range(EPOCHS):
        if method == "vanilla":
            loss, tacc = train_vanilla_kd(model, base_teacher, train_loader,
                                          optimizer, criterion, device, scaler)
        elif method == "skip_only":
            loss, tacc = train_skip_only(model, base_teacher, train_loader,
                                         optimizer, criterion, device, scaler)
        elif method == "akr_noskip":
            loss, tacc = train_akr_noskip(model, base_teacher, ta_teacher,
                                          train_loader, optimizer, criterion,
                                          device, scaler)
        elif method == "akr":
            loss, tacc = train_akr_full(model, base_teacher, ta_teacher,
                                        train_loader, optimizer, criterion,
                                        device, scaler)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            val = evaluate_model(model, test_loader, device)
            vacc = val["accuracy"]
            if vacc > best_acc:
                best_acc = vacc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5  # matched to v3 convention
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
        "method": method, "student": student_name, "k": k, "seed": seed,
        "params": params, "accuracy": final["accuracy"],
        "f1_macro": final["f1_macro"], "ece": final["ece"],
        "timestamp": datetime.now().isoformat(),
    }


# ===================================================================
# Main
# ===================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--methods", nargs="+",
                        default=["vanilla", "akr_noskip", "skip_only", "akr"])
    parser.add_argument("--k-values", nargs="+", type=int, default=[28, 7, 6, 5])
    parser.add_argument("--students", nargs="+",
                        default=["WasteNet-Micro", "WasteNet-Small"])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.benchmark = True
    ABLATION_DIR.mkdir(parents=True, exist_ok=True)

    results_file = ABLATION_DIR / "factorial_results.json"
    all_results = json.load(open(results_file)) if results_file.exists() else []

    def is_done(m, s, kv, sd):
        return any(r["method"]==m and r["student"]==s and r["k"]==kv and r["seed"]==sd
                   for r in all_results)

    total = len(args.methods) * len(args.students) * len(args.k_values) * len(SEEDS)
    done = sum(1 for m in args.methods for s in args.students
               for kv in args.k_values for sd in SEEDS if is_done(m, s, kv, sd))
    print(f"\n{'='*70}")
    print(f"  AKR Factorial Ablation — {datetime.now():%Y-%m-%d %H:%M}")
    print(f"  Methods: {args.methods}")
    print(f"  Students: {args.students}, K-values: {args.k_values}")
    print(f"  Seeds: {SEEDS}")
    print(f"  Total: {total}, Done: {done}, Remaining: {total - done}")
    print(f"  GPU: {args.gpu}")
    print(f"  Config: lr={LR}, wd={WD}, T={TEMPERATURE}, alpha={ALPHA}")
    print(f"  BS={BATCH_SIZE}, epochs={EPOCHS}, patience={PATIENCE}")
    print(f"{'='*70}\n")

    for kv in args.k_values:
        cfg = K_CONFIGS[kv]
        data_dir = Path(os.path.expanduser(cfg["data_dir"]))
        teacher_dir = OUTPUT_ROOT / cfg["teacher_dir"]
        train_dir = data_dir / "train"
        num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
        print(f"\n  K={kv}: {num_classes} classes, data={data_dir}")

        # Transforms at student resolution
        train_tf = get_transforms(96, is_train=True)
        test_tf = get_transforms(96, is_train=False)

        train_ds = ImageFolder(str(train_dir), transform=train_tf)
        test_ds = ImageFolder(str(data_dir / "test"), transform=test_tf)

        # WeightedRandomSampler for class balance (matched to v3)
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

        # Load teachers
        base_teacher = load_teacher(teacher_dir, "convnext-tiny", num_classes, device)
        ta_teacher = load_teacher(teacher_dir, "efficientnet-b0", num_classes, device)

        for student_name in args.students:
            for method in args.methods:
                for seed in SEEDS:
                    if is_done(method, student_name, kv, seed):
                        print(f"  [SKIP] {method}/{student_name}/K={kv}/s={seed}")
                        continue
                    print(f"\n  {'='*60}")
                    print(f"  {method} | {student_name} | K={kv} | seed={seed}")
                    print(f"  {'='*60}")
                    result = run_single(method, student_name, kv, seed, device,
                                        base_teacher, ta_teacher,
                                        train_loader, test_loader, num_classes)
                    all_results.append(result)
                    with open(results_file, "w") as f:
                        json.dump(all_results, f, indent=2)

        del base_teacher, ta_teacher
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  DONE! {len(all_results)} results in {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
