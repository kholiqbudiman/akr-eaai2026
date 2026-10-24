"""
=============================================================================
AKR Sensitivity Analysis (Reviewer Fixes W3, W4, W8)
=============================================================================
Three experiments in one script:

  1. THRESHOLD SENSITIVITY (W3): vary quantile splits
     - Default: 33/33/33
     - Test: 25/50/25, 20/60/20, 40/20/40, 50/0/50 (no standard tier)

  2. ALPHA BOUNDS SENSITIVITY (W8): vary alpha clamp range
     - Default: [0.1, 0.7]
     - Test: [0.1, 0.5], [0.1, 0.9], [0.3, 0.7], [0.0, 1.0], fixed 0.5

  3. LABEL SMOOTHING BASELINE (W4): compare AKR vs LS at same epsilon
     - LS(0.0), LS(0.1), LS(0.2), LS(0.3)

Config matched to run_akr_factorial_ablation.py.
Runs on compact students only (WasteNet-Micro, WasteNet-Small).
K=28 (most imbalanced) and K=6 (moderate).
10 seeds per cell.
Uses pre-cached teacher logits for maximum speed.

Usage:
  python run_akr_sensitivity.py --experiment threshold --gpu 0
  python run_akr_sensitivity.py --experiment alpha --gpu 0
  python run_akr_sensitivity.py --experiment label_smoothing --gpu 1
  python run_akr_sensitivity.py --experiment all --gpu 0
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
    kd_loss, compute_ece,
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

OUTPUT_ROOT = Path(os.path.expanduser("~/KD/results_k_ablation"))
RESULTS_DIR = OUTPUT_ROOT / "akr_sensitivity"

STUDENTS = {
    "WasteNet-Micro": {"factory": lambda nc: WasteNetMicro(nc), "input_size": 96},
    "WasteNet-Small": {"factory": lambda nc: WasteNetSmall(nc), "input_size": 96},
}

K_CONFIGS = {
    28: {"data_dir": "~/KD/wasterecycle/results_teacher_yolo11/warp_c_crops",
         "teacher_dir": "teachers_warpc_k28"},
    6:  {"data_dir": "~/KD/wasterecycle/data/trashnet",
         "teacher_dir": "teachers_trashnet6"},
}

THRESHOLD_CONFIGS = {
    "q33_33": (0.33, 0.67),
    "q25_75": (0.25, 0.75),
    "q20_80": (0.20, 0.80),
    "q40_60": (0.40, 0.60),
    "q50_50": (0.50, 0.50),
}

ALPHA_BOUND_CONFIGS = {
    "a01_07": (0.1, 0.7),
    "a01_05": (0.1, 0.5),
    "a01_09": (0.1, 0.9),
    "a03_07": (0.3, 0.7),
    "a00_10": (0.0, 1.0),
    "fixed05": (0.5, 0.5),
}


class IndexedImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label, index


def _difficulty(t_logits):
    t_probs = F.softmax(t_logits, dim=1)
    top2 = t_probs.topk(2, dim=1).values
    return 1.0 - (top2[:, 0] - top2[:, 1])


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds, all_labels, all_logits = [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for batch in dataloader:
            images, labels = batch[0].to(device), batch[1].to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.append(outputs.cpu())
    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    logits_t = torch.cat(all_logits, dim=0)
    acc = 100.0 * np.mean(preds_np == labels_np)
    f1 = round(f1_score(labels_np, preds_np, average="macro", zero_division=0) * 100, 4)
    ece = compute_ece(logits_t, labels_np)
    return {"accuracy": round(acc, 4), "f1_macro": f1, "ece": ece}


def train_akr_cached(student, cached_base, cached_ta, dataloader,
                     optimizer, criterion, device, scaler,
                     easy_quantile=0.33, hard_quantile=0.67,
                     alpha_lo=0.1, alpha_hi=0.7):
    student.train()
    rloss = correct = total = 0
    for images, labels, indices in dataloader:
        images, labels = images.to(device), labels.to(device)
        bs = images.size(0)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            s_out = student(images)
            bt_logits = cached_base[indices].to(device)
            d = _difficulty(bt_logits)
            et = d.quantile(easy_quantile).item()
            ht = d.quantile(hard_quantile).item()
            easy_mask = d < et
            hard_mask = d >= ht
            std_mask = ~easy_mask & ~hard_mask
            loss_ce = criterion(s_out, labels)
            alphas = d.clamp(alpha_lo, alpha_hi)
            loss_kd = torch.tensor(0.0, device=device)
            if hard_mask.any():
                loss_kd = loss_kd + alphas[hard_mask].mean() * kd_loss(
                    s_out[hard_mask], bt_logits[hard_mask], TEMPERATURE)
            if std_mask.any():
                ta_logits = cached_ta[indices[std_mask.cpu()]].to(device)
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


def train_label_smoothing(student, dataloader, optimizer, criterion, device, scaler):
    student.train()
    rloss = correct = total = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            s_out = student(images)
            loss = criterion(s_out, labels)
        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        rloss += loss.item()
        _, p = s_out.max(1); correct += p.eq(labels).sum().item(); total += labels.size(0)
    return rloss / len(dataloader), 100.0 * correct / total


def run_single(config_name, student_name, k, seed, device,
               cached_base, cached_ta, train_loader, test_loader,
               num_classes, experiment_type, **kwargs):
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

    if experiment_type == "label_smoothing":
        ls_eps = kwargs.get("ls_epsilon", 0.1)
        criterion = nn.CrossEntropyLoss(label_smoothing=ls_eps)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0; best_state = None; no_improve = 0

    for epoch in range(EPOCHS):
        if experiment_type == "label_smoothing":
            loss, tacc = train_label_smoothing(
                model, train_loader, optimizer, criterion, device, scaler)
        else:
            eq = kwargs.get("easy_quantile", 0.33)
            hq = kwargs.get("hard_quantile", 0.67)
            alo = kwargs.get("alpha_lo", 0.1)
            ahi = kwargs.get("alpha_hi", 0.7)
            loss, tacc = train_akr_cached(
                model, cached_base, cached_ta, train_loader,
                optimizer, criterion, device, scaler,
                easy_quantile=eq, hard_quantile=hq,
                alpha_lo=alo, alpha_hi=ahi)
        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            val = evaluate_model(model, test_loader, device)
            vacc = val["accuracy"]
            if vacc > best_acc:
                best_acc = vacc
                best_state = {k_: v.clone() for k_, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5
            if (epoch + 1) % 20 == 0 or epoch == EPOCHS - 1:
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
        "experiment": experiment_type, "config": config_name,
        "student": student_name, "k": k, "seed": seed,
        "params": params, "accuracy": final["accuracy"],
        "f1_macro": final["f1_macro"], "ece": final["ece"],
        **{k_: v for k_, v in kwargs.items()},
        "timestamp": datetime.now().isoformat(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",
                        choices=["threshold", "alpha", "label_smoothing", "all"],
                        required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--k-values", nargs="+", type=int, default=[28, 6])
    parser.add_argument("--students", nargs="+",
                        default=["WasteNet-Micro", "WasteNet-Small"])
    parser.add_argument("--output-suffix", type=str, default="",
                        help="Suffix for output file to avoid conflicts in parallel runs")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.backends.cudnn.benchmark = True
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    experiments = (["threshold", "alpha", "label_smoothing"]
                   if args.experiment == "all"
                   else [args.experiment])

    for exp_name in experiments:
        sfx = f"_{args.output_suffix}" if args.output_suffix else ""
        results_file = RESULTS_DIR / f"sensitivity_{exp_name}{sfx}.json"

        def _load_results():
            return json.load(open(results_file)) if results_file.exists() else []

        all_results = _load_results()

        def is_done(cfg, s, kv, sd):
            fresh = _load_results()
            return any(r["config"]==cfg and r["student"]==s and r["k"]==kv and r["seed"]==sd
                       for r in fresh)

        if exp_name == "threshold":
            configs = THRESHOLD_CONFIGS
        elif exp_name == "alpha":
            configs = ALPHA_BOUND_CONFIGS
        else:
            configs = {"ls00": 0.0, "ls01": 0.1, "ls02": 0.2, "ls03": 0.3}

        total_configs = len(configs) * len(args.students) * len(args.k_values) * len(SEEDS)
        done = sum(1 for c in configs for s in args.students
                   for kv in args.k_values for sd in SEEDS if is_done(c, s, kv, sd))

        print(f"\n{'='*70}")
        print(f"  AKR Sensitivity: {exp_name} (cached logits)")
        print(f"  Configs: {list(configs.keys())}")
        print(f"  Students: {args.students}, K: {args.k_values}")
        print(f"  Total: {total_configs}, Done: {done}, Remaining: {total_configs - done}")
        print(f"  GPU: {args.gpu}")
        print(f"{'='*70}\n")

        for kv in args.k_values:
            cfg = K_CONFIGS[kv]
            data_dir = Path(os.path.expanduser(cfg["data_dir"]))
            teacher_dir = OUTPUT_ROOT / cfg["teacher_dir"]
            cache_dir = teacher_dir / "cache"
            train_dir = data_dir / "train"
            num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])

            cached_base = torch.load(cache_dir / "base_train_logits.pt",
                                     map_location="cpu", weights_only=True)
            cached_ta = torch.load(cache_dir / "ta_train_logits.pt",
                                   map_location="cpu", weights_only=True)
            print(f"  K={kv}: loaded cached logits "
                  f"(base: {cached_base.shape}, ta: {cached_ta.shape})")

            train_tf = get_transforms(96, is_train=True)
            test_tf = get_transforms(96, is_train=False)
            train_ds = IndexedImageFolder(str(train_dir), transform=train_tf)
            test_ds = ImageFolder(str(data_dir / "test"), transform=test_tf)

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

            for config_name, config_val in configs.items():
                for student_name in args.students:
                    for seed in SEEDS:
                        if is_done(config_name, student_name, kv, seed):
                            continue

                        print(f"\n  {config_name} | {student_name} | K={kv} | seed={seed}")

                        if exp_name == "threshold":
                            eq, hq = config_val
                            result = run_single(
                                config_name, student_name, kv, seed, device,
                                cached_base, cached_ta, train_loader, test_loader,
                                num_classes, "threshold",
                                easy_quantile=eq, hard_quantile=hq)
                        elif exp_name == "alpha":
                            alo, ahi = config_val
                            result = run_single(
                                config_name, student_name, kv, seed, device,
                                cached_base, cached_ta, train_loader, test_loader,
                                num_classes, "alpha",
                                alpha_lo=alo, alpha_hi=ahi)
                        else:
                            result = run_single(
                                config_name, student_name, kv, seed, device,
                                cached_base, cached_ta, train_loader, test_loader,
                                num_classes, "label_smoothing",
                                ls_epsilon=config_val)

                        all_results = _load_results()
                        all_results.append(result)
                        with open(results_file, "w") as f:
                            json.dump(all_results, f, indent=2)

            del cached_base, cached_ta
            torch.cuda.empty_cache()

        print(f"\n{'='*70}")
        print(f"  DONE {exp_name}! {len(all_results)} results in {results_file}")
        print(f"{'='*70}")


if __name__ == "__main__":
    main()
