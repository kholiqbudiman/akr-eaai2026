"""
=============================================================================
CDR K-Ablation v3: 13 Methods, 7 Students, 10 Seeds
=============================================================================
Extends v2 with 5 new KD methods + Label Smoothing control.

Methods (13):
  Logit-based:  none, vanilla, takd, dkd, nkd, ctkd, dist
  Adaptive:     akr
  Feature-based: fitnet, simkd, crd
  Self-distill: ban, cord
  Control:      label_smoothing

Writes to SAME result files as v2. Existing results auto-skipped via
is_done() check (requires f1_macro field).

Usage:
  # Run new methods only (old methods already done by v2)
  python run_k_ablation_v3.py --gpu 0 --methods nkd ctkd fitnet simkd crd label_smoothing

  # Run specific K values
  python run_k_ablation_v3.py --gpu 0 --k-values 3 4 5 --methods nkd ctkd

  # Run everything (skip already-done)
  python run_k_ablation_v3.py --gpu 0

=============================================================================
"""

import os
import sys
import json
import copy
import fcntl
import random
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_kd_framework import (
    WasteNetMicro, WasteNetSmall, create_mobilenetv3_small,
    STUDENT_CONFIGS, evaluate, count_params,
    get_transforms,
)
from speed_utils_v3 import (
    IndexedImageFolder,
    create_indexed_dataloaders,
    cache_teacher_logits,
    cache_teacher_test_logits,
    cache_teacher_features,
    run_experiment_v3,
    run_self_distill_v3,
    get_penultimate_dim,
)
from torch.utils.data import WeightedRandomSampler


# ===========================================================================
# Configuration
# ===========================================================================

SEEDS = [42, 123, 456]

ALL_METHODS = [
    "none", "vanilla", "takd", "dkd", "nkd", "ctkd", "dist",
    "akr", "fitnet", "simkd", "crd",
    "ban", "cord",
    "label_smoothing",
]

METHODS_TEACHER = ["vanilla", "takd", "dkd", "nkd", "ctkd", "dist",
                   "akr", "fitnet", "simkd", "crd"]
METHODS_SELFDISTILL = ["ban", "cord"]
METHODS_NO_TEACHER = ["none", "label_smoothing"]

ALL_STUDENTS = [
    "WasteNet-Micro",    # 14K
    "WasteNet-Small",    # 32K
    "WN-64K",            # 64K
    "WN-128K",           # 128K
    "WN-256K",           # 256K
    "WN-512K",           # 512K
    "MobileNetV3-Small", # 1.5M
]

WIDTH_CONFIGS = {
    "WN-64K":  {"base": "WasteNetMicro", "width": 2.0, "input_size": 96},
    "WN-128K": {"base": "WasteNetMicro", "width": 2.8, "input_size": 96},
    "WN-256K": {"base": "WasteNetMicro", "width": 4.0, "input_size": 96},
    "WN-512K": {"base": "WasteNetMicro", "width": 5.6, "input_size": 96},
}

# K-value to dataset config (must match v2 directory naming)
K_CONFIGS = {
    2:  {"results_dir": "results_waste_k2",   "data_dir_name": "waste_k2/DATASET",  "teacher_dir": "teachers_waste_k2"},
    3:  {"results_dir": "results_k3",         "data_dir_name": "warpc_k3",           "teacher_dir": "teachers_k3"},
    4:  {"results_dir": "results_k4",         "data_dir_name": "warpc_k4",           "teacher_dir": "teachers_k4"},
    5:  {"results_dir": "results_trashnet5",  "data_dir_name": "trashnet_k5",        "teacher_dir": "teachers_trashnet5"},
    6:  {"results_dir": "results_trashnet6",  "data_dir_name": None,                 "teacher_dir": "teachers_trashnet6"},
    7:  {"results_dir": "results_k7",         "data_dir_name": "warpc_k7",           "teacher_dir": "teachers_k7"},
    10: {"results_dir": "results_cifar10",    "data_dir_name": None,                 "teacher_dir": "teachers_cifar10"},
    28: {"results_dir": "results_warpc_k28",  "data_dir_name": None,                 "teacher_dir": "teachers_warpc_k28"},
}

# Hardcoded data_dir paths for datasets not under output_root
K_DATA_OVERRIDES = {
    2:  "~/KD/data/waste_k2/DATASET",
    6:  "~/KD/wasterecycle/data/trashnet",
    10: "~/KD/data/cifar10_images",
    28: "~/KD/wasterecycle/results_teacher_yolo11/warp_c_crops",
}


# ===========================================================================
# Student and input size utilities
# ===========================================================================

def create_student(student_name, num_classes):
    if student_name in STUDENT_CONFIGS:
        return STUDENT_CONFIGS[student_name]["factory"](num_classes)
    elif student_name in WIDTH_CONFIGS:
        wcfg = WIDTH_CONFIGS[student_name]
        if wcfg["base"] == "WasteNetMicro":
            return WasteNetMicro(num_classes, width_mult=wcfg["width"])
        else:
            return WasteNetSmall(num_classes, width_mult=wcfg["width"])
    raise ValueError(f"Unknown student: {student_name}")


def get_input_size(student_name):
    if student_name in STUDENT_CONFIGS:
        return STUDENT_CONFIGS[student_name]["input_size"]
    elif student_name in WIDTH_CONFIGS:
        return WIDTH_CONFIGS[student_name]["input_size"]
    return 96


# ===========================================================================
# Result file I/O with file locking
# ===========================================================================

def load_results_locked(results_file):
    """Load results.json with flock for concurrent worker safety."""
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    if not results_file.exists():
        results_file.write_text("[]")

    with open(results_file, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try:
            data = json.load(f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

    existing = {}
    for r in data:
        key = f"{r.get('seed','')}_{r.get('student','')}_{r.get('method','')}"
        if "f1_macro" in r:
            existing[key] = r
    return existing


def dedup_results_file(results_file):
    """Remove duplicate entries from results.json, keeping last per key."""
    results_file = Path(results_file)
    if not results_file.exists():
        return 0
    with open(results_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
            seen = {}
            for i, r in enumerate(data):
                key = f"{r.get('seed','')}_{r.get('student','')}_{r.get('method','')}"
                seen[key] = i
            unique = [data[i] for i in sorted(seen.values())]
            removed = len(data) - len(unique)
            if removed > 0:
                f.seek(0)
                f.truncate()
                json.dump(unique, f, indent=2)
                print(f"  Dedup: removed {removed} duplicates ({len(data)} -> {len(unique)})")
            return removed
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def claim_run(results_dir, key):
    """Atomically claim a run via exclusive file creation. Returns True if claimed."""
    claim_dir = Path(results_dir) / ".claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace("/", "_").replace(" ", "_")
    claim_file = claim_dir / f"{safe_key}.claim"
    try:
        fd = os.open(str(claim_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return True
    except FileExistsError:
        return False


def save_result_locked(results_file, entry):
    """Append one result entry with exclusive lock."""
    results_file = Path(results_file)
    with open(results_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f)
            data.append(entry)
            f.seek(0)
            f.truncate()
            json.dump(data, f, indent=2)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ===========================================================================
# Data loader creation (returns indexed for CRD, standard for others)
# ===========================================================================

def create_dataloaders_for_method(data_dir, input_size, batch_size, num_workers,
                                  method, n_train_out=None):
    """Create appropriate dataloaders based on method requirements."""
    data_root = Path(data_dir)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_transform = get_transforms(input_size, is_train=True)
    test_transform = get_transforms(input_size, is_train=False)

    needs_index = method in ("crd",)
    if needs_index:
        train_ds = IndexedImageFolder(str(train_dir), transform=train_transform)
    else:
        train_ds = ImageFolder(str(train_dir), transform=train_transform)

    test_ds = ImageFolder(str(test_dir), transform=test_transform)

    targets = train_ds.targets
    class_counts = [0] * len(train_ds.classes)
    for t in targets:
        class_counts[t] += 1
    weights = [1.0 / class_counts[t] for t in targets]
    sampler = WeightedRandomSampler(weights, len(targets), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    if n_train_out is not None:
        n_train_out.append(len(train_ds))

    return train_loader, test_loader, train_ds


# ===========================================================================
# Teacher loading
# ===========================================================================

def load_teacher_model(teacher_dir, teacher_name, num_classes, device):
    """Load a trained teacher checkpoint."""
    teacher_dir = Path(teacher_dir)
    ckpt_path = teacher_dir / f"teacher_{teacher_name}.pt"

    if not ckpt_path.exists():
        alt_names = [
            f"classifier_{teacher_name}.pt",
            f"teacher_{teacher_name.replace('-', '_')}.pt",
        ]
        for alt in alt_names:
            alt_path = teacher_dir / alt
            if alt_path.exists():
                ckpt_path = alt_path
                break

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Teacher not found: {ckpt_path}")

    if "convnext" in teacher_name.lower():
        model = models.convnext_tiny(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif "efficientnet" in teacher_name.lower():
        model = models.efficientnet_b0(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    else:
        raise ValueError(f"Unknown teacher architecture: {teacher_name}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)

    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


# ===========================================================================
# Main experiment loop
# ===========================================================================

def run_k_experiments(k, output_root, device, args):
    """Run all experiments for a single K value."""
    output_root = Path(output_root)
    cfg = K_CONFIGS[k]

    # Resolve data directory
    if k in K_DATA_OVERRIDES:
        data_dir = Path(os.path.expanduser(K_DATA_OVERRIDES[k]))
    else:
        data_dir = output_root / cfg["data_dir_name"]

    teacher_dir = output_root / cfg["teacher_dir"]
    results_dir = output_root / cfg["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.json"

    print(f"\n{'='*70}")
    print(f"  K={k}: {data_dir}")
    print(f"  Methods: {args.methods}")
    print(f"  Students: {len(ALL_STUDENTS)}, Seeds: {len(SEEDS)}")
    print(f"  Total: {len(ALL_STUDENTS) * len(args.methods) * len(SEEDS)}")
    print(f"{'='*70}")

    # ---- Load teachers ONCE per K ----
    base_teacher = None
    ta_teacher = None

    needs_teacher = any(m in METHODS_TEACHER for m in args.methods)
    if needs_teacher:
        print(f"\n  Loading teachers for K={k}...")
        base_teacher = load_teacher_model(teacher_dir, "convnext-tiny", k, device)
        ta_teacher = load_teacher_model(teacher_dir, "efficientnet-b0", k, device)
        print(f"  Teachers loaded: ConvNeXT-Tiny + EfficientNet-B0")

    # ---- Cache teacher logits ONCE per K ----
    cached_base_logits = None
    cached_ta_logits = None
    teacher_test_logits = None

    if base_teacher is not None:
        cache_dir = teacher_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)

        cached_base_logits = cache_teacher_logits(
            base_teacher, data_dir, cache_dir / "base_train_logits.pt", device)
        teacher_test_logits = cache_teacher_test_logits(
            base_teacher, data_dir, cache_dir / "base_test_logits.pt", device)

    if ta_teacher is not None and any(m in ("takd", "akr") for m in args.methods):
        cache_dir = teacher_dir / "cache"
        cached_ta_logits = cache_teacher_logits(
            ta_teacher, data_dir, cache_dir / "ta_train_logits.pt", device)

    # ---- Count training samples (for CRD memory bank) ----
    train_dir_path = Path(data_dir) / "train"
    n_train = sum(len(list((train_dir_path / c).iterdir()))
                  for c in os.listdir(train_dir_path)
                  if (train_dir_path / c).is_dir())
    print(f"  Training samples: {n_train}")

    # ---- Dataset metadata for result entries ----
    class_dirs = sorted([d for d in os.listdir(train_dir_path)
                        if (train_dir_path / d).is_dir()])
    samples_per_class = {}
    for c in class_dirs:
        samples_per_class[c] = len(list((train_dir_path / c).iterdir()))
    total_train = sum(samples_per_class.values())
    max_c = max(samples_per_class.values())
    min_c = min(samples_per_class.values())
    imbalance_ratio = round(max_c / max(min_c, 1), 2)

    teacher_acc = None
    if teacher_test_logits is not None:
        test_dir_path = Path(data_dir) / "test"
        test_ds = ImageFolder(str(test_dir_path))
        test_labels = [s[1] for s in test_ds.samples]
        t_preds = teacher_test_logits.argmax(dim=1).numpy()
        teacher_acc = round(100.0 * float(np.mean(np.array(test_labels) == t_preds)), 2)
        print(f"  Teacher accuracy: {teacher_acc}%")

    # ---- Main loop: seed → student → method ----
    # Dedup any existing duplicates from prior concurrent runs
    dedup_results_file(results_file)
    existing = load_results_locked(results_file)
    total = len(ALL_STUDENTS) * len(args.methods) * len(SEEDS)
    done = 0
    new_runs = 0

    for seed in SEEDS:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        for student_name in ALL_STUDENTS:
            input_size = get_input_size(student_name)
            loader_cache = {}

            for method in args.methods:
                key = f"{seed}_{student_name}_{method}"
                done += 1

                if key in existing:
                    if done % 20 == 0:
                        print(f"  [{done}/{total}] SKIP: {key[:60]}")
                    continue

                # Atomic claim to prevent duplicate runs across processes
                if not claim_run(results_dir, key):
                    print(f"  [{done}/{total}] CLAIMED: {key[:60]}")
                    continue

                # Create dataloaders (cached per student+method_type)
                loader_key = f"{input_size}_{method in ('crd',)}"
                if loader_key not in loader_cache:
                    n_out = []
                    tl, vl, tds = create_dataloaders_for_method(
                        data_dir, input_size, args.batch_size, args.workers,
                        method, n_out)
                    loader_cache[loader_key] = (tl, vl, tds, n_out[0] if n_out else n_train)
                train_loader, test_loader, train_ds, n_data = loader_cache[loader_key]

                actual_nc = len(train_ds.classes)
                assert actual_nc == k, f"Expected {k} classes, got {actual_nc}"

                print(f"\n  [{done}/{total}] K={k} seed={seed} "
                      f"{student_name}/{method}")

                try:
                    if method in METHODS_SELFDISTILL:
                        student_factory = lambda nc, sn=student_name: create_student(sn, nc)
                        result, _ = run_self_distill_v3(
                            f"k{k}_s{seed}_{method}_{student_name}",
                            student_name, student_factory, k,
                            train_loader, test_loader, device,
                            epochs=args.epochs, lr=args.lr, method=method,
                            temperature=2.0, alpha=0.5, entropy_gate=0.3,
                            teacher_test_logits=teacher_test_logits,
                        )
                    else:
                        student_model = create_student(student_name, k)
                        result, _ = run_experiment_v3(
                            f"k{k}_s{seed}_{method}_{student_name}",
                            student_name, student_model,
                            train_loader, test_loader, device,
                            epochs=args.epochs, lr=args.lr,
                            teacher=base_teacher, ta_teacher=ta_teacher,
                            kd_method=method,
                            temperature=args.temperature, alpha=args.alpha,
                            teacher_input_size=224,
                            cached_base_logits=cached_base_logits,
                            cached_ta_logits=cached_ta_logits,
                            teacher_test_logits=teacher_test_logits,
                            num_classes=k, n_train=n_data,
                        )

                    # Enrich with CDR metadata
                    params = result["params"]
                    cdr = round(params / max(1, k - 1), 2)
                    capacity_ratio = round(params / 28_000_000, 6)  # vs ConvNeXT-Tiny

                    entry = {
                        "k": k,
                        "seed": seed,
                        **result,
                        "cdr": cdr,
                        "capacity_ratio": capacity_ratio,
                        "num_classes": k,
                        "total_train": total_train,
                        "imbalance_ratio": imbalance_ratio,
                        "samples_per_class": samples_per_class,
                        "teacher_name": "convnext-tiny",
                        "teacher_params": 28_000_000,
                        "teacher_acc": teacher_acc,
                    }

                    save_result_locked(results_file, entry)
                    existing[key] = entry
                    new_runs += 1

                except Exception as e:
                    print(f"  ERROR: {key}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

    print(f"\n  K={k} complete: {new_runs} new runs, {len(existing)} total results")
    print(f"  Results saved: {results_file}")


# ===========================================================================
# Entry point
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description="CDR K-Ablation v3 (13 methods)")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output-root", default="~/KD/results_k_ablation")
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[2, 3, 4, 5, 6, 7, 10, 28])
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS,
                        choices=ALL_METHODS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_root = Path(os.path.expanduser(args.output_root))
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"  CDR K-Ablation v3")
    print(f"  Device: {device}")
    print(f"  K values: {args.k_values}")
    print(f"  Methods ({len(args.methods)}): {args.methods}")
    print(f"  Students: {len(ALL_STUDENTS)}")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Total runs per K: {len(ALL_STUDENTS) * len(args.methods) * len(SEEDS)}")
    print(f"  Output: {output_root}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    for k in args.k_values:
        if k not in K_CONFIGS:
            print(f"  Skipping K={k}: not configured")
            continue
        run_k_experiments(k, output_root, device, args)

    print(f"\n  All done! {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
