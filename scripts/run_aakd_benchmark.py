"""
=============================================================================
AAKD Benchmark: Ability-Aware Knowledge Distillation (Xiong et al. 2023)
=============================================================================
10 seeds × 7 students × 7 waste datasets = 490 runs
Uses existing k-ablation infrastructure with AAKD added to speed_utils_v3.

AAKD mechanism:
  1. Knowledge sample selection: sort by student target-class probability,
     select top cos(val_accuracy) fraction (hardest samples first)
  2. Adaptive teacher switching: start with TA, switch to base teacher
     when CE loss drops below 60% of initial CE loss

Usage:  python run_aakd_benchmark.py --gpu 0
        python run_aakd_benchmark.py --gpu 0 --seeds 42 123 456
=============================================================================
"""

import os, sys, json, copy, random, argparse, fcntl
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_kd_framework import (
    WasteNetMicro, WasteNetSmall, create_mobilenetv3_small,
    STUDENT_CONFIGS, evaluate, count_params, get_transforms,
)
from speed_utils_v3 import (
    IndexedImageFolder,
    cache_teacher_logits,
    cache_teacher_test_logits,
    run_experiment_v3,
)

ALL_SEEDS = [5, 42, 99, 123, 456, 777, 789, 1024, 2026, 2048]
EPOCHS = 100
LR = 1e-3
BATCH_SIZE = 64
NUM_WORKERS = 4
TEMPERATURE = 4.0
ALPHA = 0.5

ALL_STUDENTS = [
    "WasteNet-Micro",
    "WasteNet-Small",
    "WN-64K",
    "WN-128K",
    "WN-256K",
    "WN-512K",
    "MobileNetV3-Small",
]

WIDTH_CONFIGS = {
    "WN-64K":  {"base": "WasteNetMicro", "width": 2.0, "input_size": 96},
    "WN-128K": {"base": "WasteNetMicro", "width": 2.8, "input_size": 96},
    "WN-256K": {"base": "WasteNetMicro", "width": 4.0, "input_size": 96},
    "WN-512K": {"base": "WasteNetMicro", "width": 5.6, "input_size": 96},
}

K_CONFIGS = {
    28: {"data_dir": "~/KD/wasterecycle/results_teacher_yolo11/warp_c_crops",
         "teacher_dir": "teachers_warpc_k28"},
    7:  {"data_dir": "~/KD/results_k_ablation/warpc_k7",
         "teacher_dir": "teachers_k7"},
    4:  {"data_dir": "~/KD/results_k_ablation/warpc_k4",
         "teacher_dir": "teachers_k4"},
    3:  {"data_dir": "~/KD/results_k_ablation/warpc_k3",
         "teacher_dir": "teachers_k3"},
    6:  {"data_dir": "~/KD/wasterecycle/data/trashnet",
         "teacher_dir": "teachers_trashnet6"},
    5:  {"data_dir": "~/KD/results_k_ablation/trashnet_k5",
         "teacher_dir": "teachers_trashnet5"},
    2:  {"data_dir": "~/KD/data/waste_k2/DATASET",
         "teacher_dir": "teachers_waste_k2"},
}


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


def load_teacher_model(teacher_dir, teacher_name, num_classes, device):
    teacher_dir = Path(teacher_dir)
    ckpt_path = teacher_dir / f"teacher_{teacher_name}.pt"
    if not ckpt_path.exists():
        for alt in [f"classifier_{teacher_name}.pt",
                    f"teacher_{teacher_name.replace('-', '_')}.pt"]:
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
        raise ValueError(f"Unknown teacher: {teacher_name}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    else:
        model.load_state_dict(state)
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_results(results_file):
    results_file = Path(results_file)
    if not results_file.exists():
        return {}
    with open(results_file, "r") as f:
        data = json.load(f)
    existing = {}
    for r in data:
        key = f"{r.get('k','')}_{r.get('seed','')}_{r.get('student','')}_{r.get('method','')}"
        if "f1_macro" in r:
            existing[key] = r
    return existing


def save_result(results_file, entry):
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    if results_file.exists():
        with open(results_file, "r") as f:
            data = json.load(f)
    else:
        data = []
    data.append(entry)
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS)
    parser.add_argument("--k-values", type=int, nargs="+",
                        default=[28, 7, 6, 5, 4, 3, 2])
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    seeds = args.seeds

    results_file = Path(os.path.expanduser(
        "~/KD/results_aakd_benchmark/aakd_benchmark.json"))
    results_file.parent.mkdir(parents=True, exist_ok=True)
    existing = load_results(results_file)

    total_runs = len(ALL_STUDENTS) * len(args.k_values) * len(seeds)
    run_count = 0
    new_runs = 0

    print(f"\n{'='*70}")
    print(f"  AAKD Benchmark (10-seed)")
    print(f"  Seeds: {seeds}, GPU: {args.gpu}")
    print(f"  K-values: {args.k_values}")
    print(f"  Students: {len(ALL_STUDENTS)}")
    print(f"  Total planned: {total_runs} runs")
    print(f"{'='*70}")

    for k in args.k_values:
        cfg = K_CONFIGS[k]
        data_dir = Path(os.path.expanduser(cfg["data_dir"]))
        teacher_base_dir = Path(os.path.expanduser(
            f"~/KD/results_k_ablation/{cfg['teacher_dir']}"))

        print(f"\n{'='*70}")
        print(f"  K={k}: {data_dir}")
        print(f"{'='*70}")

        base_teacher = load_teacher_model(
            teacher_base_dir, "convnext-tiny", k, device)
        ta_teacher = load_teacher_model(
            teacher_base_dir, "efficientnet-b0", k, device)
        print(f"  Teachers loaded: ConvNeXT-Tiny + EfficientNet-B0")

        cache_dir = teacher_base_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_base_logits = cache_teacher_logits(
            base_teacher, data_dir, cache_dir / "base_train_logits.pt", device)
        cached_ta_logits = cache_teacher_logits(
            ta_teacher, data_dir, cache_dir / "ta_train_logits.pt", device)
        teacher_test_logits = cache_teacher_test_logits(
            base_teacher, data_dir, cache_dir / "base_test_logits.pt", device)

        train_dir_path = data_dir / "train"
        class_dirs = sorted([d for d in os.listdir(train_dir_path)
                            if (train_dir_path / d).is_dir()])
        samples_per_class = {}
        for c in class_dirs:
            samples_per_class[c] = len(list((train_dir_path / c).iterdir()))
        total_train = sum(samples_per_class.values())
        max_c = max(samples_per_class.values())
        min_c = min(samples_per_class.values())
        imbalance_ratio = round(max_c / max(min_c, 1), 2)

        test_ds = ImageFolder(str(data_dir / "test"))
        test_labels = [s[1] for s in test_ds.samples]
        t_preds = teacher_test_logits.argmax(dim=1).numpy()
        teacher_acc = round(100.0 * float(
            np.mean(np.array(test_labels) == t_preds)), 2)
        print(f"  Teacher acc: {teacher_acc}%, IR: {imbalance_ratio}")

        for seed in seeds:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            for student_name in ALL_STUDENTS:
                run_count += 1
                key = f"{k}_{seed}_{student_name}_aakd"

                if key in existing:
                    print(f"  [{run_count}/{total_runs}] SKIP: K={k} s={seed} {student_name}")
                    continue

                print(f"\n  [{run_count}/{total_runs}] K={k} seed={seed} "
                      f"{student_name}/aakd")

                input_size = get_input_size(student_name)
                train_transform = get_transforms(input_size, is_train=True)
                test_transform = get_transforms(input_size, is_train=False)

                train_ds = IndexedImageFolder(
                    str(train_dir_path), transform=train_transform)
                test_ds_eval = ImageFolder(
                    str(data_dir / "test"), transform=test_transform)

                targets = train_ds.targets
                class_counts = [0] * len(train_ds.classes)
                for t in targets:
                    class_counts[t] += 1
                weights = [1.0 / class_counts[t] for t in targets]
                sampler = WeightedRandomSampler(weights, len(targets), replacement=True)

                train_loader = DataLoader(
                    train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                    num_workers=NUM_WORKERS, pin_memory=True)
                test_loader = DataLoader(
                    test_ds_eval, batch_size=BATCH_SIZE, shuffle=False,
                    num_workers=NUM_WORKERS, pin_memory=True)

                try:
                    student_model = create_student(student_name, k)
                    result, _ = run_experiment_v3(
                        f"aakd_k{k}_s{seed}_{student_name}",
                        student_name, student_model,
                        train_loader, test_loader, device,
                        epochs=EPOCHS, lr=LR,
                        teacher=base_teacher, ta_teacher=ta_teacher,
                        kd_method="aakd",
                        temperature=TEMPERATURE, alpha=ALPHA,
                        teacher_input_size=224,
                        cached_base_logits=cached_base_logits,
                        cached_ta_logits=cached_ta_logits,
                        teacher_test_logits=teacher_test_logits,
                        num_classes=k, n_train=total_train,
                    )

                    params = result["params"]
                    cdr = round(params / max(1, k - 1), 2)

                    entry = {
                        "k": k,
                        "seed": seed,
                        **result,
                        "cdr": cdr,
                        "num_classes": k,
                        "total_train": total_train,
                        "imbalance_ratio": imbalance_ratio,
                        "teacher_acc": teacher_acc,
                    }
                    save_result(results_file, entry)
                    existing[key] = entry
                    new_runs += 1

                except Exception as e:
                    import traceback
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                    continue

        del base_teacher, ta_teacher
        torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"  AAKD Benchmark DONE: {new_runs} new runs")
    print(f"  Results: {results_file}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
