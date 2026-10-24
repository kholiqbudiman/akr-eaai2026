"""
CIFAR-10 experiments on new 4070 (LabRPL).
Seeds: [789, 1024, 2048, 5, 99] — non-overlapping with A100 [42, 123, 456].
"""
import os, sys, json, copy, fcntl, random, argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from train_kd_framework import (
    WasteNetMicro, WasteNetSmall, create_mobilenetv3_small,
    STUDENT_CONFIGS, evaluate, count_params, get_transforms,
)
from speed_utils_v3 import (
    IndexedImageFolder, create_indexed_dataloaders,
    cache_teacher_logits, cache_teacher_test_logits,
    cache_teacher_features, run_experiment_v3, run_self_distill_v3,
    get_penultimate_dim,
)

SEEDS = [2048, 5, 99, 777, 2026]

ALL_METHODS = [
    "none", "vanilla", "takd", "dkd", "nkd", "ctkd", "dist",
    "akr", "fitnet", "simkd", "crd", "ban", "cord", "label_smoothing",
]
METHODS_TEACHER = ["vanilla", "takd", "dkd", "nkd", "ctkd", "dist",
                   "akr", "fitnet", "simkd", "crd"]
METHODS_SELFDISTILL = ["ban", "cord"]
METHODS_NO_TEACHER = ["none", "label_smoothing"]

ALL_STUDENTS = [
    "WasteNet-Micro", "WasteNet-Small", "WN-64K", "WN-128K",
    "WN-256K", "WN-512K", "MobileNetV3-Small",
]
WIDTH_CONFIGS = {
    "WN-64K":  {"base": "WasteNetMicro", "width": 2.0, "input_size": 96},
    "WN-128K": {"base": "WasteNetMicro", "width": 2.8, "input_size": 96},
    "WN-256K": {"base": "WasteNetMicro", "width": 4.0, "input_size": 96},
    "WN-512K": {"base": "WasteNetMicro", "width": 5.6, "input_size": 96},
}

DATA_DIR = os.path.expanduser("~/KD/results_k_ablation/cifar10_imgfolder")
TEACHER_DIR = os.path.expanduser("~/KD/results_k_ablation/teachers_cifar10")
RESULTS_DIR = os.path.expanduser("~/KD/results_k_ablation/results_cifar10")
K = 10

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

def load_results_locked(results_file):
    results_file = Path(results_file)
    results_file.parent.mkdir(parents=True, exist_ok=True)
    if not results_file.exists():
        results_file.write_text("[]")
    with open(results_file, "r") as f:
        fcntl.flock(f, fcntl.LOCK_SH)
        try: data = json.load(f)
        finally: fcntl.flock(f, fcntl.LOCK_UN)
    existing = {}
    for r in data:
        key = f"{r.get('seed','')}_{r.get('student','')}_{r.get('method','')}";
        if "f1_macro" in r: existing[key] = r
    return existing

def save_result_locked(results_file, entry):
    results_file = Path(results_file)
    with open(results_file, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            data = json.load(f); data.append(entry)
            f.seek(0); f.truncate(); json.dump(data, f, indent=2)
        finally: fcntl.flock(f, fcntl.LOCK_UN)

def claim_run(results_dir, key):
    claim_dir = Path(results_dir) / ".claims"
    claim_dir.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace("/", "_").replace(" ", "_")
    claim_file = claim_dir / f"{safe_key}.claim"
    try:
        fd = os.open(str(claim_file), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd); return True
    except FileExistsError: return False

def load_teacher_model(teacher_dir, teacher_name, num_classes, device):
    teacher_dir = Path(teacher_dir)
    ckpt_path = teacher_dir / f"teacher_{teacher_name}.pt"
    if "convnext" in teacher_name.lower():
        model = models.convnext_tiny(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    elif "efficientnet" in teacher_name.lower():
        model = models.efficientnet_b0(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if "model_state_dict" in state: model.load_state_dict(state["model_state_dict"])
    else: model.load_state_dict(state)
    model = model.to(device).eval()
    for p in model.parameters(): p.requires_grad = False
    return model

def create_dataloaders(data_dir, input_size, batch_size, num_workers, method):
    train_dir = Path(data_dir) / "train"
    test_dir = Path(data_dir) / "test"
    train_tf = get_transforms(input_size, is_train=True)
    test_tf = get_transforms(input_size, is_train=False)
    if method == "crd":
        train_ds = IndexedImageFolder(str(train_dir), transform=train_tf)
    else:
        train_ds = ImageFolder(str(train_dir), transform=train_tf)
    test_ds = ImageFolder(str(test_dir), transform=test_tf)
    targets = train_ds.targets
    cc = [0]*len(train_ds.classes)
    for t in targets: cc[t] += 1
    weights = [1.0/cc[t] for t in targets]
    sampler = WeightedRandomSampler(weights, len(targets), replacement=True)
    tl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                    num_workers=num_workers, pin_memory=True)
    vl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=True)
    return tl, vl, train_ds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}")
    results_dir = Path(RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "results.json"

    print(f"\n{'='*70}")
    print(f"  CIFAR-10 on new 4070 (LabRPL)")
    print(f"  Seeds: {SEEDS}")
    print(f"  Methods: {args.methods}")
    print(f"  Total: {len(ALL_STUDENTS) * len(args.methods) * len(SEEDS)}")
    print(f"  Started: {datetime.now()}")
    print(f"{'='*70}")

    # Load teachers
    base_teacher = load_teacher_model(TEACHER_DIR, "convnext-tiny", K, device)
    ta_teacher = load_teacher_model(TEACHER_DIR, "efficientnet-b0", K, device)

    # Cache logits
    cache_dir = Path(TEACHER_DIR) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_base = cache_teacher_logits(base_teacher, DATA_DIR, cache_dir/"base_train_logits.pt", device)
    teacher_test = cache_teacher_test_logits(base_teacher, DATA_DIR, cache_dir/"base_test_logits.pt", device)
    cached_ta = cache_teacher_logits(ta_teacher, DATA_DIR, cache_dir/"ta_train_logits.pt", device)

    # Dataset metadata
    train_dir = Path(DATA_DIR)/"train"
    class_dirs = sorted([d for d in os.listdir(train_dir) if (train_dir/d).is_dir()])
    spc = {c: len(list((train_dir/c).iterdir())) for c in class_dirs}
    total_train = sum(spc.values())
    ir = round(max(spc.values()) / max(min(spc.values()), 1), 2)

    test_ds = ImageFolder(str(Path(DATA_DIR)/"test"))
    t_preds = teacher_test.argmax(dim=1).numpy()
    t_labels = [s[1] for s in test_ds.samples]
    teacher_acc = round(100.0*float(np.mean(np.array(t_labels)==t_preds)), 2)
    n_train = total_train
    print(f"  Teacher acc: {teacher_acc}%, IR: {ir}, Train: {total_train}")

    existing = load_results_locked(results_file)
    total = len(ALL_STUDENTS) * len(args.methods) * len(SEEDS)
    done = 0; new_runs = 0

    for seed in SEEDS:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for student_name in ALL_STUDENTS:
            input_size = get_input_size(student_name)
            loader_cache = {}
            for method in args.methods:
                key = f"{seed}_{student_name}_{method}"; done += 1
                if key in existing:
                    if done % 20 == 0: print(f"  [{done}/{total}] SKIP {key[:50]}")
                    continue
                if not claim_run(RESULTS_DIR, key):
                    print(f"  [{done}/{total}] CLAIMED {key[:50]}"); continue

                lk = f"{input_size}_{method=='crd'}"
                if lk not in loader_cache:
                    tl, vl, tds = create_dataloaders(DATA_DIR, input_size, args.batch_size, args.workers, method)
                    loader_cache[lk] = (tl, vl, tds)
                train_loader, test_loader, train_ds = loader_cache[lk]

                print(f"\n  [{done}/{total}] K={K} seed={seed} {student_name}/{method}")
                try:
                    if method in METHODS_SELFDISTILL:
                        factory = lambda nc, sn=student_name: create_student(sn, nc)
                        result, _ = run_self_distill_v3(
                            f"k{K}_s{seed}_{method}_{student_name}",
                            student_name, factory, K, train_loader, test_loader, device,
                            epochs=args.epochs, lr=1e-3, method=method,
                            temperature=2.0, alpha=0.5, entropy_gate=0.3,
                            teacher_test_logits=teacher_test,
                        )
                    else:
                        student_model = create_student(student_name, K)
                        result, _ = run_experiment_v3(
                            f"k{K}_s{seed}_{method}_{student_name}",
                            student_name, student_model, train_loader, test_loader, device,
                            epochs=args.epochs, lr=1e-3,
                            teacher=base_teacher, ta_teacher=ta_teacher,
                            kd_method=method, temperature=4.0, alpha=0.5,
                            teacher_input_size=224,
                            cached_base_logits=cached_base, cached_ta_logits=cached_ta,
                            teacher_test_logits=teacher_test,
                            num_classes=K, n_train=n_train,
                        )
                    params = result["params"]
                    entry = {"k": K, "dataset": "cifar10", "seed": seed, **result,
                             "cdr": round(params/max(1,K-1),2),
                             "capacity_ratio": round(params/28_000_000,6),
                             "num_classes": K, "total_train": total_train,
                             "imbalance_ratio": ir, "samples_per_class": spc,
                             "teacher_name": "convnext-tiny", "teacher_params": 28_000_000,
                             "teacher_acc": teacher_acc}
                    save_result_locked(results_file, entry)
                    existing[key] = entry; new_runs += 1
                except Exception as e:
                    print(f"  ERROR: {key}: {e}")
                    import traceback; traceback.print_exc()

    print(f"\n  Done! {new_runs} new runs. {datetime.now()}")

if __name__ == "__main__":
    main()
