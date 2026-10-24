"""
Speed utilities v3 for K-ablation experiments.
Extends v2 with 5 new KD methods: NKD, CTKD, FitNet, SimKD, CRD.
Also adds Label Smoothing (LS) as a control method.

All training functions use AMP + early stopping + comprehensive metrics.
"""
import os
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.autograd import Function

from train_kd_framework import (
    resize_for_teacher, kd_loss, dkd_loss, evaluate,
    count_params, count_flops, measure_latency,
    STUDENT_CONFIGS, get_transforms,
)

PATIENCE = 15


# ===========================================================================
# Dataset utilities (from v2, unchanged)
# ===========================================================================

class IndexedImageFolder(ImageFolder):
    """ImageFolder that also returns sample index for cached logits lookup."""
    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        return img, label, index


def create_indexed_dataloaders(data_root, input_size=224, batch_size=32, num_workers=4):
    data_root = Path(data_root)
    train_dir = data_root / "train"
    test_dir = data_root / "test"

    train_transform = get_transforms(input_size, is_train=True)
    test_transform = get_transforms(input_size, is_train=False)

    train_ds = IndexedImageFolder(str(train_dir), transform=train_transform)
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
    return train_loader, test_loader, train_ds, test_ds


# ===========================================================================
# Teacher logit / feature caching
# ===========================================================================

def cache_teacher_logits(teacher, data_dir, cache_path, device):
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  Loaded cached logits: {cache_path} ({cached.shape[0]} samples)")
        return cached

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(str(Path(data_dir) / "train"), transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4,
                        pin_memory=True)

    teacher.eval()
    all_logits = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, _ in loader:
            logits = teacher(images.to(device))
            all_logits.append(logits.cpu())

    cached = torch.cat(all_logits, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_path)
    print(f"  Cached teacher logits: {cache_path} ({cached.shape[0]} samples, {cached.shape[1]} classes)")
    return cached


def cache_teacher_test_logits(teacher, data_dir, cache_path, device):
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  Loaded cached test logits: {cache_path} ({cached.shape[0]} samples)")
        return cached

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(str(Path(data_dir) / "test"), transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4,
                        pin_memory=True)

    teacher.eval()
    all_logits = []
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, _ in loader:
            logits = teacher(images.to(device))
            all_logits.append(logits.cpu())

    cached = torch.cat(all_logits, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_path)
    print(f"  Cached teacher test logits: {cache_path} ({cached.shape[0]} samples)")
    return cached


def cache_teacher_features(teacher, data_dir, cache_path, device, layer_name="avgpool"):
    """Cache teacher penultimate features (after avgpool, before classifier)."""
    cache_path = Path(cache_path)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        print(f"  Loaded cached teacher features: {cache_path} ({cached.shape})")
        return cached

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(str(Path(data_dir) / "train"), transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4,
                        pin_memory=True)

    features_list = []
    hook_output = {}

    def hook_fn(module, input, output):
        hook_output["feat"] = output

    # ConvNeXT-Tiny: features → avgpool → classifier
    # We hook the avgpool to get penultimate features
    handle = teacher.avgpool.register_forward_hook(hook_fn)

    teacher.eval()
    with torch.no_grad(), torch.cuda.amp.autocast():
        for images, _ in loader:
            _ = teacher(images.to(device))
            feat = hook_output["feat"]
            if feat.dim() == 4:
                feat = feat.flatten(1)
            features_list.append(feat.cpu().float())

    handle.remove()

    cached = torch.cat(features_list, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cached, cache_path)
    print(f"  Cached teacher features: {cache_path} ({cached.shape})")
    return cached


# ===========================================================================
# Comprehensive metrics (from v2, unchanged)
# ===========================================================================

def compute_ece(logits, labels, n_bins=15):
    if isinstance(logits, np.ndarray):
        logits = torch.tensor(logits, dtype=torch.float32)
    if isinstance(labels, np.ndarray):
        labels = torch.tensor(labels, dtype=torch.long)

    probs = F.softmax(logits, dim=1)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels).float()

    ece = 0.0
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean().item()
            avg_acc = accuracies[mask].mean().item()
            ece += (mask.sum().item() / len(labels)) * abs(avg_conf - avg_acc)
    return round(ece, 6)


def compute_comprehensive_metrics(eval_result, teacher_test_logits=None):
    preds = eval_result["predictions"]
    labels = eval_result["labels"]
    logits = eval_result["logits"]

    f1 = round(f1_score(labels, preds, average="macro", zero_division=0) * 100, 4)
    prec = round(precision_score(labels, preds, average="macro", zero_division=0) * 100, 4)
    rec = round(recall_score(labels, preds, average="macro", zero_division=0) * 100, 4)
    f1_pc = [round(x * 100, 2) for x in
             f1_score(labels, preds, average=None, zero_division=0).tolist()]

    probs = F.softmax(logits, dim=1)
    confidences = probs.max(dim=1).values
    conf_mean = round(confidences.mean().item() * 100, 4)
    conf_std = round(confidences.std().item() * 100, 4)

    ece = compute_ece(logits, labels)

    labels_t = torch.tensor(labels, dtype=torch.long)
    log_probs = F.log_softmax(logits, dim=1)
    nll = F.nll_loss(log_probs, labels_t).item()

    metrics = {
        "f1_macro": f1,
        "precision_macro": prec,
        "recall_macro": rec,
        "f1_per_class": f1_pc,
        "ece": ece,
        "nll": round(nll, 6),
        "confidence_mean": conf_mean,
        "confidence_std": conf_std,
    }

    if teacher_test_logits is not None:
        t_preds = teacher_test_logits.argmax(dim=1).numpy()
        agreement = round(100.0 * float(np.mean(preds == t_preds)), 4)
        t_probs = F.softmax(teacher_test_logits, dim=1)
        s_log_probs = F.log_softmax(logits, dim=1)
        kl = F.kl_div(s_log_probs, t_probs, reduction="batchmean").item()
        metrics["teacher_agreement"] = agreement
        metrics["kl_div_test"] = round(kl, 6)

    return metrics


# ===========================================================================
# Existing training epochs (from v2, unchanged) — CE, Vanilla, DKD, AKR, DIST
# ===========================================================================

def train_one_epoch_fast(model, dataloader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = correct = total = 0
    for batch in dataloader:
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


def train_one_epoch_kd_fast(student, teacher, dataloader, optimizer, criterion,
                            device, scaler, temperature=4.0, alpha=0.5,
                            teacher_input_size=224, cached_logits=None):
    student.train()
    if teacher is not None:
        teacher.eval()
    running_loss = correct = total = 0

    for batch in dataloader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch[0], batch[1]
            indices = None
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            student_out = student(images)
            if cached_logits is not None and indices is not None:
                teacher_out = cached_logits[indices].to(device)
            else:
                teacher_images = resize_for_teacher(images, teacher_input_size)
                with torch.no_grad():
                    teacher_out = teacher(teacher_images)
            loss_ce = criterion(student_out, labels)
            loss_kd = kd_loss(student_out, teacher_out, temperature)
            loss = alpha * loss_kd + (1 - alpha) * loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


def train_one_epoch_dkd_fast(student, teacher, dataloader, optimizer, criterion,
                             device, scaler, temperature=4.0, alpha=0.5,
                             teacher_input_size=224, cached_logits=None,
                             alpha_tc=1.0, beta_nc=8.0):
    student.train()
    if teacher is not None:
        teacher.eval()
    running_loss = correct = total = 0

    for batch in dataloader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch[0], batch[1]
            indices = None
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            student_out = student(images)
            if cached_logits is not None and indices is not None:
                teacher_out = cached_logits[indices].to(device)
            else:
                teacher_images = resize_for_teacher(images, teacher_input_size)
                with torch.no_grad():
                    teacher_out = teacher(teacher_images)
            loss_ce = criterion(student_out, labels)
            loss_kd = dkd_loss(student_out, teacher_out, labels,
                               temperature=temperature,
                               alpha_tc=alpha_tc, beta_nc=beta_nc)
            loss = alpha * loss_kd + (1 - alpha) * loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


def train_one_epoch_akr_fast(student, base_teacher, ta_teacher, dataloader,
                             optimizer, criterion, device, scaler,
                             temperature=4.0, alpha_kd=0.5,
                             teacher_input_size=224, cached_base_logits=None):
    student.train()
    base_teacher.eval()
    ta_teacher.eval()
    running_loss = correct = total = 0

    for batch in dataloader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch[0], batch[1]
            indices = None
        batch_size = images.size(0)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            student_out = student(images)
            teacher_images = resize_for_teacher(images, teacher_input_size)

            if cached_base_logits is not None and indices is not None:
                bt_logits = cached_base_logits[indices].to(device)
            else:
                with torch.no_grad():
                    bt_logits = base_teacher(teacher_images)

            with torch.no_grad():
                bt_probs = F.softmax(bt_logits / 1.0, dim=1)
                top2_vals = bt_probs.topk(2, dim=1).values
                margins = top2_vals[:, 0] - top2_vals[:, 1]
                d_tensor = 1.0 - margins

            easy_thresh_val = d_tensor.quantile(0.33).item()
            hard_thresh_val = d_tensor.quantile(0.67).item()
            easy_mask = d_tensor < easy_thresh_val
            hard_mask = d_tensor >= hard_thresh_val
            std_mask = ~easy_mask & ~hard_mask

            loss_ce = criterion(student_out, labels)
            alphas = d_tensor.clamp(0.1, 0.7)
            loss_kd = torch.tensor(0.0, device=device)

            if hard_mask.any():
                hard_kd = kd_loss(student_out[hard_mask], bt_logits[hard_mask], temperature)
                loss_kd = loss_kd + alphas[hard_mask].mean() * hard_kd

            if std_mask.any():
                with torch.no_grad():
                    ta_logits = ta_teacher(teacher_images[std_mask])
                std_kd = kd_loss(student_out[std_mask], ta_logits, temperature)
                loss_kd = loss_kd + alphas[std_mask].mean() * std_kd

            n_distilled = hard_mask.sum() + std_mask.sum()
            if n_distilled > 0:
                kd_weight = alpha_kd * (n_distilled.float() / batch_size)
                loss = kd_weight * loss_kd + (1.0 - kd_weight) * loss_ce
            else:
                loss = loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


def pearson_correlation(x, y, eps=1e-8):
    x_c = x - x.mean(dim=-1, keepdim=True)
    y_c = y - y.mean(dim=-1, keepdim=True)
    x_std = torch.sqrt((x_c ** 2).sum(dim=-1, keepdim=True) + eps)
    y_std = torch.sqrt((y_c ** 2).sum(dim=-1, keepdim=True) + eps)
    return ((x_c * y_c).sum(dim=-1, keepdim=True) / (x_std * y_std + eps)).squeeze(-1)


def dist_loss(student_logits, teacher_logits, temperature=4.0, beta=2.0, gamma=2.0):
    p_t = F.softmax(teacher_logits / temperature, dim=-1)
    p_s = F.softmax(student_logits / temperature, dim=-1)
    inter_loss = 1.0 - pearson_correlation(p_s, p_t).mean()
    intra_loss = 1.0 - pearson_correlation(p_s.t(), p_t.t()).mean()
    return beta * inter_loss + gamma * intra_loss


def train_one_epoch_dist_fast(student, teacher, dataloader, optimizer, criterion,
                              device, scaler, temperature=4.0, alpha=1.0,
                              beta=2.0, gamma=2.0, teacher_input_size=224,
                              cached_logits=None):
    student.train()
    if teacher is not None:
        teacher.eval()
    running_loss = correct = total = 0

    for batch in dataloader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch[0], batch[1]
            indices = None
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            student_out = student(images)
            if cached_logits is not None and indices is not None:
                teacher_out = cached_logits[indices].to(device)
            else:
                teacher_images = resize_for_teacher(images, teacher_input_size)
                with torch.no_grad():
                    teacher_out = teacher(teacher_images)
            loss_ce = criterion(student_out, labels)
            loss_d = dist_loss(student_out, teacher_out, temperature, beta, gamma)
            loss = alpha * loss_ce + loss_d

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


# ===========================================================================
# NEW METHOD 1: NKD (Yang et al., ICCV 2023)
# "From Knowledge Distillation to Self-Knowledge Distillation:
#  A Unified Approach with Normalized Loss and Customized Soft Labels"
# Reference: https://github.com/yzd-v/cls_KD
# ===========================================================================

def nkd_loss(logit_s, logit_t, gt_label, temp=1.0, gamma=1.5):
    """Normalized Knowledge Distillation loss."""
    N, c = logit_s.shape
    label = gt_label.view(-1, 1)

    s_i = F.log_softmax(logit_s, dim=1)
    t_i = F.softmax(logit_t, dim=1)

    s_t = torch.gather(s_i, 1, label)
    t_t = torch.gather(t_i, 1, label).detach()
    loss_t = -(t_t * s_t).mean()

    mask = torch.ones_like(logit_s).scatter_(1, label, 0).bool()
    logit_s_masked = logit_s[mask].reshape(N, -1)
    logit_t_masked = logit_t[mask].reshape(N, -1)

    S_i = F.log_softmax(logit_s_masked / temp, dim=1)
    T_i = F.softmax(logit_t_masked / temp, dim=1)
    loss_non = -(T_i * S_i).sum(dim=1).mean()
    loss_non = gamma * (temp ** 2) * loss_non

    return loss_t + loss_non


def train_one_epoch_nkd_fast(student, teacher, dataloader, optimizer, criterion,
                             device, scaler, alpha=0.5,
                             nkd_temp=1.0, nkd_gamma=1.5,
                             teacher_input_size=224, cached_logits=None):
    student.train()
    if teacher is not None:
        teacher.eval()
    running_loss = correct = total = 0

    for batch in dataloader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch[0], batch[1]
            indices = None
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            student_out = student(images)
            if cached_logits is not None and indices is not None:
                teacher_out = cached_logits[indices].to(device)
            else:
                teacher_images = resize_for_teacher(images, teacher_input_size)
                with torch.no_grad():
                    teacher_out = teacher(teacher_images)
            loss_ce = criterion(student_out, labels)
            loss_nkd = nkd_loss(student_out, teacher_out, labels,
                                temp=nkd_temp, gamma=nkd_gamma)
            loss = alpha * loss_nkd + (1 - alpha) * loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


# ===========================================================================
# NEW METHOD 2: CTKD (Li et al., AAAI 2023)
# "Curriculum Temperature for Knowledge Distillation"
# Reference: https://github.com/zhengli97/CTKD
# Global-T variant (zero architecture overhead)
# ===========================================================================

class GradientReversal(Function):
    @staticmethod
    def forward(ctx, x, lambda_val):
        ctx.lambda_val = lambda_val
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return ctx.lambda_val * grad_output, None


class CTKD_GlobalT(nn.Module):
    """Learnable global temperature with gradient reversal."""
    def __init__(self, t_start=1.0, t_end=20.0):
        super().__init__()
        self.t_start = t_start
        self.t_end = t_end
        self.raw_T = nn.Parameter(torch.zeros(1))

    def forward(self, lambda_val):
        reversed_T = GradientReversal.apply(self.raw_T, lambda_val)
        T = self.t_start + self.t_end * torch.sigmoid(reversed_T)
        return T


def ctkd_lambda_schedule(epoch, total_epochs, lambda_min=0.0, lambda_max=1.0,
                         warmup_loops=10):
    """Cosine lambda schedule for CTKD."""
    ratio = min(epoch / max(warmup_loops, 1), 1.0)
    cos_val = 0.5 * (1.0 + math.cos((1.0 + ratio) * math.pi))
    return -(lambda_min + (lambda_max - lambda_min) * cos_val)


def train_one_epoch_ctkd_fast(student, teacher, dataloader, optimizer, criterion,
                              device, scaler, global_t_module, epoch, total_epochs,
                              alpha=0.5, teacher_input_size=224, cached_logits=None):
    student.train()
    if teacher is not None:
        teacher.eval()
    running_loss = correct = total = 0

    lambda_val = ctkd_lambda_schedule(epoch, total_epochs)

    for batch in dataloader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch[0], batch[1]
            indices = None
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            student_out = student(images)
            if cached_logits is not None and indices is not None:
                teacher_out = cached_logits[indices].to(device)
            else:
                teacher_images = resize_for_teacher(images, teacher_input_size)
                with torch.no_grad():
                    teacher_out = teacher(teacher_images)

            T = global_t_module(lambda_val)
            loss_ce = criterion(student_out, labels)
            loss_kd = kd_loss(student_out, teacher_out, temperature=T.squeeze())
            loss = alpha * loss_kd + (1 - alpha) * loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


# ===========================================================================
# NEW METHOD 3: FitNet (Romero et al., ICLR 2015)
# "FitNets: Hints for Thin Deep Nets"
# Reference: https://github.com/HobbitLong/RepDistiller
# ===========================================================================

class FitNetRegressor(nn.Module):
    """Maps student feature to teacher feature space."""
    def __init__(self, s_channels, t_channels, s_spatial=None, t_spatial=None):
        super().__init__()
        layers = []
        if s_spatial is not None and t_spatial is not None and s_spatial != t_spatial:
            if s_spatial > t_spatial:
                k = s_spatial - t_spatial + 1
                layers.append(nn.Conv2d(s_channels, t_channels, kernel_size=k, stride=1, padding=0))
            else:
                layers.append(nn.Conv2d(s_channels, t_channels, kernel_size=1))
                layers.append(nn.Upsample(size=t_spatial, mode='bilinear', align_corners=False))
        else:
            layers.append(nn.Conv2d(s_channels, t_channels, kernel_size=1))
        layers.append(nn.BatchNorm2d(t_channels))
        layers.append(nn.ReLU(inplace=True))
        self.regressor = nn.Sequential(*layers)

    def forward(self, x):
        return self.regressor(x)


def get_feature_shapes(model, input_size, device, hook_layer_name="features"):
    """Run a dummy forward to determine intermediate feature shapes."""
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    features = {}

    def make_hook(name):
        def hook_fn(module, inp, out):
            features[name] = out.shape
        return hook_fn

    handles = []
    for name, module in model.named_modules():
        if name == hook_layer_name or name.startswith(hook_layer_name + "."):
            handles.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(dummy)

    for h in handles:
        h.remove()
    return features


class FeatureExtractor(nn.Module):
    """Wraps a model to also return intermediate features."""
    def __init__(self, model, layer_name):
        super().__init__()
        self.model = model
        self.layer_name = layer_name
        self._feature = None
        self._handle = None
        self._register_hook()

    def _register_hook(self):
        for name, module in self.model.named_modules():
            if name == self.layer_name:
                self._handle = module.register_forward_hook(self._hook_fn)
                return
        raise ValueError(f"Layer '{self.layer_name}' not found")

    def _hook_fn(self, module, inp, out):
        self._feature = out

    def forward(self, x):
        logits = self.model(x)
        return logits, self._feature

    def remove_hook(self):
        if self._handle:
            self._handle.remove()


def train_one_epoch_fitnet_fast(student_ext, teacher, dataloader, optimizer,
                                criterion, device, scaler, regressor,
                                alpha=0.5, beta_hint=100.0,
                                teacher_input_size=224,
                                teacher_hint_layer="features.5"):
    """FitNet: CE + hint loss (MSE between regressed student and teacher features)."""
    student_ext.model.train()
    regressor.train()
    if teacher is not None:
        teacher.eval()
    running_loss = correct = total = 0

    teacher_hook_out = {}
    def t_hook(module, inp, out):
        teacher_hook_out["feat"] = out

    t_handle = None
    for name, module in teacher.named_modules():
        if name == teacher_hint_layer:
            t_handle = module.register_forward_hook(t_hook)
            break

    for batch in dataloader:
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            s_logits, s_feat = student_ext(images)

            teacher_images = resize_for_teacher(images, teacher_input_size)
            with torch.no_grad():
                _ = teacher(teacher_images)
                t_feat = teacher_hook_out["feat"].detach()

            s_feat_reg = regressor(s_feat)
            if s_feat_reg.shape != t_feat.shape:
                s_feat_reg = F.adaptive_avg_pool2d(s_feat_reg, t_feat.shape[2:])

            loss_ce = criterion(s_logits, labels)
            loss_hint = F.mse_loss(s_feat_reg, t_feat)
            loss = (1 - alpha) * loss_ce + beta_hint * loss_hint

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = s_logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    if t_handle:
        t_handle.remove()
    return running_loss / len(dataloader), 100.0 * correct / total


# ===========================================================================
# NEW METHOD 4: SimKD (Chen et al., CVPR 2022)
# "Knowledge Distillation with the Reused Teacher Classifier"
# Reference: https://github.com/DefangChen/SimKD
# ===========================================================================

class SimKDProjector(nn.Module):
    """Bottleneck projector: maps student features to teacher feature space."""
    def __init__(self, s_dim, t_dim, factor=2):
        super().__init__()
        mid = max(t_dim // factor, 8)
        self.transfer = nn.Sequential(
            nn.Linear(s_dim, mid),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, mid),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, t_dim),
            nn.BatchNorm1d(t_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.transfer(x)


def _eval_simkd(student, projector, teacher_classifier, test_loader, device):
    """Evaluate SimKD using projector + teacher_classifier instead of student classifier."""
    student.eval()
    projector.eval()
    hook_out = {}
    def hook_fn(mod, inp, out):
        hook_out["feat"] = out
    if hasattr(student, 'pool'):
        handle = student.pool.register_forward_hook(hook_fn)
    elif hasattr(student, 'avgpool'):
        handle = student.avgpool.register_forward_hook(hook_fn)
    else:
        for name, mod in student.named_modules():
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                handle = mod.register_forward_hook(hook_fn)
                break
    correct = total = 0
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch[0].to(device), batch[1].to(device)
            _ = student(images)
            feat = hook_out["feat"]
            if feat.dim() == 4:
                feat = feat.flatten(1)
            logits = teacher_classifier(projector(feat))
            correct += logits.argmax(1).eq(labels).sum().item()
            total += labels.size(0)
    handle.remove()
    student.train()
    projector.train()
    return 100.0 * correct / total


def get_penultimate_dim(model, input_size, device):
    """Get the dimension of features after avgpool (before classifier)."""
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(device)
    hook_out = {}

    def hook_fn(module, inp, out):
        hook_out["feat"] = out

    if hasattr(model, 'pool'):
        handle = model.pool.register_forward_hook(hook_fn)
    elif hasattr(model, 'avgpool'):
        handle = model.avgpool.register_forward_hook(hook_fn)
    else:
        for name, module in model.named_modules():
            if isinstance(module, nn.AdaptiveAvgPool2d):
                handle = module.register_forward_hook(hook_fn)
                break

    with torch.no_grad():
        model(dummy)
    handle.remove()
    feat = hook_out["feat"]
    if feat.dim() == 4:
        feat = feat.flatten(1)
    return feat.shape[1]


def train_one_epoch_simkd_fast(student, teacher, dataloader, optimizer,
                               criterion, device, scaler, projector,
                               teacher_classifier, teacher_input_size=224):
    """SimKD: student features → projector → teacher classifier."""
    student.train()
    projector.train()
    teacher.eval()
    running_loss = correct = total = 0

    s_feat_hook = {}
    t_feat_hook = {}

    def s_hook(module, inp, out):
        s_feat_hook["feat"] = out

    def t_hook(module, inp, out):
        t_feat_hook["feat"] = out

    if hasattr(student, 'pool'):
        sh = student.pool.register_forward_hook(s_hook)
    elif hasattr(student, 'avgpool'):
        sh = student.avgpool.register_forward_hook(s_hook)
    else:
        for name, mod in student.named_modules():
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                sh = mod.register_forward_hook(s_hook)
                break

    if hasattr(teacher, 'avgpool'):
        th = teacher.avgpool.register_forward_hook(t_hook)
    else:
        for name, mod in teacher.named_modules():
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                th = mod.register_forward_hook(t_hook)
                break

    for batch in dataloader:
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            _ = student(images)
            s_feat = s_feat_hook["feat"]
            if s_feat.dim() == 4:
                s_feat = s_feat.flatten(1)

            teacher_images = resize_for_teacher(images, teacher_input_size)
            with torch.no_grad():
                _ = teacher(teacher_images)
                t_feat = t_feat_hook["feat"].detach()
                if t_feat.dim() == 4:
                    t_feat = t_feat.flatten(1)

            proj_feat = projector(s_feat)
            loss_mse = F.mse_loss(proj_feat, t_feat)

            s_logits = teacher_classifier(proj_feat)
            loss_ce = criterion(s_logits, labels)
            loss = loss_ce + loss_mse

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = s_logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    sh.remove()
    th.remove()
    return running_loss / len(dataloader), 100.0 * correct / total


# ===========================================================================
# NEW METHOD 5: CRD (Tian et al., ICLR 2020)
# "Contrastive Representation Distillation"
# Reference: https://github.com/HobbitLong/RepDistiller
# Simplified version using embedding + NCE loss with memory bank
# ===========================================================================

class CRDEmbed(nn.Module):
    """Linear projection + L2 normalization."""
    def __init__(self, dim_in, dim_out=128):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out)

    def forward(self, x):
        x = x.flatten(1)
        x = self.linear(x)
        return F.normalize(x, p=2, dim=1)


class CRDMemory(nn.Module):
    """
    Contrastive memory bank for CRD.
    Stores L2-normalized embeddings and computes NCE scores.
    """
    def __init__(self, feat_dim, n_data, nce_k=4096, nce_t=0.07, nce_m=0.5):
        super().__init__()
        self.nce_k = nce_k
        self.nce_t = nce_t
        self.nce_m = nce_m
        self.n_data = n_data

        self.register_buffer('memory_s', torch.randn(n_data, feat_dim))
        self.register_buffer('memory_t', torch.randn(n_data, feat_dim))
        self.memory_s = F.normalize(self.memory_s, dim=1)
        self.memory_t = F.normalize(self.memory_t, dim=1)

        self.register_buffer('Z_s', torch.tensor([-1.0]))
        self.register_buffer('Z_t', torch.tensor([-1.0]))

    def forward(self, feat_s, feat_t, idx):
        batch_size = feat_s.size(0)
        K = self.nce_k

        neg_idx = torch.randint(0, self.n_data, (batch_size, K), device=feat_s.device)

        weight_s = self.memory_s[neg_idx]  # [B, K, dim]
        weight_t = self.memory_t[neg_idx]  # [B, K, dim]

        pos_s = torch.bmm(feat_s.unsqueeze(1), feat_t.unsqueeze(2)).squeeze() / self.nce_t
        neg_s = torch.bmm(weight_t, feat_s.unsqueeze(2)).squeeze(2) / self.nce_t  # [B, K]

        pos_t = pos_s
        neg_t = torch.bmm(weight_s, feat_t.unsqueeze(2)).squeeze(2) / self.nce_t  # [B, K]

        logits_s = torch.cat([pos_s.unsqueeze(1), neg_s], dim=1)  # [B, 1+K]
        logits_t = torch.cat([pos_t.unsqueeze(1), neg_t], dim=1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=feat_s.device)
        loss_s = F.cross_entropy(logits_s, labels)
        loss_t = F.cross_entropy(logits_t, labels)

        with torch.no_grad():
            self.memory_s[idx] = F.normalize(
                self.nce_m * self.memory_s[idx] + (1 - self.nce_m) * feat_s.detach(), dim=1)
            self.memory_t[idx] = F.normalize(
                self.nce_m * self.memory_t[idx] + (1 - self.nce_m) * feat_t.detach(), dim=1)

        return loss_s + loss_t


class CRDLoss(nn.Module):
    """Full CRD loss module."""
    def __init__(self, s_dim, t_dim, feat_dim=128, n_data=50000,
                 nce_k=4096, nce_t=0.07, nce_m=0.5):
        super().__init__()
        self.embed_s = CRDEmbed(s_dim, feat_dim)
        self.embed_t = CRDEmbed(t_dim, feat_dim)
        self.memory = CRDMemory(feat_dim, n_data, nce_k, nce_t, nce_m)

    def forward(self, f_s, f_t, idx):
        f_s = self.embed_s(f_s)
        f_t = self.embed_t(f_t)
        return self.memory(f_s, f_t, idx)


def train_one_epoch_crd_fast(student, teacher, dataloader, optimizer,
                             criterion, device, scaler, crd_module,
                             alpha=0.5, beta_crd=0.8,
                             teacher_input_size=224):
    """CRD: CE + contrastive loss on penultimate features."""
    student.train()
    crd_module.train()
    teacher.eval()
    running_loss = correct = total = 0

    s_feat_hook = {}
    t_feat_hook = {}

    def s_hook(module, inp, out):
        s_feat_hook["feat"] = out

    def t_hook(module, inp, out):
        t_feat_hook["feat"] = out

    if hasattr(student, 'pool'):
        sh = student.pool.register_forward_hook(s_hook)
    elif hasattr(student, 'avgpool'):
        sh = student.avgpool.register_forward_hook(s_hook)
    else:
        for name, mod in student.named_modules():
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                sh = mod.register_forward_hook(s_hook)
                break

    if hasattr(teacher, 'avgpool'):
        th = teacher.avgpool.register_forward_hook(t_hook)
    else:
        for name, mod in teacher.named_modules():
            if isinstance(mod, nn.AdaptiveAvgPool2d):
                th = mod.register_forward_hook(t_hook)
                break

    for batch in dataloader:
        images, labels, indices = batch
        images, labels = images.to(device), labels.to(device)
        indices = indices.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            s_logits = student(images)
            s_feat = s_feat_hook["feat"]
            if s_feat.dim() == 4:
                s_feat = s_feat.flatten(1)

            teacher_images = resize_for_teacher(images, teacher_input_size)
            with torch.no_grad():
                _ = teacher(teacher_images)
                t_feat = t_feat_hook["feat"]
                if t_feat.dim() == 4:
                    t_feat = t_feat.flatten(1)
                t_feat = t_feat.detach()

            loss_ce = criterion(s_logits, labels)

        loss_crd = crd_module(s_feat.float(), t_feat.float(), indices)
        loss = (1 - alpha) * loss_ce + beta_crd * loss_crd

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = s_logits.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    sh.remove()
    th.remove()
    return running_loss / len(dataloader), 100.0 * correct / total


# ===========================================================================
# NEW METHOD 6: Label Smoothing (higher epsilon, as control)
# ===========================================================================

def train_one_epoch_ls_fast(model, dataloader, optimizer, device, scaler,
                            ls_epsilon=0.2):
    """CE with higher label smoothing as non-KD control."""
    criterion_ls = nn.CrossEntropyLoss(label_smoothing=ls_epsilon)
    model.train()
    running_loss = correct = total = 0
    for batch in dataloader:
        images, labels = batch[0], batch[1]
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion_ls(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


# ===========================================================================
# NEW METHOD 7: AAKD — Ability-Aware Knowledge Distillation (Xiong et al. 2023)
# ===========================================================================

def train_one_epoch_aakd_fast(student, dataloader, optimizer, criterion,
                              device, scaler, temperature=4.0, alpha=0.5,
                              cached_logits=None, val_accuracy=0.0):
    """
    AAKD training epoch with knowledge sample selection.
    Uses cosine mapping of val_accuracy to determine selection ratio.
    Only lowest-confidence samples receive KD; CE applied to all.
    """
    student.train()
    running_loss = correct = total = 0

    va_frac = val_accuracy / 100.0
    cos_ratio = math.cos(va_frac)

    for batch in dataloader:
        if len(batch) == 3:
            images, labels, indices = batch
        else:
            images, labels = batch[0], batch[1]
            indices = None
        batch_size = images.size(0)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast():
            student_out = student(images)
            loss_ce = criterion(student_out, labels)

            with torch.no_grad():
                s_probs = F.softmax(student_out, dim=1)
                target_probs = s_probs.gather(1, labels.unsqueeze(1)).squeeze(1)

            n_sample = max(1, int(math.ceil(batch_size * cos_ratio)))
            n_sample = min(n_sample, batch_size)
            _, rank = target_probs.sort()
            selected = rank[:n_sample]
            selected_cpu = selected.cpu()

            if cached_logits is not None and indices is not None:
                teacher_out = cached_logits[indices[selected_cpu]].to(device)
            else:
                teacher_out = None

            if teacher_out is not None and n_sample > 0:
                loss_kd = kd_loss(student_out[selected], teacher_out, temperature)
                loss = alpha * loss_kd + (1 - alpha) * loss_ce
            else:
                loss = loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return running_loss / len(dataloader), 100.0 * correct / total


# ===========================================================================
# Unified experiment runners
# ===========================================================================

def _build_optimizer_scheduler(params_list, epochs, lr=1e-3):
    """Shared optimizer + scheduler setup for all methods."""
    optimizer = torch.optim.AdamW(params_list, lr=lr, weight_decay=1e-3)
    warmup_epochs = min(5, epochs // 10)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])
    return optimizer, scheduler


def run_experiment_v3(experiment_name, student_name, student_model,
                      train_loader, test_loader, device,
                      epochs=100, lr=1e-3,
                      teacher=None, ta_teacher=None,
                      kd_method="none",
                      temperature=4.0, alpha=0.5,
                      teacher_input_size=224,
                      cached_base_logits=None, cached_ta_logits=None,
                      teacher_test_logits=None,
                      num_classes=None, n_train=None):
    """
    Unified experiment runner for all 14 methods.
    Handles logit-based (none, vanilla, takd, dkd, nkd, ctkd, dist, akr, aakd),
    feature-based (fitnet, simkd, crd), self-distill (ban, cord),
    and control (label_smoothing).
    """

    print(f"\n  {'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Student: {student_name}, Method: {kd_method}")
    print(f"  Epochs: {epochs}, LR: {lr}, T: {temperature}, alpha: {alpha}")
    cache_tag = " [CACHED]" if cached_base_logits is not None else ""
    print(f"  AMP: ON, Early stop: {PATIENCE} epochs{cache_tag}")
    print(f"  {'='*60}")

    model = copy.deepcopy(student_model).to(device)
    params = count_params(model)
    print(f"  Params: {params:,}")

    # ---- Setup method-specific modules ----
    extra_params = []
    regressor = None
    projector = None
    teacher_classifier = None
    best_proj_state = None
    crd_module = None
    global_t = None
    student_ext = None

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler()

    if kd_method == "fitnet":
        t_hint_layer = "features.5"  # ConvNeXT-Tiny: 384ch, 14x14
        t_ch = 384

        # Select student hint layer based on architecture
        s_input_size = STUDENT_CONFIGS.get(student_name, {}).get("input_size", 96)

        if "MobileNetV3" in student_name:
            s_hint_layer = "features.8"  # MobileNetV3-Small: 48ch
            s_ch = _get_layer_channels(model, s_hint_layer, s_input_size)
        elif "WasteNet-Small" in student_name:
            s_hint_layer = "features.7"  # WasteNetSmall: 72ch
            s_ch = _get_layer_channels(model, s_hint_layer, s_input_size)
        else:
            # WasteNetMicro and width-scaled variants (all input_size=96)
            s_hint_layer = "features.5"  # 48ch (scaled by width_mult)
            s_ch = _get_layer_channels(model, s_hint_layer, 96)

        if s_ch is None:
            s_ch = _get_last_conv_channels(model)

        student_ext = FeatureExtractor(model, s_hint_layer)
        regressor = FitNetRegressor(s_ch, t_ch).to(device)
        extra_params += list(regressor.parameters())

    elif kd_method == "simkd":
        s_dim = get_penultimate_dim(model, STUDENT_CONFIGS.get(student_name, {}).get("input_size", 96), device)
        t_dim = get_penultimate_dim(teacher, 224, device)
        projector = SimKDProjector(s_dim, t_dim).to(device)

        if hasattr(teacher, 'classifier'):
            tc = teacher.classifier
            # ConvNeXT: Sequential(LayerNorm2d, Flatten, Linear) — projector outputs 2D,
            # so extract only the final Linear to avoid LayerNorm2d's 4D expectation
            if isinstance(tc, nn.Sequential):
                linears = [m for m in tc if isinstance(m, nn.Linear)]
                if linears:
                    tc = linears[-1]
        elif hasattr(teacher, 'fc'):
            tc = teacher.fc
        else:
            tc = list(teacher.children())[-1]
        teacher_classifier = copy.deepcopy(tc).to(device)
        teacher_classifier.eval()
        for p in teacher_classifier.parameters():
            p.requires_grad = False
        extra_params += list(projector.parameters())

    elif kd_method == "crd":
        s_dim = get_penultimate_dim(model, STUDENT_CONFIGS.get(student_name, {}).get("input_size", 96), device)
        t_dim = get_penultimate_dim(teacher, 224, device)
        n_data = n_train if n_train else 10000
        nce_k = min(4096, n_data - 1)
        crd_module = CRDLoss(s_dim, t_dim, feat_dim=128, n_data=n_data,
                             nce_k=nce_k, nce_t=0.07, nce_m=0.5).to(device)
        extra_params += list(crd_module.parameters())

    elif kd_method == "ctkd":
        global_t = CTKD_GlobalT(t_start=1.0, t_end=20.0).to(device)
        extra_params += list(global_t.parameters())

    all_params = list(model.parameters()) + extra_params
    optimizer, scheduler = _build_optimizer_scheduler(all_params, epochs, lr)

    best_acc = 0
    best_state = None
    no_improve = 0
    epochs_trained = 0
    epoch_log = []

    # AAKD state: teacher switching + val_accuracy tracking
    aakd_use_base = False
    aakd_val_acc = 0.0
    aakd_epoch_ce = None
    aakd_initial_ce = None

    for epoch in range(epochs):
        epochs_trained = epoch + 1

        if kd_method == "none":
            loss, train_acc = train_one_epoch_fast(
                model, train_loader, optimizer, criterion, device, scaler)

        elif kd_method == "vanilla":
            loss, train_acc = train_one_epoch_kd_fast(
                model, teacher, train_loader, optimizer, criterion, device, scaler,
                temperature, alpha, teacher_input_size, cached_base_logits)

        elif kd_method == "takd":
            loss, train_acc = train_one_epoch_kd_fast(
                model, ta_teacher, train_loader, optimizer, criterion, device, scaler,
                temperature, alpha, teacher_input_size, cached_ta_logits)

        elif kd_method == "dkd":
            loss, train_acc = train_one_epoch_dkd_fast(
                model, teacher, train_loader, optimizer, criterion, device, scaler,
                temperature, alpha, teacher_input_size, cached_base_logits)

        elif kd_method == "akr":
            loss, train_acc = train_one_epoch_akr_fast(
                model, teacher, ta_teacher, train_loader, optimizer, criterion,
                device, scaler, temperature, alpha,
                teacher_input_size, cached_base_logits)

        elif kd_method == "dist":
            loss, train_acc = train_one_epoch_dist_fast(
                model, teacher, train_loader, optimizer, criterion, device, scaler,
                temperature=temperature, teacher_input_size=teacher_input_size,
                cached_logits=cached_base_logits)

        elif kd_method == "nkd":
            loss, train_acc = train_one_epoch_nkd_fast(
                model, teacher, train_loader, optimizer, criterion, device, scaler,
                alpha=alpha, nkd_temp=1.0, nkd_gamma=1.5,
                teacher_input_size=teacher_input_size,
                cached_logits=cached_base_logits)

        elif kd_method == "ctkd":
            loss, train_acc = train_one_epoch_ctkd_fast(
                model, teacher, train_loader, optimizer, criterion, device, scaler,
                global_t, epoch, epochs,
                alpha=alpha, teacher_input_size=teacher_input_size,
                cached_logits=cached_base_logits)

        elif kd_method == "fitnet":
            loss, train_acc = train_one_epoch_fitnet_fast(
                student_ext, teacher, train_loader, optimizer, criterion,
                device, scaler, regressor,
                alpha=alpha, beta_hint=100.0,
                teacher_input_size=teacher_input_size)

        elif kd_method == "simkd":
            loss, train_acc = train_one_epoch_simkd_fast(
                model, teacher, train_loader, optimizer, criterion,
                device, scaler, projector, teacher_classifier,
                teacher_input_size=teacher_input_size)

        elif kd_method == "crd":
            loss, train_acc = train_one_epoch_crd_fast(
                model, teacher, train_loader, optimizer, criterion,
                device, scaler, crd_module,
                alpha=alpha, beta_crd=0.8,
                teacher_input_size=teacher_input_size)

        elif kd_method == "label_smoothing":
            loss, train_acc = train_one_epoch_ls_fast(
                model, train_loader, optimizer, device, scaler,
                ls_epsilon=0.2)

        elif kd_method == "aakd":
            aakd_logits = cached_base_logits if aakd_use_base else cached_ta_logits
            loss, train_acc = train_one_epoch_aakd_fast(
                model, train_loader, optimizer, criterion, device, scaler,
                temperature, alpha,
                cached_logits=aakd_logits,
                val_accuracy=aakd_val_acc)
            aakd_epoch_ce = loss

        else:
            raise ValueError(f"Unknown method: {kd_method}")

        scheduler.step()

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            if kd_method == "simkd":
                val_acc = _eval_simkd(model, projector, teacher_classifier, test_loader, device)
            else:
                val_result = evaluate(model, test_loader, device)
                val_acc = val_result["accuracy"]
            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                if kd_method == "simkd":
                    best_proj_state = {k: v.clone() for k, v in projector.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 5

            if kd_method == "aakd":
                aakd_val_acc = val_acc
                if aakd_initial_ce is None and aakd_epoch_ce is not None:
                    aakd_initial_ce = aakd_epoch_ce
                if (not aakd_use_base and aakd_initial_ce is not None
                        and aakd_epoch_ce is not None
                        and aakd_epoch_ce < aakd_initial_ce * 0.6):
                    aakd_use_base = True
                    print(f"    [AAKD] Switching to base teacher at epoch {epoch+1} "
                          f"(CE {aakd_epoch_ce:.4f} < {aakd_initial_ce*0.6:.4f})")

            epoch_log.append({
                "e": epoch + 1,
                "loss": round(loss, 4),
                "train_acc": round(train_acc, 2),
                "val_acc": round(val_acc, 2),
            })

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1}/{epochs} -- Loss: {loss:.4f}, "
                      f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}% "
                      f"(best: {best_acc:.2f}%)")

        if no_improve >= PATIENCE:
            print(f"    Early stop at epoch {epoch+1} (no improve for {PATIENCE} epochs)")
            break

    if best_state:
        model.load_state_dict(best_state)

    if kd_method == "simkd" and best_proj_state:
        projector.load_state_dict(best_proj_state)
        projector.eval()
        simkd_head = nn.Sequential(projector, teacher_classifier)
        if hasattr(model, 'classifier'):
            model.classifier = simkd_head
        elif hasattr(model, 'fc'):
            model.fc = simkd_head

    # Clean up hooks
    if student_ext is not None:
        student_ext.remove_hook()

    final_result = evaluate(model, test_loader, device)
    metrics = compute_comprehensive_metrics(final_result, teacher_test_logits)

    input_size = STUDENT_CONFIGS.get(student_name, {}).get("input_size", 96)
    latency = measure_latency(model, input_size, device)
    flops = count_flops(model, input_size, device)

    result = {
        "experiment": experiment_name,
        "student": student_name,
        "method": kd_method,
        "accuracy": final_result["accuracy"],
        "best_val_acc": best_acc,
        "params": params,
        "flops": flops,
        "latency_ms": latency,
        "epochs_trained": epochs_trained,
        "epoch_log": epoch_log,
    }
    result.update(metrics)

    print(f"\n  Final: Acc={final_result['accuracy']:.2f}%, F1={metrics['f1_macro']:.2f}%, "
          f"ECE={metrics['ece']:.4f}, Params={params:,}, "
          f"FLOPs={flops:,}, Epochs={epochs_trained}")
    return result, model


def run_self_distill_v3(experiment_name, student_name, student_factory,
                        num_classes, train_loader, test_loader, device,
                        epochs=100, lr=1e-3, method="ban",
                        temperature=2.0, alpha=0.5, entropy_gate=0.3,
                        teacher_test_logits=None):
    """BAN/BAN-CL with AMP + early stopping + comprehensive metrics."""

    def _train_one_gen(model, teacher, is_cord=False):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
        warmup_ep = min(5, epochs // 10)
        warmup = LinearLR(opt, start_factor=0.01, end_factor=1.0, total_iters=warmup_ep)
        cosine = CosineAnnealingLR(opt, T_max=epochs - warmup_ep, eta_min=1e-6)
        sched = SequentialLR(opt, [warmup, cosine], milestones=[warmup_ep])
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        scaler = torch.cuda.amp.GradScaler()
        best_acc = 0
        best_state = None
        no_improve = 0
        ep_trained = 0
        ep_log = []

        for epoch in range(epochs):
            ep_trained = epoch + 1
            model.train()
            rloss = correct = total = 0
            if teacher is not None:
                teacher.eval()
            for batch in train_loader:
                images, labels = batch[0], batch[1]
                images, labels = images.to(device), labels.to(device)
                opt.zero_grad(set_to_none=True)

                with torch.cuda.amp.autocast():
                    s_logits = model(images)
                    loss_ce = crit(s_logits, labels)

                    if teacher is None:
                        loss = loss_ce
                    else:
                        with torch.no_grad():
                            t_logits = teacher(images)
                            t_probs = F.softmax(t_logits / temperature, dim=1)
                        kl = F.kl_div(
                            F.log_softmax(s_logits / temperature, dim=1),
                            t_probs, reduction='none'
                        ).sum(dim=1) * (temperature ** 2)

                        if is_cord:
                            h = -(t_probs * (t_probs + 1e-8).log()).sum(dim=1)
                            h = h / math.log(max(num_classes, 2))
                            kd_mask = (h >= entropy_gate).float()
                            skip_mask = 1.0 - kd_mask
                            loss = (kd_mask * (alpha * kl + loss_ce.expand_as(kl))
                                    + skip_mask * loss_ce.expand_as(kl)).mean()
                        else:
                            loss = alpha * kl.mean() + (1 - alpha) * loss_ce

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                rloss += loss.item()
                _, preds = s_logits.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            sched.step()
            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                val_r = evaluate(model, test_loader, device)
                if val_r["accuracy"] > best_acc:
                    best_acc = val_r["accuracy"]
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 5

                ep_log.append({
                    "e": epoch + 1,
                    "loss": round(rloss / len(train_loader), 4),
                    "train_acc": round(100.0 * correct / total, 2),
                    "val_acc": round(val_r["accuracy"], 2),
                })

                if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                    print(f"    Epoch {epoch+1}/{epochs} -- Loss: {rloss/len(train_loader):.4f}, "
                          f"Train: {100.*correct/total:.2f}%, Val: {val_r['accuracy']:.2f}% "
                          f"(best: {best_acc:.2f}%)")

            if no_improve >= PATIENCE:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)
        return model, best_acc, ep_trained, ep_log

    params = sum(p.numel() for p in student_factory(num_classes).parameters())
    print(f"\n  {'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Student: {student_name}, Method: {method.upper()}")
    print(f"  Params: {params:,}, Epochs/gen: {epochs}, T: {temperature}")
    print(f"  AMP: ON, Early stop: {PATIENCE} epochs")
    print(f"  {'='*60}")

    print(f"  [Gen 0] CE training...")
    gen0 = student_factory(num_classes).to(device)
    gen0, gen0_acc, gen0_epochs, gen0_log = _train_one_gen(gen0, teacher=None)
    print(f"  [Gen 0] best_val_acc: {gen0_acc:.2f}%, epochs: {gen0_epochs}")

    gen0_eval = evaluate(gen0, test_loader, device)
    gen0_test_logits = gen0_eval["logits"]

    tag = "BAN-CL (entropy-gated)" if method == "cord" else "BAN"
    print(f"  [Gen 1] {tag} distilling from gen0...")
    gen1 = student_factory(num_classes).to(device)
    gen0.eval()
    gen1, gen1_acc, gen1_epochs, gen1_log = _train_one_gen(
        gen1, teacher=gen0, is_cord=(method == "cord"))
    print(f"  [Gen 1] best_val_acc: {gen1_acc:.2f}%, epochs: {gen1_epochs}")

    final_r = evaluate(gen1, test_loader, device)
    metrics = compute_comprehensive_metrics(final_r, teacher_test_logits)

    gen1_preds = final_r["predictions"]
    gen0_preds = gen0_test_logits.argmax(dim=1).numpy()
    self_teacher_agreement = round(100.0 * float(np.mean(gen1_preds == gen0_preds)), 4)

    result = {
        "experiment": experiment_name,
        "student": student_name,
        "method": method,
        "accuracy": final_r["accuracy"],
        "best_val_acc": gen1_acc,
        "params": params,
        "gen0_acc": gen0_acc,
        "gen0_epochs": gen0_epochs,
        "gen1_epochs": gen1_epochs,
        "epochs_trained": gen0_epochs + gen1_epochs,
        "self_teacher_agreement": self_teacher_agreement,
        "epoch_log_gen0": gen0_log,
        "epoch_log_gen1": gen1_log,
        "flops": 0,
    }
    result.update(metrics)

    print(f"\n  Final: Acc={final_r['accuracy']:.2f}%, F1={metrics['f1_macro']:.2f}%, "
          f"ECE={metrics['ece']:.4f}, Self-Agree={self_teacher_agreement:.1f}%, "
          f"Params={params:,}")
    return result, gen1


# ===========================================================================
# Helper
# ===========================================================================

def _get_last_conv_channels(model):
    """Walk backwards through model to find last conv output channels."""
    last_ch = 64
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_ch = m.out_channels
    return last_ch


def _get_layer_channels(model, layer_name, input_size=96):
    """Get output channels of a named layer via dummy forward."""
    hook_out = {}
    handle = None
    for name, module in model.named_modules():
        if name == layer_name:
            def hook_fn(mod, inp, out, _d=hook_out):
                _d["out"] = out
            handle = module.register_forward_hook(hook_fn)
            break
    if handle is None:
        return None

    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(next(model.parameters()).device)
    with torch.no_grad():
        try:
            model(dummy)
        except Exception:
            handle.remove()
            return None
    handle.remove()
    if "out" in hook_out:
        return hook_out["out"].shape[1]
    return None
