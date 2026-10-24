"""
=============================================================================
Adaptive Knowledge Routing via Hybrid Sample Selection
Knowledge Distillation Framework for WaRP-C Waste Classification
=============================================================================

Dissertation : Adaptive Knowledge Routing melalui Hybrid Sample Selection
               pada Knowledge Distillation untuk Klasifikasi Sampah
               Berbasis Lightweight CNN di Edge Computing

Author       : Kholiq Budiman (25/571711/SPA/01180), UGM
Promotor     : Prof. Dr. Ir. Jazi Eko Istiyanto, M.Sc.
               Dr. techn. Aufaclav Zatu Kusuma Frisky S.Si., M.Sc.

Pipeline:
  Phase 1: Load teachers + generate difficulty scores
  Phase 2: Train student baselines (no KD)
  Phase 3: Vanilla KD (Hinton et al. 2015)
  Phase 4: TAKD — Teacher Assistant KD (Mirzadeh et al. 2020)
  Phase 5: Proposed — Adaptive Knowledge Routing (AKR)
  Phase 6: Analysis (confusion matrix, difficulty distribution, Pareto plot)
  Phase 7: Statistical significance (bootstrap CI, multi-seed)

Metrics: Top-1 Accuracy, FLOPs, Params, Inference Latency
=============================================================================
"""

import os
import sys
import json
import time
import copy
import math
import random
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================================
# CONSTANTS
# ============================================================================

CLASS_NAMES = [
    "bottle-blue", "bottle-green", "bottle-dark", "bottle-milk",
    "bottle-transp", "bottle-multicolor", "bottle-yogurt", "bottle-oil",
    "cans", "juice-cardboard", "milk-cardboard", "detergent-color",
    "detergent-transparent", "detergent-box", "canister",
    "bottle-blue-full", "bottle-transp-full", "bottle-dark-full",
    "bottle-green-full", "bottle-multicolorv-full", "bottle-milk-full",
    "bottle-oil-full", "detergent-white", "bottle-blue5l",
    "bottle-blue5l-full", "glass-transp", "glass-dark", "glass-green",
]

GROUPS = {
    "bottle": [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18, 19, 20, 21, 23, 24],
    "cardboard": [9, 10],
    "detergent": [11, 12, 13, 22],
    "canister": [14],
    "cans": [8],
    "glass": [25, 26, 27],
}


# ============================================================================
# STUDENT MODEL ARCHITECTURES (for ESP32-CAM: <1M params, <500KB)
# ============================================================================

class MicroConvBlock(nn.Module):
    """Depthwise separable conv block — efficient for edge."""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        x = self.act(self.bn2(self.pw(x)))
        return x


class WasteNetMicro(nn.Module):
    """
    Custom micro CNN for ESP32-CAM deployment.
    Target: <200K params, input 96x96, FLOPs ~10M
    Inspired by MCUNet (Lin et al., 2020) architecture search results.
    """
    def __init__(self, num_classes=28, width_mult=1.0):
        super().__init__()
        def ch(c):
            return max(8, int(c * width_mult))

        self.features = nn.Sequential(
            # Stem: 96x96 -> 48x48
            nn.Conv2d(3, ch(16), 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch(16)),
            nn.ReLU6(inplace=True),
            # Stage 1: 48x48 -> 24x24
            MicroConvBlock(ch(16), ch(24), stride=2),
            # Stage 2: 24x24 -> 12x12
            MicroConvBlock(ch(24), ch(32), stride=2),
            MicroConvBlock(ch(32), ch(32), stride=1),
            # Stage 3: 12x12 -> 6x6
            MicroConvBlock(ch(32), ch(48), stride=2),
            MicroConvBlock(ch(48), ch(48), stride=1),
            # Stage 4: 6x6 -> 3x3
            MicroConvBlock(ch(48), ch(64), stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch(64), num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x


class WasteNetSmall(nn.Module):
    """
    Slightly larger student for higher accuracy.
    Target: <500K params, input 128x128, FLOPs ~30M
    """
    def __init__(self, num_classes=28, width_mult=1.0):
        super().__init__()
        def ch(c):
            return max(8, int(c * width_mult))

        self.features = nn.Sequential(
            nn.Conv2d(3, ch(24), 3, 2, 1, bias=False),
            nn.BatchNorm2d(ch(24)),
            nn.ReLU6(inplace=True),
            MicroConvBlock(ch(24), ch(32), stride=2),
            MicroConvBlock(ch(32), ch(32), stride=1),
            MicroConvBlock(ch(32), ch(48), stride=2),
            MicroConvBlock(ch(48), ch(48), stride=1),
            MicroConvBlock(ch(48), ch(48), stride=1),
            MicroConvBlock(ch(48), ch(72), stride=2),
            MicroConvBlock(ch(72), ch(72), stride=1),
            MicroConvBlock(ch(72), ch(96), stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch(96), num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.classifier(x)
        return x


def create_mobilenetv3_small(num_classes=28, width_mult=0.75):
    """MobileNetV3-Small with reduced width — practical edge student."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


STUDENT_CONFIGS = {
    "WasteNet-Micro": {
        "factory": lambda nc: WasteNetMicro(nc, width_mult=1.0),
        "input_size": 96,
        "pretrained": False,
        "description": "Ultra-lightweight for ESP32-CAM (<200K params)",
    },
    "WasteNet-Small": {
        "factory": lambda nc: WasteNetSmall(nc, width_mult=1.0),
        "input_size": 128,
        "pretrained": False,
        "description": "Balanced micro model (~400K params)",
    },
    "MobileNetV3-Small": {
        "factory": lambda nc: create_mobilenetv3_small(nc),
        "input_size": 224,
        "pretrained": True,
        "description": "Pretrained MobileNetV3-Small (~2.5M params)",
    },
}

# Teacher configs — loaded from pre-trained weights
TEACHER_CONFIGS = {
    "ConvNeXT-Tiny": {
        "factory": lambda nc: models.convnext_tiny(weights=None),
        "head_type": "convnext",
        "input_size": 224,
    },
    "EfficientNet-B0": {
        "factory": lambda nc: models.efficientnet_b0(weights=None),
        "head_type": "efficientnet",
        "input_size": 224,
    },
    "MobileNetV3": {
        "factory": lambda nc: models.mobilenet_v3_large(weights=None),
        "head_type": "mobilenet",
        "input_size": 224,
    },
}


# ============================================================================
# DATA LOADING
# ============================================================================

def get_transforms(input_size, is_train=True, use_randaugment=False):
    """Build transforms for given input size."""
    if is_train:
        t_list = [
            transforms.Resize((input_size, input_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
        if use_randaugment:
            t_list.append(transforms.RandAugment(num_ops=2, magnitude=7))
        t_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
        return transforms.Compose(t_list)
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


def create_dataloaders(data_root, input_size=224, batch_size=32, num_workers=4):
    """Create train/test dataloaders with balanced sampling."""
    train_dir = Path(data_root) / "train"
    test_dir = Path(data_root) / "test"

    train_dataset = ImageFolder(str(train_dir),
                                transform=get_transforms(input_size, is_train=True))
    test_dataset = ImageFolder(str(test_dir),
                               transform=get_transforms(input_size, is_train=False))

    # Balanced sampler
    targets = np.array(train_dataset.targets)
    class_counts = np.bincount(targets, minlength=len(train_dataset.classes))
    weights = 1.0 / np.maximum(class_counts[targets], 1)
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    print(f"  Data: {len(train_dataset)} train, {len(test_dataset)} test, "
          f"{len(train_dataset.classes)} classes")

    return train_loader, test_loader, train_dataset, test_dataset


# ============================================================================
# DIFFICULTY ESTIMATION (Core of Hybrid Sample Selection)
# ============================================================================

class DifficultyEstimator:
    """
    Estimate per-sample difficulty using teacher confidence margin.
    d_i = 1 - (p_teacher(y_i|x_i) - max_{k!=y_i} p_teacher(k|x_i))
    """

    def __init__(self, teacher_model, device, temperature=1.0):
        self.teacher = teacher_model
        self.device = device
        self.temperature = temperature
        self.teacher.eval()

    @torch.no_grad()
    def compute_difficulty(self, dataloader):
        """Compute difficulty scores for all samples in dataloader."""
        all_difficulties = []
        all_entropies = []
        all_labels = []
        all_teacher_logits = []

        for images, labels in dataloader:
            images = images.to(self.device)
            logits = self.teacher(images)
            probs = F.softmax(logits / self.temperature, dim=1)

            # Margin-based difficulty
            top2 = probs.topk(2, dim=1).values
            margins = top2[:, 0] - top2[:, 1]
            difficulties = 1.0 - margins

            # Entropy
            entropies = -(probs * (probs + 1e-8).log()).sum(dim=1)

            all_difficulties.extend(difficulties.cpu().numpy())
            all_entropies.extend(entropies.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_teacher_logits.append(logits.cpu())

        return {
            "difficulty": np.array(all_difficulties),
            "entropy": np.array(all_entropies),
            "labels": np.array(all_labels),
            "teacher_logits": torch.cat(all_teacher_logits, dim=0),
        }

    def categorize_samples(self, difficulties, method="quantile",
                           hard_pct=0.33, easy_pct=0.33):
        """
        Categorize samples into hard/standard/easy.

        Args:
            difficulties: array of difficulty scores
            method: 'quantile' or 'gmm'
            hard_pct: fraction of samples considered hard
            easy_pct: fraction of samples considered easy
        """
        if method == "quantile":
            easy_thresh = np.percentile(difficulties, easy_pct * 100)
            hard_thresh = np.percentile(difficulties, (1 - hard_pct) * 100)

            categories = np.full(len(difficulties), 1)  # 1 = standard
            categories[difficulties <= easy_thresh] = 0  # 0 = easy
            categories[difficulties >= hard_thresh] = 2  # 2 = hard

        elif method == "gmm":
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=3, random_state=42)
            gmm.fit(difficulties.reshape(-1, 1))
            categories = gmm.predict(difficulties.reshape(-1, 1))
            # Reorder so that lowest mean = easy, highest = hard
            means = gmm.means_.flatten()
            order = np.argsort(means)
            remap = {order[i]: i for i in range(3)}
            categories = np.array([remap[c] for c in categories])

        counts = np.bincount(categories, minlength=3)
        print(f"  Difficulty distribution: Easy={counts[0]}, Standard={counts[1]}, Hard={counts[2]}")

        return categories  # 0=easy, 1=standard, 2=hard


# ============================================================================
# KD LOSS FUNCTIONS
# ============================================================================

def kd_loss(student_logits, teacher_logits, temperature=4.0):
    """Standard KL-divergence KD loss (Hinton et al. 2015)."""
    student_soft = F.log_softmax(student_logits / temperature, dim=1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=1)
    loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean") * (temperature ** 2)
    return loss


def dkd_loss(student_logits, teacher_logits, targets, temperature=4.0,
             alpha_tc=1.0, beta_nc=8.0):
    """
    Decoupled Knowledge Distillation (Zhao et al., CVPR 2022).

    Splits KD loss into Target-Class KD (TCKD) + Non-target-class KD (NCKD).
    NCKD contains the "dark knowledge" that Hinton KD suppresses.

    Args:
        student_logits: [B, K]
        teacher_logits: [B, K]
        targets: [B] ground truth class indices
        alpha_tc: weight for target class loss (default 1.0)
        beta_nc: weight for non-target class loss (default 8.0) — key insight of DKD

    Reference: Zhao et al., "Decoupled Knowledge Distillation", CVPR 2022
    """
    gt_mask = F.one_hot(targets, num_classes=student_logits.size(1)).bool()
    other_mask = ~gt_mask

    # Student and teacher softmax at temperature T
    s_log = F.log_softmax(student_logits / temperature, dim=1)
    t_prob = F.softmax(teacher_logits / temperature, dim=1)

    # Binary target-class distribution: [p_t, 1-p_t]
    s_tc = s_log[gt_mask]  # student log-prob of GT class
    t_tc = t_prob[gt_mask]  # teacher prob of GT class

    # Target-class KD: treats GT vs rest as binary classification
    s_nc_sum = torch.log(1.0 - torch.exp(s_tc) + 1e-8)  # log(1 - p_gt)
    t_nc_sum = 1.0 - t_tc  # 1 - p_gt
    tckd_loss = F.kl_div(
        torch.stack([s_tc, s_nc_sum], dim=1),
        torch.stack([t_tc, t_nc_sum], dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)

    # Non-target-class KD: renormalize over non-target classes and compute KL
    s_log_masked = F.log_softmax(
        student_logits / temperature - 1e9 * gt_mask.float(), dim=1
    )
    t_prob_masked = F.softmax(
        teacher_logits / temperature - 1e9 * gt_mask.float(), dim=1
    )
    nckd_loss = F.kl_div(
        s_log_masked, t_prob_masked, reduction="batchmean"
    ) * (temperature ** 2)

    return alpha_tc * tckd_loss + beta_nc * nckd_loss


def adaptive_kd_loss(student_logits, teacher_logits, difficulties, temperature=4.0):
    """
    Adaptive KD loss — weight varies by sample difficulty.
    Hard samples get stronger distillation signal.
    """
    base_loss = kd_loss(student_logits, teacher_logits, temperature)

    # Per-sample alpha based on difficulty (higher difficulty = higher alpha)
    alphas = difficulties.clamp(0.1, 0.9)
    # Weight the loss per sample
    per_sample_kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="none",
    ).sum(dim=1) * (temperature ** 2)

    weighted_loss = (alphas * per_sample_kd).mean()
    return weighted_loss


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """Standard supervised training epoch."""
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), 100.0 * correct / total


def resize_for_teacher(images, teacher_input_size=224):
    """Resize images to teacher's expected input size if needed."""
    if images.shape[2] != teacher_input_size or images.shape[3] != teacher_input_size:
        return F.interpolate(images, size=(teacher_input_size, teacher_input_size),
                             mode="bilinear", align_corners=False)
    return images


def train_one_epoch_kd(student, teacher, dataloader, optimizer, criterion,
                       device, temperature=4.0, alpha=0.5, teacher_input_size=224):
    """Vanilla KD training epoch."""
    student.train()
    teacher.eval()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        student_out = student(images)
        # Resize images to teacher's expected input size
        teacher_images = resize_for_teacher(images, teacher_input_size)
        with torch.no_grad():
            teacher_out = teacher(teacher_images)

        loss_ce = criterion(student_out, labels)
        loss_kd = kd_loss(student_out, teacher_out, temperature)
        loss = alpha * loss_kd + (1 - alpha) * loss_ce

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), 100.0 * correct / total


def train_one_epoch_dkd(student, teacher, dataloader, optimizer, criterion,
                        device, temperature=4.0, alpha=0.5,
                        teacher_input_size=224,
                        alpha_tc=1.0, beta_nc=8.0):
    """Decoupled Knowledge Distillation training epoch (Zhao CVPR 2022)."""
    student.train()
    teacher.eval()
    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        student_out = student(images)
        teacher_images = resize_for_teacher(images, teacher_input_size)
        with torch.no_grad():
            teacher_out = teacher(teacher_images)

        loss_ce = criterion(student_out, labels)
        loss_kd = dkd_loss(student_out, teacher_out, labels,
                           temperature=temperature,
                           alpha_tc=alpha_tc, beta_nc=beta_nc)
        loss = alpha * loss_kd + (1 - alpha) * loss_ce

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), 100.0 * correct / total


def train_one_epoch_akr(student, base_teacher, ta_teacher, dataloader,
                        optimizer, criterion, difficulty_fn, device,
                        temperature=4.0, alpha_kd=0.5,
                        easy_thresh=None, hard_thresh=None,
                        teacher_input_size=224):
    """
    Adaptive Knowledge Routing (AKR) training epoch.

    Key fix: difficulty is computed ON-THE-FLY from teacher output per batch
    (not from pre-computed scores indexed sequentially — that causes shuffle mismatch).

    Routing policy:
      - Hard samples (high difficulty): distill from base_teacher (large)
      - Standard samples (mid difficulty): distill from ta_teacher (intermediate)
      - Easy samples (low difficulty): pure CE, no distillation
    """
    student.train()
    base_teacher.eval()
    ta_teacher.eval()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in dataloader:
        batch_size = images.size(0)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        student_out = student(images)

        # Resize for teacher if student uses different input size
        teacher_images = resize_for_teacher(images, teacher_input_size)

        # Compute difficulty ON-THE-FLY from base teacher (fixes shuffle mismatch)
        with torch.no_grad():
            bt_logits = base_teacher(teacher_images)
            bt_probs = F.softmax(bt_logits / 1.0, dim=1)
            # Margin-based difficulty: d = 1 - (top1_prob - top2_prob)
            top2_vals = bt_probs.topk(2, dim=1).values
            margins = top2_vals[:, 0] - top2_vals[:, 1]
            d_tensor = 1.0 - margins  # higher = harder

        # Use quantile-based thresholds from batch if not provided
        if easy_thresh is None:
            easy_thresh_val = d_tensor.quantile(0.33).item()
        else:
            easy_thresh_val = easy_thresh
        if hard_thresh is None:
            hard_thresh_val = d_tensor.quantile(0.67).item()
        else:
            hard_thresh_val = hard_thresh

        easy_mask = d_tensor < easy_thresh_val
        hard_mask = d_tensor >= hard_thresh_val
        std_mask = ~easy_mask & ~hard_mask

        # CE loss for all samples
        loss_ce = criterion(student_out, labels)

        # Per-sample adaptive alpha: harder samples get more KD weight
        alphas = d_tensor.clamp(0.1, 0.7)

        # KD loss with routing
        loss_kd = torch.tensor(0.0, device=device, requires_grad=False)

        if hard_mask.any():
            # Hard → base teacher (already computed above)
            hard_kd = kd_loss(student_out[hard_mask], bt_logits[hard_mask], temperature)
            loss_kd = loss_kd + alphas[hard_mask].mean() * hard_kd

        if std_mask.any():
            # Standard → TA teacher
            with torch.no_grad():
                ta_logits = ta_teacher(teacher_images[std_mask])
            std_kd = kd_loss(student_out[std_mask], ta_logits, temperature)
            loss_kd = loss_kd + alphas[std_mask].mean() * std_kd

        # Easy → no distillation, pure CE (already in loss_ce)

        # Combined loss: weighted sum
        # CE always contributes; KD contributes proportionally to non-easy fraction
        n_distilled = hard_mask.sum() + std_mask.sum()
        if n_distilled > 0:
            kd_weight = alpha_kd * (n_distilled.float() / batch_size)
            loss = kd_weight * loss_kd + (1.0 - kd_weight) * loss_ce
        else:
            loss = loss_ce

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = student_out.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / len(dataloader), 100.0 * correct / total


@torch.no_grad()
def evaluate(model, dataloader, device):
    """Evaluate model and return detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    all_logits = []

    for images, labels in dataloader:
        images = images.to(device)
        outputs = model(images)
        _, preds = outputs.max(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_logits.append(outputs.cpu())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = torch.cat(all_logits, dim=0)

    accuracy = 100.0 * np.mean(all_preds == all_labels)

    # Per-class accuracy
    num_classes = all_logits.shape[1]
    per_class_acc = {}
    for c in range(num_classes):
        mask = all_labels == c
        if mask.sum() > 0:
            per_class_acc[c] = 100.0 * (all_preds[mask] == c).mean()

    return {
        "accuracy": accuracy,
        "per_class_acc": per_class_acc,
        "predictions": all_preds,
        "labels": all_labels,
        "logits": all_logits,
    }


def count_params(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_flops(model, input_size, device):
    """Estimate FLOPs using a forward pass hook."""
    try:
        from fvcore.nn import FlopCountAnalysis
        dummy = torch.randn(1, 3, input_size, input_size).to(device)
        flops = FlopCountAnalysis(model, dummy)
        return flops.total()
    except ImportError:
        # Fallback: rough estimate
        dummy = torch.randn(1, 3, input_size, input_size).to(device)
        # Just return param count * 2 as rough estimate
        return count_params(model) * 2


def measure_latency(model, input_size, device, num_runs=100):
    """Measure inference latency in ms."""
    model.eval()
    dummy = torch.randn(1, 3, input_size, input_size).to(device)

    # Warmup
    for _ in range(20):
        with torch.no_grad():
            _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_runs):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    return (elapsed / num_runs) * 1000  # ms


# ============================================================================
# FULL TRAINING PIPELINE
# ============================================================================

def load_teacher(teacher_name, weights_path, num_classes, device):
    """Load a pre-trained teacher model."""
    cfg = TEACHER_CONFIGS[teacher_name]
    model = cfg["factory"](num_classes)

    # Adapt head
    head_type = cfg["head_type"]
    if head_type == "convnext":
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif head_type == "efficientnet":
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
    elif head_type == "mobilenet":
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)

    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"  Loaded teacher: {teacher_name} from {weights_path}")
    return model


def run_experiment(experiment_name, student_name, student_model, train_loader,
                   test_loader, device, epochs=100, lr=1e-3,
                   teacher=None, ta_teacher=None, kd_method="none",
                   temperature=4.0, alpha=0.5, difficulty_scores=None,
                   teacher_input_size=224):
    """
    Run a single KD experiment.

    kd_method: 'none' | 'vanilla' | 'takd' | 'akr'
    teacher_input_size: input resolution the teacher was trained on (for resize)
    """
    print(f"\n  {'='*60}")
    print(f"  Experiment: {experiment_name}")
    print(f"  Student: {student_name}, Method: {kd_method}")
    print(f"  Epochs: {epochs}, LR: {lr}, T: {temperature}, alpha: {alpha}")
    print(f"  Teacher input size: {teacher_input_size}")
    print(f"  {'='*60}")

    model = copy.deepcopy(student_model).to(device)
    params = count_params(model)
    print(f"  Params: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # Warmup + cosine scheduler
    warmup_epochs = min(5, epochs // 10)
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_epochs])

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_acc = 0
    best_state = None
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(epochs):
        if kd_method == "none":
            loss, train_acc = train_one_epoch(model, train_loader, optimizer,
                                              criterion, device)
        elif kd_method == "vanilla":
            loss, train_acc = train_one_epoch_kd(model, teacher, train_loader,
                                                  optimizer, criterion, device,
                                                  temperature, alpha,
                                                  teacher_input_size=teacher_input_size)
        elif kd_method == "takd":
            # TAKD: distill from TA (intermediate teacher) only
            loss, train_acc = train_one_epoch_kd(model, ta_teacher, train_loader,
                                                  optimizer, criterion, device,
                                                  temperature, alpha,
                                                  teacher_input_size=teacher_input_size)
        elif kd_method == "akr":
            loss, train_acc = train_one_epoch_akr(
                model, teacher, ta_teacher, train_loader,
                optimizer, criterion, None, device,
                temperature, alpha_kd=alpha,
                teacher_input_size=teacher_input_size,
            )
        elif kd_method == "dkd":
            # DKD (Zhao et al. CVPR 2022): Decoupled Knowledge Distillation
            loss, train_acc = train_one_epoch_dkd(
                model, teacher, train_loader,
                optimizer, criterion, device,
                temperature=temperature, alpha=alpha,
                teacher_input_size=teacher_input_size,
                alpha_tc=1.0, beta_nc=8.0,
            )

        scheduler.step()

        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)

        # Validate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_result = evaluate(model, test_loader, device)
            val_acc = val_result["accuracy"]
            history["val_acc"].append(val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

            if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch+1}/{epochs} — Loss: {loss:.4f}, "
                      f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}% "
                      f"(best: {best_acc:.2f}%)")

    # Load best and final evaluate
    if best_state:
        model.load_state_dict(best_state)

    final_result = evaluate(model, test_loader, device)
    input_size = STUDENT_CONFIGS.get(student_name, {}).get("input_size", 224)
    latency = measure_latency(model, input_size, device)
    flops = count_flops(model, input_size, device)

    result = {
        "experiment": experiment_name,
        "student": student_name,
        "method": kd_method,
        "accuracy": final_result["accuracy"],
        "best_val_acc": best_acc,
        "per_class_acc": final_result["per_class_acc"],
        "predictions": final_result["predictions"],
        "labels": final_result["labels"],
        "params": params,
        "flops": flops,
        "latency_ms": latency,
        "history": history,
        "temperature": temperature,
        "alpha": alpha,
    }

    print(f"\n  Final: Acc={final_result['accuracy']:.2f}%, "
          f"Params={params:,}, FLOPs={flops:,}, Latency={latency:.2f}ms")

    return result, model


# ============================================================================
# ANALYSIS & VISUALIZATION
# ============================================================================

def generate_confusion_matrix(results, class_names, output_dir, experiment_name):
    """Generate and save confusion matrix."""
    from sklearn.metrics import confusion_matrix as cm_func

    preds = results["predictions"]
    labels = results["labels"]

    num_classes = max(labels.max(), preds.max()) + 1
    names = [class_names[i] if i < len(class_names) else f"cls_{i}"
             for i in range(num_classes)]

    matrix = cm_func(labels, preds, labels=range(num_classes))

    fig, ax = plt.subplots(figsize=(16, 14))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=names, yticklabels=names, ax=ax,
                annot_kws={"size": 6})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title(f"Confusion Matrix — {experiment_name}", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    path = output_dir / f"confusion_{experiment_name.replace(' ', '_')}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_difficulty_analysis(difficulty_data, class_names, output_dir):
    """Generate difficulty distribution plots."""
    difficulties = difficulty_data["difficulty"]
    entropies = difficulty_data["entropy"]
    labels = difficulty_data["labels"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Overall difficulty histogram
    axes[0].hist(difficulties, bins=50, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Difficulty Score", fontsize=11)
    axes[0].set_ylabel("Count", fontsize=11)
    axes[0].set_title("Sample Difficulty Distribution", fontsize=13)
    axes[0].axvline(np.percentile(difficulties, 33), color="green", ls="--",
                    label=f"Easy threshold ({np.percentile(difficulties, 33):.2f})")
    axes[0].axvline(np.percentile(difficulties, 67), color="red", ls="--",
                    label=f"Hard threshold ({np.percentile(difficulties, 67):.2f})")
    axes[0].legend(fontsize=8)

    # 2. Per-class mean difficulty
    unique_labels = np.unique(labels)
    class_diffs = []
    class_labels_str = []
    for c in unique_labels:
        mask = labels == c
        name = class_names[c] if c < len(class_names) else f"cls_{c}"
        class_diffs.append(difficulties[mask].mean())
        class_labels_str.append(name)

    sorted_idx = np.argsort(class_diffs)[::-1]
    axes[1].barh(range(len(class_diffs)),
                 [class_diffs[i] for i in sorted_idx],
                 color=["#e74c3c" if class_diffs[i] > np.percentile(difficulties, 67)
                        else "#f39c12" if class_diffs[i] > np.percentile(difficulties, 33)
                        else "#2ecc71" for i in sorted_idx],
                 edgecolor="white")
    axes[1].set_yticks(range(len(class_diffs)))
    axes[1].set_yticklabels([class_labels_str[i] for i in sorted_idx], fontsize=7)
    axes[1].set_xlabel("Mean Difficulty", fontsize=11)
    axes[1].set_title("Per-Class Mean Difficulty", fontsize=13)
    axes[1].invert_yaxis()

    # 3. Difficulty vs Entropy scatter
    axes[2].scatter(difficulties, entropies, alpha=0.3, s=5, c="#3498db")
    axes[2].set_xlabel("Difficulty Score", fontsize=11)
    axes[2].set_ylabel("Prediction Entropy", fontsize=11)
    axes[2].set_title("Difficulty vs Entropy", fontsize=13)

    plt.tight_layout()
    path = output_dir / "difficulty_analysis.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_pareto_plot(all_results, output_dir):
    """Generate FLOPs vs Accuracy Pareto front."""
    fig, ax = plt.subplots(figsize=(10, 7))

    method_colors = {
        "none": "#95a5a6",
        "vanilla": "#3498db",
        "takd": "#2ecc71",
        "akr": "#e74c3c",
    }
    method_labels = {
        "none": "No KD (Baseline)",
        "vanilla": "Vanilla KD",
        "takd": "TAKD",
        "akr": "AKR (Proposed)",
    }
    student_markers = {
        "WasteNet-Micro": "o",
        "WasteNet-Small": "s",
        "MobileNetV3-Small": "D",
    }

    for result in all_results:
        method = result["method"]
        student = result["student"]
        color = method_colors.get(method, "#333333")
        marker = student_markers.get(student, "o")
        label = f"{method_labels.get(method, method)} ({student})"

        ax.scatter(result["flops"] / 1e6, result["accuracy"],
                   c=color, marker=marker, s=100, edgecolors="black",
                   linewidth=0.5, zorder=3, label=label)

    ax.set_xlabel("FLOPs (M)", fontsize=12)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=12)
    ax.set_title("FLOPs vs Accuracy — Pareto Front", fontsize=14, fontweight="bold")
    ax.set_xscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    plt.tight_layout()
    path = output_dir / "pareto_flops_vs_accuracy.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_comparison_table(all_results, output_dir):
    """Generate comprehensive comparison table."""
    rows = []
    for r in all_results:
        rows.append({
            "Experiment": r["experiment"],
            "Student": r["student"],
            "Method": r["method"],
            "Top-1 Acc (%)": f"{r['accuracy']:.2f}",
            "Best Val Acc (%)": f"{r['best_val_acc']:.2f}",
            "Params": f"{r['params']:,}",
            "FLOPs (M)": f"{r['flops']/1e6:.1f}",
            "Latency (ms)": f"{r['latency_ms']:.2f}",
            "Temperature": r.get("temperature", "-"),
            "Alpha": r.get("alpha", "-"),
        })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "kd_comparison_results.csv"
    df.to_csv(csv_path, index=False)

    latex_path = output_dir / "kd_comparison_results.tex"
    with open(latex_path, "w") as f:
        f.write("% KD Comparison Results — Adaptive Knowledge Routing\n")
        f.write(df.to_latex(index=False, escape=True))

    # Pretty print
    print(f"\n  {'='*100}")
    print(f"  KD Comparison Results")
    print(f"  {'='*100}")
    print(df.to_string(index=False))
    print(f"  {'='*100}")
    print(f"  Saved: {csv_path}")

    return df


# ============================================================================
# ADVANCED ANALYSIS & VISUALIZATION (for paper & dissertation)
# ============================================================================

@torch.no_grad()
def extract_logits_and_features(model, dataloader, device, input_size=224):
    """Extract logits, softmax probs, and penultimate features from a model."""
    model.eval()
    all_logits = []
    all_probs = []
    all_labels = []

    for images, labels in dataloader:
        images = images.to(device)
        if images.shape[2] != input_size:
            images = F.interpolate(images, size=(input_size, input_size),
                                   mode="bilinear", align_corners=False)
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        all_logits.append(logits.cpu())
        all_probs.append(probs.cpu())
        all_labels.extend(labels.numpy())

    return {
        "logits": torch.cat(all_logits, 0),
        "probs": torch.cat(all_probs, 0),
        "labels": np.array(all_labels),
    }


def generate_logit_tsne(teacher_data, student_data_dict, class_names, output_dir,
                        perplexity=30, n_samples=2000):
    """
    t-SNE visualization of dark knowledge (logit space).
    Shows how teacher and student logit distributions cluster.
    """
    from sklearn.manifold import TSNE

    print(f"  Generating t-SNE of logit space...")

    # Subsample for speed
    n = min(n_samples, len(teacher_data["logits"]))
    idx = np.random.choice(len(teacher_data["logits"]), n, replace=False)

    teacher_logits = teacher_data["logits"][idx].numpy()
    labels = teacher_data["labels"][idx]

    # Combine teacher + all students
    all_logits = [teacher_logits]
    model_labels = ["Teacher (ConvNeXT)"] * n

    for student_name, sdata in student_data_dict.items():
        if sdata is not None and "logits" in sdata:
            s_logits = sdata["logits"][idx].numpy() if len(sdata["logits"]) > max(idx) else sdata["logits"][:n].numpy()
            all_logits.append(s_logits[:n])
            model_labels.extend([student_name] * min(n, len(s_logits)))

    combined = np.vstack(all_logits)

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, learning_rate="auto", init="pca")
    embedded = tsne.fit_transform(combined)

    # Plot 1: Color by class (teacher only)
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    teacher_emb = embedded[:n]
    cmap = plt.colormaps.get_cmap("tab20")
    unique_labels = np.unique(labels)
    for c in unique_labels:
        mask = labels == c
        name = class_names[c] if c < len(class_names) else f"cls_{c}"
        color = cmap(c / max(len(unique_labels), 1))
        axes[0].scatter(teacher_emb[mask, 0], teacher_emb[mask, 1],
                        c=[color], s=15, alpha=0.6, label=name)
    axes[0].set_title("Teacher Logit Space (colored by class)", fontsize=13)
    axes[0].legend(fontsize=5, ncol=3, loc="upper right", markerscale=2)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    # Plot 2: Color by model (teacher vs students)
    model_colors = {"Teacher (ConvNeXT)": "#e74c3c"}
    palette = ["#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c"]
    for i, sname in enumerate(student_data_dict.keys()):
        model_colors[sname] = palette[i % len(palette)]

    offset = 0
    for mname in ["Teacher (ConvNeXT)"] + list(student_data_dict.keys()):
        count = model_labels.count(mname)
        if count == 0:
            continue
        emb_slice = embedded[offset:offset + count]
        axes[1].scatter(emb_slice[:, 0], emb_slice[:, 1],
                        c=model_colors.get(mname, "#333"), s=10, alpha=0.5, label=mname)
        offset += count
    axes[1].set_title("Logit Space (colored by model)", fontsize=13)
    axes[1].legend(fontsize=9, markerscale=3)
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.suptitle("t-SNE of Dark Knowledge (Logit Distributions)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "tsne_logit_space.png"
    plt.savefig(path, dpi=200)
    plt.savefig(output_dir / "tsne_logit_space.pdf")
    plt.close()
    print(f"  Saved: {path}")


def generate_soft_label_analysis(teacher_data, class_names, output_dir, temperatures=[1, 3, 5, 10, 20]):
    """
    Visualize how soft labels (dark knowledge) change with temperature.
    Shows that higher T reveals more inter-class relationships.
    """
    print(f"  Generating soft label / dark knowledge analysis...")
    logits = teacher_data["logits"]
    labels = teacher_data["labels"]

    # Pick representative samples: one easy, one medium, one hard
    probs_t1 = F.softmax(logits, dim=1)
    top1_conf = probs_t1.max(dim=1).values.numpy()
    easy_idx = np.argmax(top1_conf)  # most confident
    hard_idx = np.argmin(top1_conf)  # least confident
    mid_idx = np.argsort(top1_conf)[len(top1_conf) // 2]  # median

    samples = {"Easy sample": easy_idx, "Medium sample": mid_idx, "Hard sample": hard_idx}

    fig, axes = plt.subplots(len(samples), len(temperatures), figsize=(4 * len(temperatures), 4 * len(samples)))

    for row, (sample_name, s_idx) in enumerate(samples.items()):
        s_logits = logits[s_idx]
        true_label = labels[s_idx]
        true_name = class_names[true_label] if true_label < len(class_names) else f"cls_{true_label}"

        for col, T in enumerate(temperatures):
            soft = F.softmax(s_logits / T, dim=0).numpy()
            ax = axes[row, col] if len(samples) > 1 else axes[col]

            sorted_idx = np.argsort(soft)[::-1]
            top_k = min(10, len(soft))
            top_names = [class_names[i] if i < len(class_names) else f"cls_{i}" for i in sorted_idx[:top_k]]
            top_vals = soft[sorted_idx[:top_k]]

            colors = ["#e74c3c" if sorted_idx[i] == true_label else "#3498db" for i in range(top_k)]
            ax.barh(range(top_k), top_vals, color=colors, edgecolor="white")
            ax.set_yticks(range(top_k))
            ax.set_yticklabels(top_names, fontsize=7)
            ax.invert_yaxis()
            ax.set_xlim(0, max(0.5, top_vals[0] * 1.1))

            if row == 0:
                ax.set_title(f"T = {T}", fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"{sample_name}\n(GT: {true_name})", fontsize=10)

            # Entropy
            entropy = -(soft * np.log(soft + 1e-10)).sum()
            ax.text(0.95, 0.05, f"H={entropy:.2f}", transform=ax.transAxes,
                    fontsize=8, ha="right", va="bottom",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.suptitle("Dark Knowledge: Soft Label Distributions at Different Temperatures\n"
                 "(Red = ground truth class)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "dark_knowledge_temperature.png"
    plt.savefig(path, dpi=200)
    plt.savefig(output_dir / "dark_knowledge_temperature.pdf")
    plt.close()
    print(f"  Saved: {path}")


def generate_routing_distribution(teacher_data, class_names, output_dir):
    """
    Visualize how Adaptive Knowledge Routing distributes samples.
    Shows per-class routing decisions (hard→base, standard→TA, easy→no KD).
    """
    print(f"  Generating routing distribution analysis...")

    logits = teacher_data["logits"]
    labels = teacher_data["labels"]
    probs = F.softmax(logits, dim=1)

    # Compute difficulty
    top2 = probs.topk(2, dim=1).values
    margins = top2[:, 0] - top2[:, 1]
    difficulties = (1.0 - margins).numpy()

    easy_thresh = np.percentile(difficulties, 33)
    hard_thresh = np.percentile(difficulties, 67)

    categories = np.full(len(difficulties), 1)  # standard
    categories[difficulties <= easy_thresh] = 0  # easy
    categories[difficulties >= hard_thresh] = 2  # hard

    # Per-class routing distribution
    unique_classes = np.unique(labels)
    class_routing = {}
    for c in unique_classes:
        mask = labels == c
        name = class_names[c] if c < len(class_names) else f"cls_{c}"
        cats = categories[mask]
        class_routing[name] = {
            "easy": (cats == 0).sum(),
            "standard": (cats == 1).sum(),
            "hard": (cats == 2).sum(),
            "total": len(cats),
            "mean_difficulty": difficulties[mask].mean(),
        }

    # Sort by mean difficulty
    sorted_classes = sorted(class_routing.keys(), key=lambda x: class_routing[x]["mean_difficulty"], reverse=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot 1: Stacked bar — routing distribution per class
    easy_counts = [class_routing[c]["easy"] for c in sorted_classes]
    std_counts = [class_routing[c]["standard"] for c in sorted_classes]
    hard_counts = [class_routing[c]["hard"] for c in sorted_classes]
    totals = [class_routing[c]["total"] for c in sorted_classes]

    # Normalize to percentages
    easy_pct = [100 * e / t for e, t in zip(easy_counts, totals)]
    std_pct = [100 * s / t for s, t in zip(std_counts, totals)]
    hard_pct = [100 * h / t for h, t in zip(hard_counts, totals)]

    y_pos = range(len(sorted_classes))
    axes[0].barh(y_pos, easy_pct, color="#2ecc71", label="Easy (no KD)", edgecolor="white")
    axes[0].barh(y_pos, std_pct, left=easy_pct, color="#f39c12", label="Standard (→TA)", edgecolor="white")
    left2 = [e + s for e, s in zip(easy_pct, std_pct)]
    axes[0].barh(y_pos, hard_pct, left=left2, color="#e74c3c", label="Hard (→Base Teacher)", edgecolor="white")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(sorted_classes, fontsize=7)
    axes[0].set_xlabel("Routing Distribution (%)", fontsize=11)
    axes[0].set_title("Per-Class Routing Distribution\n(sorted by mean difficulty)", fontsize=13)
    axes[0].legend(fontsize=9, loc="lower right")
    axes[0].invert_yaxis()

    # Plot 2: Difficulty histogram with routing zones
    axes[1].hist(difficulties, bins=60, color="#3498db", edgecolor="white", alpha=0.8, density=True)
    axes[1].axvline(easy_thresh, color="#2ecc71", ls="--", lw=2, label=f"Easy < {easy_thresh:.3f}")
    axes[1].axvline(hard_thresh, color="#e74c3c", ls="--", lw=2, label=f"Hard > {hard_thresh:.3f}")
    axes[1].axvspan(0, easy_thresh, alpha=0.1, color="#2ecc71")
    axes[1].axvspan(hard_thresh, 1.0, alpha=0.1, color="#e74c3c")
    axes[1].axvspan(easy_thresh, hard_thresh, alpha=0.1, color="#f39c12")
    axes[1].set_xlabel("Difficulty Score (d = 1 - margin)", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)
    axes[1].set_title("Sample Difficulty Distribution with Routing Zones", fontsize=13)
    axes[1].legend(fontsize=9)

    plt.suptitle("Adaptive Knowledge Routing — Sample Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = output_dir / "routing_distribution.png"
    plt.savefig(path, dpi=200)
    plt.savefig(output_dir / "routing_distribution.pdf")
    plt.close()
    print(f"  Saved: {path}")


def generate_perclass_comparison(all_results, class_names, output_dir):
    """
    Per-class accuracy comparison across KD methods.
    Shows which classes benefit most from each method.
    """
    print(f"  Generating per-class accuracy comparison...")

    # Group by student
    students = {}
    for r in all_results:
        sname = r.get("student", "?")
        if sname not in students:
            students[sname] = {}
        method = r.get("method", "?")
        if "per_class_acc" in r and r["per_class_acc"]:
            students[sname][method] = r["per_class_acc"]

    for student_name, methods in students.items():
        if len(methods) < 2:
            continue

        fig, ax = plt.subplots(figsize=(16, 8))

        method_colors = {"none": "#95a5a6", "vanilla": "#3498db", "takd": "#2ecc71", "akr": "#e74c3c"}
        method_labels = {"none": "Baseline", "vanilla": "Vanilla KD", "takd": "TAKD", "akr": "AKR (Proposed)"}

        all_classes = set()
        for m_data in methods.values():
            all_classes.update(m_data.keys())
        all_classes = sorted(all_classes)

        x = np.arange(len(all_classes))
        n_methods = len(methods)
        width = 0.8 / n_methods

        for i, (method, per_class) in enumerate(methods.items()):
            vals = [per_class.get(c, 0) for c in all_classes]
            names = [class_names[int(c)] if isinstance(c, (int, np.integer)) and int(c) < len(class_names)
                     else str(c) for c in all_classes]
            ax.bar(x + i * width - 0.4 + width / 2, vals, width,
                   label=method_labels.get(method, method),
                   color=method_colors.get(method, "#333"),
                   edgecolor="white", linewidth=0.3)

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("Per-Class Accuracy (%)", fontsize=11)
        ax.set_title(f"Per-Class Accuracy — {student_name}", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"perclass_accuracy_{student_name.replace('-','_')}.png"
        plt.savefig(path, dpi=150)
        plt.savefig(output_dir / f"perclass_accuracy_{student_name.replace('-','_')}.pdf")
        plt.close()
        print(f"  Saved: {path}")


def generate_training_dynamics(all_results, output_dir):
    """Plot training loss and accuracy curves for all experiments."""
    print(f"  Generating training dynamics plots...")

    students = {}
    for r in all_results:
        sname = r.get("student", "?")
        if sname not in students:
            students[sname] = []
        if "history" in r and r["history"]:
            students[sname].append(r)

    method_colors = {"none": "#95a5a6", "vanilla": "#3498db", "takd": "#2ecc71", "akr": "#e74c3c"}
    method_labels = {"none": "Baseline", "vanilla": "Vanilla KD", "takd": "TAKD", "akr": "AKR (Proposed)"}

    for student_name, results in students.items():
        if not results:
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for r in results:
            method = r.get("method", "?")
            hist = r["history"]
            color = method_colors.get(method, "#333")
            label = method_labels.get(method, method)

            # Loss curve
            if hist.get("train_loss"):
                axes[0].plot(hist["train_loss"], color=color, label=label, alpha=0.8)

            # Val accuracy curve
            if hist.get("val_acc"):
                # val_acc is recorded every 5 epochs
                val_epochs = list(range(4, len(hist.get("train_loss", [])), 5))
                if len(val_epochs) > len(hist["val_acc"]):
                    val_epochs = val_epochs[:len(hist["val_acc"])]
                elif len(val_epochs) < len(hist["val_acc"]):
                    val_epochs = list(range(len(hist["val_acc"])))
                axes[1].plot(val_epochs, hist["val_acc"], color=color, label=label,
                             marker="o", markersize=3, alpha=0.8)

        axes[0].set_xlabel("Epoch", fontsize=11)
        axes[0].set_ylabel("Training Loss", fontsize=11)
        axes[0].set_title(f"Training Loss — {student_name}", fontsize=13)
        axes[0].legend(fontsize=9)
        axes[0].grid(alpha=0.3)

        axes[1].set_xlabel("Epoch", fontsize=11)
        axes[1].set_ylabel("Validation Accuracy (%)", fontsize=11)
        axes[1].set_title(f"Validation Accuracy — {student_name}", fontsize=13)
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        path = output_dir / f"training_dynamics_{student_name.replace('-','_')}.png"
        plt.savefig(path, dpi=150)
        plt.savefig(output_dir / f"training_dynamics_{student_name.replace('-','_')}.pdf")
        plt.close()
        print(f"  Saved: {path}")


def generate_kl_divergence_heatmap(teacher_data, class_names, output_dir):
    """
    KL divergence between teacher's soft predictions for each class pair.
    Shows which classes the teacher considers similar (low KL) — this IS the dark knowledge.
    """
    print(f"  Generating KL divergence inter-class heatmap...")

    logits = teacher_data["logits"]
    labels = teacher_data["labels"]
    probs = F.softmax(logits / 3.0, dim=1)  # T=3 for visible dark knowledge

    unique_classes = np.unique(labels)
    nc = len(unique_classes)

    # Compute mean soft distribution per class
    class_means = {}
    for c in unique_classes:
        mask = labels == c
        class_means[c] = probs[mask].mean(dim=0)

    # KL divergence matrix
    kl_matrix = np.zeros((nc, nc))
    for i, ci in enumerate(unique_classes):
        for j, cj in enumerate(unique_classes):
            p = class_means[ci]
            q = class_means[cj]
            kl = F.kl_div(q.log(), p, reduction="sum").item()
            kl_matrix[i, j] = kl

    names = [class_names[c] if c < len(class_names) else f"cls_{c}" for c in unique_classes]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(kl_matrix, xticklabels=names, yticklabels=names,
                cmap="YlOrRd", ax=ax, annot=False, linewidths=0.5)
    ax.set_title("Inter-Class KL Divergence (Teacher Soft Labels, T=3)\n"
                 "Low KL = classes teacher considers similar (dark knowledge)",
                 fontsize=13, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()

    path = output_dir / "kl_divergence_interclass.png"
    plt.savefig(path, dpi=200)
    plt.savefig(output_dir / "kl_divergence_interclass.pdf")
    plt.close()
    print(f"  Saved: {path}")


def generate_improvement_waterfall(all_results, output_dir):
    """
    Waterfall chart showing improvement of each KD method over baseline per student.
    """
    print(f"  Generating improvement waterfall chart...")

    students = {}
    for r in all_results:
        sname = r.get("student", "?")
        if sname not in students:
            students[sname] = {}
        students[sname][r.get("method", "?")] = r.get("accuracy", 0)

    method_order = ["none", "vanilla", "takd", "akr"]
    method_labels = {"none": "Baseline", "vanilla": "Vanilla KD", "takd": "TAKD", "akr": "AKR (Proposed)"}
    method_colors = {"none": "#95a5a6", "vanilla": "#3498db", "takd": "#2ecc71", "akr": "#e74c3c"}

    fig, ax = plt.subplots(figsize=(12, 6))

    x_offset = 0
    xticks = []
    xtick_labels = []

    for student_name, methods in students.items():
        baseline = methods.get("none", 0)

        for method in method_order:
            if method not in methods:
                continue
            acc = methods[method]
            delta = acc - baseline if method != "none" else 0

            if method == "none":
                ax.bar(x_offset, acc, color=method_colors[method], edgecolor="white", width=0.7)
                ax.text(x_offset, acc + 0.5, f"{acc:.1f}%", ha="center", fontsize=8)
            else:
                color = "#2ecc71" if delta > 0 else "#e74c3c"
                ax.bar(x_offset, delta, bottom=baseline, color=color,
                       edgecolor="white", width=0.7, alpha=0.8)
                ax.text(x_offset, baseline + delta + (0.5 if delta >= 0 else -1.5),
                        f"{'+' if delta > 0 else ''}{delta:.1f}%", ha="center", fontsize=8,
                        fontweight="bold" if method == "akr" else "normal")
                ax.hlines(baseline, x_offset - 0.4, x_offset + 0.4, colors="gray",
                          linestyles="dotted", linewidth=0.5)

            xticks.append(x_offset)
            xtick_labels.append(f"{method_labels.get(method, method)}\n({student_name})")
            x_offset += 1

        x_offset += 0.5  # gap between students

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("KD Method Improvement Over Baseline", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "improvement_waterfall.png"
    plt.savefig(path, dpi=150)
    plt.savefig(output_dir / "improvement_waterfall.pdf")
    plt.close()
    print(f"  Saved: {path}")


def run_comprehensive_analysis(all_results, base_teacher, ta_teacher,
                               test_loader, device, class_names,
                               output_dir, teacher_input_size=224):
    """
    Run all advanced analysis — generates all figures for paper & dissertation.
    """
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Comprehensive Analysis for Paper & Dissertation")
    print(f"{'='*70}")

    # 1. Extract teacher logits/features
    print(f"\n  [1/7] Extracting teacher dark knowledge...")
    teacher_data = extract_logits_and_features(base_teacher, test_loader, device,
                                               input_size=teacher_input_size)

    # 2. Dark knowledge visualization (soft labels at different temperatures)
    print(f"\n  [2/7] Dark knowledge analysis...")
    generate_soft_label_analysis(teacher_data, class_names, fig_dir)

    # 3. KL divergence inter-class heatmap
    print(f"\n  [3/7] KL divergence heatmap...")
    generate_kl_divergence_heatmap(teacher_data, class_names, fig_dir)

    # 4. Routing distribution
    print(f"\n  [4/7] Routing distribution analysis...")
    generate_routing_distribution(teacher_data, class_names, fig_dir)

    # 5. t-SNE of logit space
    print(f"\n  [5/7] t-SNE visualization...")
    # Extract student logits for comparison (from saved results with logits)
    student_logit_data = {}
    for r in all_results:
        if "logits" in r and r["logits"] is not None:
            key = f"{r.get('method','?')}_{r.get('student','?')}"
            student_logit_data[key] = r
    if student_logit_data:
        generate_logit_tsne(teacher_data, student_logit_data, class_names, fig_dir)
    else:
        print(f"    Skip t-SNE: no student logits available (from checkpoint)")

    # 6. Per-class accuracy comparison
    print(f"\n  [6/7] Per-class accuracy comparison...")
    generate_perclass_comparison(all_results, class_names, fig_dir)

    # 7. Training dynamics
    print(f"\n  [7/7] Training dynamics...")
    generate_training_dynamics(all_results, fig_dir)

    # 8. Improvement waterfall
    generate_improvement_waterfall(all_results, fig_dir)

    # 9. Pareto front (already exists but regenerate)
    generate_pareto_plot(all_results, fig_dir)

    print(f"\n  All analysis complete. Figures saved to: {fig_dir}/")
    print(f"  Files generated:")
    for f in sorted(fig_dir.glob("*.png")):
        print(f"    {f.name}")


# ============================================================================
# CHECKPOINT SYSTEM
# ============================================================================

def save_ckpt(output_dir, name, data):
    ckpt_dir = Path(output_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}.json"
    serializable = json.loads(json.dumps(data, default=str))
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_ckpt(output_dir, name):
    path = Path(output_dir) / "checkpoints" / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f), True
    return None, False


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    p = argparse.ArgumentParser(description="KD Framework for WaRP-C Waste Classification")
    p.add_argument("--data", type=str, default=None,
                   help="Path to WaRP-C crops root (with train/ and test/ subdirs)")
    p.add_argument("--base-teacher-weights", type=str, default=None,
                   help="Path to base teacher weights (ConvNeXT-Tiny)")
    p.add_argument("--base-teacher-name", type=str, default="ConvNeXT-Tiny",
                   choices=list(TEACHER_CONFIGS.keys()))
    p.add_argument("--ta-teacher-weights", type=str, default=None,
                   help="Path to TA teacher weights (EfficientNet-B0 or MobileNetV3)")
    p.add_argument("--ta-teacher-name", type=str, default="EfficientNet-B0",
                   choices=list(TEACHER_CONFIGS.keys()))
    p.add_argument("--students", nargs="+",
                   default=["WasteNet-Micro", "WasteNet-Small", "MobileNetV3-Small"],
                   choices=list(STUDENT_CONFIGS.keys()))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--output", type=str, default="./results_kd_framework")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--skip-baseline", action="store_true")
    p.add_argument("--skip-vanilla", action="store_true")
    p.add_argument("--skip-takd", action="store_true")
    p.add_argument("--skip-akr", action="store_true")
    p.add_argument("--rerun-from", type=int, default=0,
                   help="Force re-run from phase N (1=difficulty, 2=experiments, 6=analysis)")
    p.add_argument("--rerun-exp", type=str, default=None,
                   help="Force re-run a specific experiment by name "
                        "(e.g., Baseline_WasteNet-Micro, AKR_MobileNetV3-Small)")
    p.add_argument("--status", action="store_true",
                   help="Show checkpoint status and exit")
    return p.parse_args()


def experiment_ckpt_key(student_name, method):
    """Generate a unique checkpoint key for each experiment."""
    return f"exp_{student_name}_{method}".replace("-", "_").replace(" ", "_")


def show_status(output_dir):
    """Show which experiments have completed checkpoints."""
    ckpt_dir = Path(output_dir) / "checkpoints"
    if not ckpt_dir.exists():
        print("  No checkpoints found.")
        return

    print(f"\n  {'='*60}")
    print(f"  CHECKPOINT STATUS ({ckpt_dir})")
    print(f"  {'='*60}")

    # Phase 1
    p1 = ckpt_dir / "phase1_difficulty.json"
    status = "DONE" if p1.exists() else "PENDING"
    print(f"  Phase 1 (Difficulty Estimation):  [{status}]")

    # Individual experiments
    for f in sorted(ckpt_dir.glob("exp_*.json")):
        name = f.stem.replace("exp_", "")
        mtime = datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        with open(f) as fh:
            data = json.load(fh)
        acc = data.get("accuracy", "?")
        print(f"  Experiment {name}:  [DONE] Acc={acc}% ({mtime})")

    # Analysis
    p6 = ckpt_dir / "phase_analysis.json"
    status = "DONE" if p6.exists() else "PENDING"
    print(f"  Phase Analysis (tables/figures):  [{status}]")

    print(f"  {'='*60}")
    print(f"\n  To re-run a specific experiment:")
    print(f"    --rerun-exp Baseline_WasteNet-Micro")
    print(f"  To re-run from Phase 1 (difficulty):")
    print(f"    --rerun-from 1")
    print(f"  To reset everything:")
    print(f"    rm -rf {ckpt_dir}/\n")


def main():
    args = parse_args()

    # Show status and exit
    if args.status:
        show_status(args.output)
        return

    # Validate required args for training mode
    if not args.data:
        print("ERROR: --data is required (path to WaRP-C crops root)")
        sys.exit(1)
    if not args.base_teacher_weights:
        print("ERROR: --base-teacher-weights is required")
        sys.exit(1)

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    (output_dir / "figures").mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    (output_dir / "weights").mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    rerun_from = args.rerun_from
    rerun_exp = getattr(args, 'rerun_exp', None)

    print("\n" + "=" * 70)
    print("  Adaptive Knowledge Routing — KD Framework")
    print("  Dataset: WaRP-C (28 waste classes)")
    print("  Target: Lightweight student for ESP32-CAM")
    print("  ** Checkpoint-enabled: crash-safe, resumable **")
    print("=" * 70)

    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Students: {', '.join(args.students)}")
    print(f"  Epochs: {args.epochs}, LR: {args.lr}")
    print(f"  Base Teacher: {args.base_teacher_name}")
    print(f"  TA Teacher: {args.ta_teacher_name}")
    if rerun_from > 0:
        print(f"  Re-run from: Phase {rerun_from}")
    if rerun_exp:
        print(f"  Re-run experiment: {rerun_exp}")

    # ================================================================
    # PHASE 1: Load teachers and compute difficulty scores
    # ================================================================
    # Teachers are always loaded (needed for KD experiments)
    train_dir = Path(args.data) / "train"
    num_classes = len([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"\n  Detected {num_classes} classes")

    base_teacher = load_teacher(args.base_teacher_name, args.base_teacher_weights,
                                num_classes, device)

    ta_teacher = None
    if args.ta_teacher_weights:
        ta_teacher = load_teacher(args.ta_teacher_name, args.ta_teacher_weights,
                                  num_classes, device)
    else:
        print("  No TA teacher weights provided. TAKD and AKR will use base teacher as TA.")
        ta_teacher = base_teacher

    # Difficulty estimation — check checkpoint
    phase1_data, phase1_exists = load_ckpt(output_dir, "phase1_difficulty")
    difficulty_data = None

    if phase1_exists and rerun_from < 1:
        print(f"\n  [Phase 1] LOADED from checkpoint (difficulty already computed)")
        # Still need to recompute difficulty_data for AKR training
        # because numpy arrays aren't in the JSON checkpoint
        print(f"  [Phase 1] Recomputing difficulty scores from teacher...")
        teacher_input_size = TEACHER_CONFIGS[args.base_teacher_name]["input_size"]
        diff_loader, _, _, _ = create_dataloaders(
            args.data, input_size=teacher_input_size, batch_size=args.batch_size,
            num_workers=args.workers,
        )
        estimator = DifficultyEstimator(base_teacher, device)
        difficulty_data = estimator.compute_difficulty(diff_loader)
        categories = estimator.categorize_samples(difficulty_data["difficulty"])
    else:
        print(f"\n{'='*70}")
        print(f"  Phase 1: Compute Difficulty Scores")
        print(f"{'='*70}")

        teacher_input_size = TEACHER_CONFIGS[args.base_teacher_name]["input_size"]
        diff_loader, _, _, _ = create_dataloaders(
            args.data, input_size=teacher_input_size, batch_size=args.batch_size,
            num_workers=args.workers,
        )
        estimator = DifficultyEstimator(base_teacher, device)
        difficulty_data = estimator.compute_difficulty(diff_loader)
        categories = estimator.categorize_samples(difficulty_data["difficulty"])

        generate_difficulty_analysis(difficulty_data, CLASS_NAMES,
                                     output_dir / "figures")

        save_ckpt(output_dir, "phase1_difficulty", {
            "num_classes": num_classes,
            "difficulty_stats": {
                "mean": float(difficulty_data["difficulty"].mean()),
                "std": float(difficulty_data["difficulty"].std()),
                "easy_count": int((categories == 0).sum()),
                "standard_count": int((categories == 1).sum()),
                "hard_count": int((categories == 2).sum()),
            },
        })

    # ================================================================
    # PHASE 2-5: Run experiments (each individually checkpointed)
    # ================================================================
    all_results = []

    # Build list of all experiments to run
    experiments_to_run = []
    for student_name in args.students:
        if not args.skip_baseline:
            experiments_to_run.append((student_name, "none", "Baseline"))
        if not args.skip_vanilla:
            experiments_to_run.append((student_name, "vanilla", "VanillaKD"))
        if not args.skip_takd and ta_teacher is not base_teacher:
            experiments_to_run.append((student_name, "takd", "TAKD"))
        if not args.skip_akr:
            experiments_to_run.append((student_name, "akr", "AKR"))

    total_experiments = len(experiments_to_run)
    completed = 0
    skipped = 0

    for idx, (student_name, method, label) in enumerate(experiments_to_run, 1):
        exp_name = f"{label}_{student_name}"
        ckpt_key = experiment_ckpt_key(student_name, method)

        # Check if this experiment already has a checkpoint
        existing, exists = load_ckpt(output_dir, ckpt_key)

        # Determine if we should skip
        should_skip = exists and rerun_from < 2
        if rerun_exp and rerun_exp == exp_name:
            should_skip = False  # Force re-run this specific experiment

        if should_skip:
            print(f"\n  [{idx}/{total_experiments}] {exp_name}: LOADED from checkpoint "
                  f"(Acc={existing.get('accuracy', '?')}%)")
            all_results.append(existing)
            skipped += 1
            continue

        # Clear old checkpoint if re-running
        ckpt_path = output_dir / "checkpoints" / f"{ckpt_key}.json"
        if ckpt_path.exists():
            ckpt_path.unlink()

        print(f"\n  [{idx}/{total_experiments}] {exp_name}: RUNNING")

        scfg = STUDENT_CONFIGS[student_name]
        input_size = scfg["input_size"]

        train_loader, test_loader, train_ds, test_ds = create_dataloaders(
            args.data, input_size=input_size, batch_size=args.batch_size,
            num_workers=args.workers,
        )
        actual_nc = len(train_ds.classes)
        student_model = scfg["factory"](actual_nc)

        # Determine teacher/TA args based on method
        exp_kwargs = {
            "epochs": args.epochs,
            "lr": args.lr,
            "kd_method": method,
        }
        if method == "vanilla":
            exp_kwargs["teacher"] = base_teacher
            exp_kwargs["temperature"] = args.temperature
            exp_kwargs["alpha"] = args.alpha
            exp_kwargs["teacher_input_size"] = teacher_input_size
        elif method == "takd":
            exp_kwargs["ta_teacher"] = ta_teacher
            exp_kwargs["temperature"] = args.temperature
            exp_kwargs["alpha"] = args.alpha
            exp_kwargs["teacher_input_size"] = teacher_input_size
        elif method == "akr":
            exp_kwargs["teacher"] = base_teacher
            exp_kwargs["ta_teacher"] = ta_teacher
            exp_kwargs["temperature"] = args.temperature
            exp_kwargs["alpha"] = args.alpha
            exp_kwargs["teacher_input_size"] = teacher_input_size

        result, model = run_experiment(
            exp_name, student_name, student_model,
            train_loader, test_loader, device,
            **exp_kwargs,
        )

        # Save weights
        weight_name = f"{method}_{student_name}".replace("-", "_").lower()
        torch.save(model.state_dict(),
                   output_dir / "weights" / f"{weight_name}.pt")

        # Generate confusion matrix for baseline and AKR
        if method in ("none", "akr"):
            generate_confusion_matrix(result, CLASS_NAMES, output_dir / "figures",
                                      exp_name)

        # Save checkpoint (strip non-serializable fields)
        result_serializable = {
            k: v for k, v in result.items()
            if k not in ("predictions", "labels", "logits", "history", "per_class_acc")
        }
        save_ckpt(output_dir, ckpt_key, result_serializable)
        all_results.append(result_serializable)
        completed += 1

        print(f"  [{idx}/{total_experiments}] {exp_name}: DONE "
              f"(Acc={result['accuracy']:.2f}%) — checkpointed")

    print(f"\n  Experiments: {completed} ran, {skipped} loaded from checkpoint, "
          f"{total_experiments} total")

    # ================================================================
    # PHASE 6: Generate all analysis (tables, Pareto plot)
    # ================================================================
    analysis_data, analysis_exists = load_ckpt(output_dir, "phase_analysis")

    if analysis_exists and rerun_from < 6 and not rerun_exp:
        print(f"\n  [Phase Analysis] LOADED from checkpoint (tables/figures exist)")
    else:
        print(f"\n{'='*70}")
        print(f"  Phase Analysis: Generate Tables, Figures & Advanced Visualizations")
        print(f"{'='*70}")

        # Tables
        df = generate_comparison_table(all_results, output_dir / "tables")

        # Create test dataloader for teacher analysis
        teacher_test_loader, _, _, _ = create_dataloaders(
            args.data, input_size=teacher_input_size,
            batch_size=args.batch_size, num_workers=args.workers,
        )

        # Comprehensive analysis: t-SNE, dark knowledge, routing, KL heatmap, etc.
        run_comprehensive_analysis(
            all_results, base_teacher, ta_teacher,
            teacher_test_loader, device, CLASS_NAMES,
            output_dir, teacher_input_size=teacher_input_size,
        )

        save_ckpt(output_dir, "phase_analysis", {"status": "done"})

    # Save all results master file
    save_ckpt(output_dir, "all_results", all_results)

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")

    for student_name in args.students:
        student_results = [r for r in all_results if r.get("student") == student_name]
        if not student_results:
            continue
        print(f"\n  {student_name}:")
        print(f"    {'Method':<20} {'Accuracy':>10} {'Params':>12} {'FLOPs(M)':>10} {'Latency':>10}")
        print(f"    {'-'*62}")
        for r in student_results:
            acc = r.get('accuracy', 0)
            params = r.get('params', 0)
            flops = r.get('flops', 0)
            latency = r.get('latency_ms', 0)
            print(f"    {r.get('method','?'):<20} {acc:>9.2f}% {params:>11,} "
                  f"{flops/1e6:>9.1f} {latency:>9.2f}ms")

        baseline = next((r for r in student_results if r.get("method") == "none"), None)
        akr = next((r for r in student_results if r.get("method") == "akr"), None)
        if baseline and akr:
            delta = akr["accuracy"] - baseline["accuracy"]
            print(f"    AKR vs baseline: {'+' if delta > 0 else ''}{delta:.2f}%")

    print(f"\n  Output: {output_dir}/")
    print(f"  To check status:       python train_kd_framework.py --status --output {output_dir}")
    print(f"  To re-run 1 experiment: --rerun-exp Baseline_WasteNet-Micro")
    print(f"  To re-run from phase:   --rerun-from 2")
    print(f"\n{'='*70}")
    print(f"  Done!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
