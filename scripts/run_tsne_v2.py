"""t-SNE with super-group coloring for clarity."""
import os, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

CKPT_DIR = Path(os.path.expanduser("~/KD/akr_qualitative/checkpoints"))
DATA_DIR = Path(os.path.expanduser("~/KD/wasterecycle/results_teacher_yolo11/warp_c_crops"))
OUTPUT_DIR = Path(os.path.expanduser("~/KD/akr_qualitative/output"))
NUM_CLASSES = 28
SEED = 42

class MicroConvBlock(nn.Module):
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
    def __init__(self, num_classes=28, width_mult=1.0):
        super().__init__()
        def ch(c): return max(8, int(c * width_mult))
        self.features = nn.Sequential(
            nn.Conv2d(3, ch(16), 3, 2, 1, bias=False), nn.BatchNorm2d(ch(16)), nn.ReLU6(inplace=True),
            MicroConvBlock(ch(16), ch(24), stride=2),
            MicroConvBlock(ch(24), ch(32), stride=2), MicroConvBlock(ch(32), ch(32), stride=1),
            MicroConvBlock(ch(32), ch(48), stride=2), MicroConvBlock(ch(48), ch(48), stride=1),
            MicroConvBlock(ch(48), ch(64), stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch(64), num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)

def load_student(name, method, device):
    path = CKPT_DIR / f"{method}_{name}.pt"
    model = WasteNetMicro(NUM_CLASSES)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, 96

def extract_features(model, loader, device):
    feats_list = []; labs_list = []; preds_list = []
    def hook_fn(module, input, output):
        feats_list.append(input[0].detach().cpu())
    handle = model.classifier.register_forward_hook(hook_fn)
    with torch.no_grad():
        for imgs, labs in loader:
            out = model(imgs.to(device))
            labs_list.append(labs); preds_list.append(out.argmax(1).cpu())
    handle.remove()
    return torch.cat(feats_list).numpy(), torch.cat(labs_list).numpy(), torch.cat(preds_list).numpy()

def main():
    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda:0")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = ImageFolder(str(DATA_DIR / "test"))
    class_names = ds.classes

    # Build super-group mapping
    group_map = {}  # class_idx -> group_name
    for i, cn in enumerate(class_names):
        parts = cn.split("-")
        prefix = parts[0]
        if prefix == "juice":
            group_map[i] = "cardboard"
        elif prefix == "milk":
            group_map[i] = "cardboard"
        else:
            group_map[i] = prefix

    # Is it a "full" variant?
    is_full = {}
    for i, cn in enumerate(class_names):
        is_full[i] = "-full" in cn

    group_colors = {
        "bottle":    "#3498db",
        "glass":     "#2ecc71",
        "cans":      "#e74c3c",
        "detergent": "#9b59b6",
        "canister":  "#f39c12",
        "cardboard": "#e67e22",
    }
    group_order = ["bottle", "glass", "cans", "detergent", "canister", "cardboard"]

    methods = ["none", "vanilla", "akr"]
    mlabels = {"none": "Baseline (No KD)", "vanilla": "Vanilla KD", "akr": "AKR (Proposed)"}

    tf = transforms.Compose([
        transforms.Resize((96, 96)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    loader = DataLoader(ImageFolder(str(DATA_DIR / "test"), transform=tf),
                        batch_size=64, shuffle=False, num_workers=2)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))

    for idx, m in enumerate(methods):
        model, _ = load_student("wastenet_micro", m, device)
        feats, labs, preds = extract_features(model, loader, device)
        f1 = f1_score(labs, preds, average="macro") * 100
        n_err = (preds != labs).sum()

        tsne = TSNE(n_components=2, perplexity=30, random_state=SEED,
                     max_iter=1000, learning_rate="auto", init="pca")
        emb = tsne.fit_transform(feats)

        # Plot per super-group
        for gname in group_order:
            gc = group_colors[gname]
            # Empty variants
            empty_mask = np.zeros(len(labs), dtype=bool)
            full_mask = np.zeros(len(labs), dtype=bool)
            for ci in range(NUM_CLASSES):
                if group_map[ci] == gname:
                    ci_mask = labs == ci
                    if is_full[ci]:
                        full_mask |= ci_mask
                    else:
                        empty_mask |= ci_mask

            if empty_mask.any():
                axes[idx].scatter(emb[empty_mask, 0], emb[empty_mask, 1],
                                 c=gc, marker="o", s=18, alpha=0.6, edgecolors="none")
            if full_mask.any():
                axes[idx].scatter(emb[full_mask, 0], emb[full_mask, 1],
                                 c=gc, marker="^", s=22, alpha=0.6, edgecolors="none")

        # Misclassified overlay
        wrong = preds != labs
        if wrong.any():
            axes[idx].scatter(emb[wrong, 0], emb[wrong, 1], c="none",
                             edgecolors="red", s=40, linewidths=0.8,
                             marker="o", alpha=0.4, zorder=5)

        axes[idx].set_title(f"{mlabels[m]}\nF1={f1:.1f}%  ({n_err} errors)",
                           fontsize=13, fontweight="bold")
        axes[idx].set_xticks([]); axes[idx].set_yticks([])

        # Grid for visual reference
        axes[idx].grid(True, alpha=0.15)

    # Legend
    handles = []
    for gname in group_order:
        handles.append(Patch(facecolor=group_colors[gname], label=gname.capitalize()))
    handles.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                          markersize=7, label="empty"))
    handles.append(Line2D([0], [0], marker="^", color="w", markerfacecolor="gray",
                          markersize=7, label="full"))
    handles.append(Line2D([0], [0], marker="o", color="w", markeredgecolor="red",
                          markerfacecolor="none", markersize=8, linewidth=0,
                          markeredgewidth=1.2, label="misclassified"))

    axes[0].legend(handles=handles, fontsize=9, loc="lower left",
                   ncol=2, framealpha=0.9, edgecolor="gray")

    plt.suptitle("t-SNE Feature Embeddings: Impact of Knowledge Distillation\n"
                 "WasteNet-Micro (21.6K params), WaRP-C 28 classes, colored by material super-group",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsne_supergroup.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "tsne_supergroup.pdf", bbox_inches="tight")
    plt.close()
    print("Saved tsne_supergroup.png")

    # Print group-level F1 for narrative
    print("\n=== Per Super-Group Accuracy ===")
    for m in methods:
        model, _ = load_student("wastenet_micro", m, device)
        feats, labs, preds = extract_features(model, loader, device)
        print(f"\n{mlabels[m]}:")
        for gname in group_order:
            mask = np.array([group_map[l] == gname for l in labs])
            if mask.any():
                acc = 100 * (preds[mask] == labs[mask]).mean()
                n = mask.sum()
                print(f"  {gname:<12} acc={acc:.1f}% (n={n})")

if __name__ == "__main__":
    main()
