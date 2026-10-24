"""AKR Qualitative Analysis — A100 server version (K=28 WaRP-C)."""
import os, sys, random, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

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


class WasteNetSmall(nn.Module):
    def __init__(self, num_classes=28, width_mult=1.0):
        super().__init__()
        def ch(c): return max(8, int(c * width_mult))
        self.features = nn.Sequential(
            nn.Conv2d(3, ch(24), 3, 2, 1, bias=False), nn.BatchNorm2d(ch(24)), nn.ReLU6(inplace=True),
            MicroConvBlock(ch(24), ch(32), stride=2), MicroConvBlock(ch(32), ch(32), stride=1),
            MicroConvBlock(ch(32), ch(48), stride=2), MicroConvBlock(ch(48), ch(48), stride=1),
            MicroConvBlock(ch(48), ch(48), stride=1),
            MicroConvBlock(ch(48), ch(72), stride=2), MicroConvBlock(ch(72), ch(72), stride=1),
            MicroConvBlock(ch(72), ch(96), stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch(96), num_classes)
    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_student(name, method, device):
    path = CKPT_DIR / f"{method}_{name}.pt"
    if "micro" in name:
        model = WasteNetMicro(NUM_CLASSES); input_size = 96
    elif "small" in name and "mobilenet" not in name:
        model = WasteNetSmall(NUM_CLASSES); input_size = 128
    else:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, NUM_CLASSES)
        input_size = 224
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model, input_size


def load_teacher(name, device):
    if "convnext" in name:
        model = models.convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, NUM_CLASSES)
        path = Path(os.path.expanduser("~/KD/results_k_ablation/teachers_warpc_k28/teacher_convnext-tiny.pt"))
    else:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
        path = Path(os.path.expanduser("~/KD/results_k_ablation/teachers_warpc_k28/teacher_efficientnet-b0.pt"))
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


# ═══════════════════════════════════════════════════════════════
# 1. GRAD-CAM
# ═══════════════════════════════════════════════════════════════
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None; self.activations = None
        target_layer.register_forward_hook(lambda m,i,o: setattr(self, "activations", o.detach()))
        target_layer.register_full_backward_hook(lambda m,gi,go: setattr(self, "gradients", go[0].detach()))

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        out = self.model(x)
        if class_idx is None: class_idx = out.argmax(dim=1)
        one_hot = torch.zeros_like(out)
        for i in range(len(class_idx)): one_hot[i, class_idx[i]] = 1.0
        out.backward(gradient=one_hot)
        w = self.gradients.mean(dim=(2,3), keepdim=True)
        cam = F.relu((w * self.activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False).squeeze(1)
        for i in range(cam.shape[0]):
            c = cam[i]
            if c.max() > 0: cam[i] = (c - c.min()) / (c.max() - c.min())
        return cam, out


def generate_gradcam(device, class_names):
    print("\n[1/3] GradCAM comparison...")
    arch = "wastenet_micro"
    methods = ["none", "vanilla", "akr"]
    mlabels = {"none": "Baseline", "vanilla": "Vanilla KD", "akr": "AKR (Proposed)"}

    mdls = {}
    for m in methods:
        model, sz = load_student(arch, m, device)
        mdls[m] = {"model": model, "cam": GradCAM(model, model.features[-1]), "size": sz}

    # Select 8 diverse classes for display
    display_classes = ["bottle-blue", "bottle-transp", "cans", "glass-dark",
                       "detergent-box", "juice-cardboard", "canister", "milk-cardboard"]
    display_classes = [c for c in display_classes if c in class_names][:8]

    n_per_class = 1
    raw_tf = transforms.Compose([transforms.Resize((96, 96)), transforms.ToTensor()])

    nrows = len(display_classes) * n_per_class
    fig, axes = plt.subplots(nrows, 1+len(methods), figsize=(3.5*(1+len(methods)), 2.8*nrows))
    if nrows == 1: axes = axes[np.newaxis, :]

    row = 0
    for cn in display_classes:
        ci = class_names.index(cn)
        cls_dir = DATA_DIR / "test" / cn
        imgs = sorted(cls_dir.iterdir())[:n_per_class]
        for img_path in imgs:
            pil = Image.open(img_path).convert("RGB")
            raw = raw_tf(pil)
            axes[row, 0].imshow(raw.permute(1,2,0).numpy())
            axes[row, 0].set_title(cn, fontsize=9, fontweight="bold")
            axes[row, 0].axis("off")

            for mi, m in enumerate(methods, 1):
                sz = mdls[m]["size"]
                tf = transforms.Compose([
                    transforms.Resize((sz,sz)), transforms.ToTensor(),
                    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                ])
                inp = tf(pil).unsqueeze(0).to(device)
                cam_map, logits = mdls[m]["cam"](inp)
                pred = logits.argmax(1).item()
                conf = F.softmax(logits, dim=1).max().item()
                cam_np = cam_map[0].cpu().numpy()
                cam_r = np.array(Image.fromarray(cam_np).resize((96,96), Image.BILINEAR))

                axes[row, mi].imshow(raw.permute(1,2,0).numpy())
                axes[row, mi].imshow(cam_r, cmap="jet", alpha=0.5)
                pred_name = class_names[pred] if pred < len(class_names) else f"cls{pred}"
                color = "green" if pred == ci else "red"
                axes[row, mi].set_title(f"{mlabels[m]}\n{pred_name[:15]} ({conf:.0%})", fontsize=7, color=color)
                axes[row, mi].axis("off")
            row += 1

    axes[0, 0].set_title("Original", fontsize=11, fontweight="bold")
    plt.suptitle("GradCAM: Baseline vs Vanilla KD vs AKR\n(WasteNet-Micro 21.6K, WaRP-C 28 classes)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "gradcam_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "gradcam_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved gradcam_comparison.png")


# ═══════════════════════════════════════════════════════════════
# 2. t-SNE
# ═══════════════════════════════════════════════════════════════
def extract_features(model, loader, device, arch_name):
    feats_list = []; labs_list = []; preds_list = []
    def hook_fn(module, input, output):
        feats_list.append(input[0].detach().cpu())
    if "mobilenet" in arch_name:
        handle = model.classifier[-1].register_forward_hook(hook_fn)
    else:
        handle = model.classifier.register_forward_hook(hook_fn)
    with torch.no_grad():
        for imgs, labs in loader:
            out = model(imgs.to(device))
            labs_list.append(labs); preds_list.append(out.argmax(1).cpu())
    handle.remove()
    return torch.cat(feats_list).numpy(), torch.cat(labs_list).numpy(), torch.cat(preds_list).numpy()


def generate_tsne(device, class_names):
    print("\n[2/3] t-SNE comparison...")
    arch = "wastenet_micro"
    methods = ["none", "vanilla", "akr"]
    mlabels = {"none": "Baseline (No KD)", "vanilla": "Vanilla KD", "akr": "AKR (Proposed)"}

    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    cmap = plt.colormaps.get_cmap('tab20').resampled(NUM_CLASSES)
    colors = [cmap(i) for i in range(NUM_CLASSES)]

    # Group classes for legend readability
    super_groups = {}
    for i, cn in enumerate(class_names):
        prefix = cn.split("-")[0]
        if prefix not in super_groups: super_groups[prefix] = []
        super_groups[prefix].append(i)

    for idx, m in enumerate(methods):
        model, sz = load_student(arch, m, device)
        tf = transforms.Compose([
            transforms.Resize((sz,sz)), transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        ds = ImageFolder(str(DATA_DIR / "test"), transform=tf)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
        feats, labs, preds = extract_features(model, loader, device, arch)
        f1 = f1_score(labs, preds, average="macro") * 100

        tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000,
                     learning_rate="auto", init="pca")
        emb = tsne.fit_transform(feats)

        # Plot by super-group with shared color
        group_colors = {"bottle": "#3498db", "glass": "#2ecc71", "cans": "#e74c3c",
                       "detergent": "#9b59b6", "canister": "#f39c12",
                       "juice": "#1abc9c", "milk": "#e67e22"}
        for prefix, indices in super_groups.items():
            gc = group_colors.get(prefix, "#95a5a6")
            for ci in indices:
                mask = labs == ci
                marker = "o" if "-full" not in class_names[ci] else "^"
                axes[idx].scatter(emb[mask,0], emb[mask,1], c=gc,
                                 marker=marker, s=12, alpha=0.6, edgecolors="none")

        # Mark misclassified
        wrong = preds != labs
        if wrong.any():
            axes[idx].scatter(emb[wrong,0], emb[wrong,1], c="none",
                             edgecolors="red", s=35, linewidths=1, marker="o", alpha=0.6)

        axes[idx].set_title(f"{mlabels[m]}\nF1={f1:.1f}% ({wrong.sum()} errors)",
                           fontsize=12, fontweight="bold")
        axes[idx].set_xticks([]); axes[idx].set_yticks([])

    # Legend for super-groups
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = []
    for prefix in sorted(super_groups):
        gc = group_colors.get(prefix, "#95a5a6")
        legend_handles.append(Patch(facecolor=gc, label=prefix))
    legend_handles.append(Line2D([0],[0], marker="o", color="w", markeredgecolor="red",
                                  markerfacecolor="none", markersize=8, label="misclassified"))
    axes[0].legend(handles=legend_handles, fontsize=8, loc="lower left", ncol=2, framealpha=0.8)

    plt.suptitle("t-SNE Feature Embeddings: Impact of Knowledge Distillation\n"
                 "(WasteNet-Micro 21.6K, WaRP-C 28 classes, red circles = misclassified)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tsne_comparison.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "tsne_comparison.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved tsne_comparison.png")


# ═══════════════════════════════════════════════════════════════
# 3. ROUTING VISUALIZATION
# ═══════════════════════════════════════════════════════════════
def generate_routing(device, class_names):
    print("\n[3/3] Routing visualization...")
    base_teacher = load_teacher("convnext", device)
    ta_teacher = load_teacher("efficientnet", device)
    akr_model, akr_sz = load_student("wastenet_micro", "akr", device)
    none_model, _ = load_student("wastenet_micro", "none", device)

    tf_t = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    tf_s = transforms.Compose([transforms.Resize((akr_sz,akr_sz)), transforms.ToTensor(),
                                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

    loader_t = DataLoader(ImageFolder(str(DATA_DIR/"test"), transform=tf_t), batch_size=64, shuffle=False, num_workers=2)
    loader_s = DataLoader(ImageFolder(str(DATA_DIR/"test"), transform=tf_s), batch_size=64, shuffle=False, num_workers=2)

    difficulties=[]; labels_all=[]; base_confs=[]
    with torch.no_grad():
        for imgs, labs in loader_t:
            imgs = imgs.to(device)
            bt = base_teacher(imgs)
            probs = F.softmax(bt, dim=1)
            top2 = probs.topk(2, dim=1).values
            d = 1.0 - (top2[:,0] - top2[:,1])
            difficulties.append(d.cpu()); labels_all.append(labs); base_confs.append(probs.max(1).values.cpu())
    difficulties = torch.cat(difficulties).numpy()
    labels_all = torch.cat(labels_all).numpy()
    base_confs = torch.cat(base_confs).numpy()

    akr_preds=[]; akr_confs=[]; none_preds=[]
    with torch.no_grad():
        for imgs, labs in loader_s:
            imgs = imgs.to(device)
            a_out = akr_model(imgs); n_out = none_model(imgs)
            akr_preds.append(a_out.argmax(1).cpu())
            akr_confs.append(F.softmax(a_out, dim=1).max(1).values.cpu())
            none_preds.append(n_out.argmax(1).cpu())
    akr_preds = torch.cat(akr_preds).numpy()
    akr_confs = torch.cat(akr_confs).numpy()
    none_preds = torch.cat(none_preds).numpy()

    easy_th = np.percentile(difficulties, 33)
    hard_th = np.percentile(difficulties, 67)
    routes = np.full(len(difficulties), 1)
    routes[difficulties <= easy_th] = 0
    routes[difficulties >= hard_th] = 2

    akr_ok = akr_preds == labels_all
    none_ok = none_preds == labels_all

    route_labels = {0:"Easy (CE only)", 1:"Standard (TA)", 2:"Hard (Base Teacher)"}
    route_colors = {0:"#2ecc71", 1:"#f39c12", 2:"#e74c3c"}

    fig = plt.figure(figsize=(22, 15))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # P1: Difficulty histogram
    ax1 = fig.add_subplot(gs[0,0])
    ax1.hist(difficulties, bins=50, color="#3498db", edgecolor="white", alpha=0.8, density=True)
    ax1.axvline(easy_th, color="#2ecc71", ls="--", lw=2, label=f"Easy < {easy_th:.3f}")
    ax1.axvline(hard_th, color="#e74c3c", ls="--", lw=2, label=f"Hard > {hard_th:.3f}")
    ax1.axvspan(0, easy_th, alpha=0.12, color="#2ecc71")
    ax1.axvspan(easy_th, hard_th, alpha=0.12, color="#f39c12")
    ax1.axvspan(hard_th, 1.0, alpha=0.12, color="#e74c3c")
    ax1.set_xlabel("Difficulty (d = 1 - margin)", fontsize=11)
    ax1.set_ylabel("Density")
    ax1.set_title("Sample Difficulty Distribution\nwith Routing Zones", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)

    # P2: Per-class routing (top 12 hardest + top 4 easiest)
    ax2 = fig.add_subplot(gs[0,1])
    class_routing = defaultdict(lambda: {0:0,1:0,2:0,"total":0,"diff":0})
    for i in range(len(labels_all)):
        cn = class_names[labels_all[i]]
        class_routing[cn][routes[i]] += 1
        class_routing[cn]["total"] += 1
    for cn in class_routing:
        ci = class_names.index(cn)
        class_routing[cn]["diff"] = np.mean(difficulties[labels_all==ci])

    sorted_cls = sorted(class_routing, key=lambda c: class_routing[c]["diff"], reverse=True)
    display_cls = sorted_cls[:10] + sorted_cls[-4:]  # top 10 hardest + 4 easiest

    easy_p = [100*class_routing[c][0]/class_routing[c]["total"] for c in display_cls]
    std_p = [100*class_routing[c][1]/class_routing[c]["total"] for c in display_cls]
    hard_p = [100*class_routing[c][2]/class_routing[c]["total"] for c in display_cls]
    y = range(len(display_cls))
    ax2.barh(y, easy_p, color="#2ecc71", label="Easy", edgecolor="white")
    ax2.barh(y, std_p, left=easy_p, color="#f39c12", label="Standard", edgecolor="white")
    left2 = [e+s for e,s in zip(easy_p, std_p)]
    ax2.barh(y, hard_p, left=left2, color="#e74c3c", label="Hard", edgecolor="white")
    ax2.set_yticks(y); ax2.set_yticklabels([f"{c[:18]}" for c in display_cls], fontsize=8)
    ax2.set_xlabel("Routing (%)"); ax2.invert_yaxis()
    ax2.set_title("Per-Class Routing (top10 hard + 4 easy)\n(sorted by difficulty)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8, loc="lower right")

    # P3: Accuracy per zone
    ax3 = fig.add_subplot(gs[0,2])
    zones = ["Easy", "Standard", "Hard"]
    akr_accs=[]; none_accs=[]; cnts=[]
    for z in [0,1,2]:
        mask = routes==z; n=mask.sum(); cnts.append(n)
        akr_accs.append(100*akr_ok[mask].mean() if n>0 else 0)
        none_accs.append(100*none_ok[mask].mean() if n>0 else 0)
    x = np.arange(3); w = 0.35
    ax3.bar(x-w/2, none_accs, w, label="Baseline", color="#95a5a6", edgecolor="white")
    ax3.bar(x+w/2, akr_accs, w, label="AKR", color="#e74c3c", edgecolor="white")
    for i in range(3):
        diff = akr_accs[i]-none_accs[i]
        ax3.annotate(f"{diff:+.1f}pp", xy=(x[i]+w/2, akr_accs[i]),
                    xytext=(0,5), textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color="green" if diff>0 else "red")
        ax3.text(x[i], 2, f"n={cnts[i]}", ha="center", fontsize=8, color="gray")
    ax3.set_xticks(x); ax3.set_xticklabels(zones)
    ax3.set_ylabel("Accuracy (%)"); ax3.set_ylim(0, 105)
    ax3.set_title("Accuracy per Routing Zone\nAKR vs Baseline", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=9)

    # P4: Scatter
    ax4 = fig.add_subplot(gs[1,0])
    for z in [0,1,2]:
        mask = routes==z
        ax4.scatter(difficulties[mask], akr_confs[mask], c=route_colors[z], s=10, alpha=0.4, label=route_labels[z])
    ax4.set_xlabel("Teacher Difficulty"); ax4.set_ylabel("AKR Student Confidence")
    ax4.set_title("Difficulty vs Student Confidence\n(color = routing)", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=8, markerscale=2)

    # P5: Fixes vs breaks
    ax5 = fig.add_subplot(gs[1,1])
    akr_fixes = akr_ok & ~none_ok
    akr_breaks = ~akr_ok & none_ok
    fix_r=[]; brk_r=[]
    for z in [0,1,2]:
        mask = routes==z; n=mask.sum()
        fix_r.append(100*akr_fixes[mask].sum()/n if n>0 else 0)
        brk_r.append(100*akr_breaks[mask].sum()/n if n>0 else 0)
    ax5.bar(x-w/2, fix_r, w, label="AKR fixes", color="#27ae60", edgecolor="white")
    ax5.bar(x+w/2, brk_r, w, label="AKR breaks", color="#c0392b", edgecolor="white")
    for i in range(3):
        net = fix_r[i]-brk_r[i]
        ax5.annotate(f"net {net:+.1f}%", xy=(x[i], max(fix_r[i],brk_r[i])),
                    xytext=(0,8), textcoords="offset points", ha="center", fontsize=9,
                    fontweight="bold", color="green" if net>0 else "red")
    ax5.set_xticks(x); ax5.set_xticklabels(zones)
    ax5.set_ylabel("% of samples")
    ax5.set_title("AKR Impact: Fixes vs Breaks\nper Routing Zone", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=9)

    # P6: Summary
    ax6 = fig.add_subplot(gs[1,2]); ax6.axis("off")
    akr_f1 = f1_score(labels_all, akr_preds, average="macro") * 100
    none_f1 = f1_score(labels_all, none_preds, average="macro") * 100
    txt = [
        f"{'='*44}",
        f"AKR ROUTING SUMMARY",
        f"WaRP-C 28 classes, WasteNet-Micro (21.6K)",
        f"{'='*44}", "",
        f"Test samples: {len(labels_all)}",
        f"  Easy  (CE only):     {(routes==0).sum():>4} ({100*(routes==0).mean():.1f}%)",
        f"  Standard (TA):       {(routes==1).sum():>4} ({100*(routes==1).mean():.1f}%)",
        f"  Hard (Base Teacher): {(routes==2).sum():>4} ({100*(routes==2).mean():.1f}%)",
        "", f"{'─'*44}",
        f"F1 macro:  Baseline={none_f1:.1f}%  AKR={akr_f1:.1f}%",
        f"Accuracy:  Baseline={100*none_ok.mean():.1f}%  AKR={100*akr_ok.mean():.1f}%",
        f"AKR F1 Delta: {akr_f1-none_f1:+.1f}pp",
        "",
        f"Fixes  (wrong->correct): {akr_fixes.sum():>4}",
        f"Breaks (correct->wrong): {akr_breaks.sum():>4}",
        f"Net: {akr_fixes.sum()-akr_breaks.sum():+d} samples",
        "",
        f"Thresholds: easy<{easy_th:.4f} hard>{hard_th:.4f}",
    ]
    ax6.text(0.05, 0.95, "\n".join(txt), transform=ax6.transAxes, fontsize=10,
             fontfamily="monospace", va="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.suptitle("Adaptive Knowledge Routing (AKR) — Routing Decision Analysis\n"
                 "WaRP-C 28 classes, WasteNet-Micro (21.6K params)", fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(OUTPUT_DIR / "routing_visualization.png", dpi=200, bbox_inches="tight")
    plt.savefig(OUTPUT_DIR / "routing_visualization.pdf", bbox_inches="tight")
    plt.close()
    print("  Saved routing_visualization.png")


def main():
    set_seed()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {device}")

    # Get class names from dataset
    ds = ImageFolder(str(DATA_DIR / "test"))
    class_names = ds.classes
    print(f"Classes ({len(class_names)}): {class_names[:5]}...")

    generate_gradcam(device, class_names)
    generate_tsne(device, class_names)
    generate_routing(device, class_names)
    print(f"\nAll done! Output: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
