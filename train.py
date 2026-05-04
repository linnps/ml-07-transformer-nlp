"""
Train a small Transformer encoder *from scratch* in PyTorch on the
synthetic sentiment dataset. No HuggingFace dependency — the model
is in this file, top to bottom.

Architecture: token embedding + positional embedding → 2 self-attention
encoder layers → mean-pool over tokens → linear classifier head.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader, TensorDataset

from generate_data import DataConfig, generate, vocab_from

# ---------------------------------------------------------------- style ----
COLOR_BG = "#FFFFFF"
COLOR_GRID = "#E5E5E5"
COLOR_TEXT = "#333333"
COLOR_BLUE = "#3B6EA8"
COLOR_RED = "#C04040"
COLOR_GRAY = "#7A7A7A"
COLOR_LIGHT_GRAY = "#CCCCCC"

mpl.rcParams.update({
    "figure.facecolor": COLOR_BG,
    "axes.facecolor": COLOR_BG,
    "axes.edgecolor": COLOR_LIGHT_GRAY,
    "axes.labelcolor": COLOR_TEXT,
    "axes.titlecolor": COLOR_TEXT,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": COLOR_TEXT,
    "ytick.color": COLOR_TEXT,
    "grid.color": COLOR_GRID,
    "grid.linewidth": 0.6,
    "axes.grid": True,
    "legend.frameon": False,
    "font.family": "sans-serif",
    "font.size": 11,
})

CMAP_BLUE = LinearSegmentedColormap.from_list("blue_only", ["#FFFFFF", COLOR_BLUE])


# --------------------------------------------------------------- model ----
class SmallTransformer(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 64, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 32, n_classes: int = 2,
                 pad_idx: int = 0) -> None:
        super().__init__()
        self.pad_idx = pad_idx
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, n_classes)
        self.max_len = max_len

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        B, T = ids.shape
        positions = torch.arange(T, device=ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(ids) + self.pos_emb(positions)
        pad_mask = (ids == self.pad_idx)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        # Mean-pool over non-pad tokens.
        mask = (~pad_mask).float().unsqueeze(-1)
        h_pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        return self.head(h_pooled)


# ---------------------------------------------------------- tokenization --
def tokenize(text: str, vocab: dict[str, int], max_len: int) -> list[int]:
    ids = [vocab.get(tok, vocab["<unk>"]) for tok in text.split()]
    ids = ids[:max_len]
    ids += [vocab["<pad>"]] * (max_len - len(ids))
    return ids


# -------------------------------------------------------------- training --
def train_model(epochs: int = 8, batch_size: int = 32, lr: float = 1e-3,
                seed: int = 42) -> dict:
    torch.manual_seed(seed)
    cfg = DataConfig()
    train, test = generate(cfg)

    vocab = vocab_from(train)
    max_len = max(len(t.split()) for t, _, _ in train + test) + 2  # safety margin

    def encode(rows):
        ids = torch.tensor([tokenize(t, vocab, max_len) for t, _, _ in rows], dtype=torch.long)
        labels = torch.tensor([lab for _, lab, _ in rows], dtype=torch.long)
        return ids, labels

    Xtr, ytr = encode(train)
    Xte, yte = encode(test)

    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)

    model = SmallTransformer(vocab_size=len(vocab), max_len=max_len, n_classes=2)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "test_acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        loss_sum, corr, n = 0.0, 0, 0
        for xb, yb in loader:
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss) * yb.size(0)
            corr += int((logits.argmax(1) == yb).sum())
            n += yb.size(0)
        tr_loss, tr_acc = loss_sum / n, corr / n

        model.eval()
        with torch.no_grad():
            te_logits = model(Xte)
            te_acc = float((te_logits.argmax(1) == yte).float().mean())

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_acc"].append(te_acc)
        print(f"epoch {ep:2d}  tr_loss {tr_loss:.4f}  tr_acc {tr_acc:.3f}  te_acc {te_acc:.3f}")

    return {
        "model": model, "vocab": vocab, "max_len": max_len,
        "test": test, "Xte": Xte, "yte": yte, "history": history,
    }


# ---------------------------------------------------------------- figures --
def fig_samples_table(out_path: Path, train: list) -> None:
    """Render some sample examples as a small text table."""
    fig, ax = plt.subplots(figsize=(11, 5.0), constrained_layout=True)
    ax.axis("off")

    rows = []
    rows.append(["Label", "Structure", "Text"])
    seen = {("plain", 0): 0, ("plain", 1): 0,
            ("negated", 0): 0, ("negated", 1): 0}
    for text, lab, struct in train:
        key = (struct, lab)
        if key in seen and seen[key] < 2:
            seen[key] += 1
            rows.append([
                "positive" if lab == 1 else "negative",
                struct,
                f"{text[:80]}{'…' if len(text) > 80 else ''}",
            ])
        if all(v >= 2 for v in seen.values()):
            break

    table = ax.table(cellText=rows, loc="center",
                     colWidths=[0.10, 0.10, 0.80], cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    # Header row styling.
    for c in range(3):
        cell = table[0, c]
        cell.set_facecolor("#E5EAF2")
        cell.set_text_props(weight="bold", color=COLOR_TEXT)
    # Color label cells.
    for r in range(1, len(rows)):
        col = COLOR_BLUE if rows[r][0] == "positive" else COLOR_RED
        table[r, 0].set_text_props(color=col, weight="bold")
    fig.suptitle("Synthetic sentiment examples — plain and negated templates",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_curves(history: dict, out_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), constrained_layout=True)
    axes[0].plot(epochs, history["train_loss"], color=COLOR_BLUE, marker="o", linewidth=1.6)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train cross-entropy")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_acc"], color=COLOR_BLUE, marker="o",
                 linewidth=1.6, label="train")
    axes[1].plot(epochs, history["test_acc"],  color=COLOR_RED,  marker="o",
                 linewidth=1.6, label="test")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.4, 1.02); axes[1].set_title("Accuracy"); axes[1].legend()
    fig.suptitle("Training dynamics", fontsize=14, fontweight="bold", y=1.05)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_per_struct_acc(out: dict, out_path: Path) -> None:
    """Accuracy on plain vs negated test slices."""
    test = out["test"]
    Xte, yte = out["Xte"], out["yte"]
    model = out["model"]
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1).numpy()
    correct = preds == yte.numpy()
    structs = np.array([s for _, _, s in test])

    accs = {s: float(correct[structs == s].mean()) for s in ["plain", "negated"]}

    fig, ax = plt.subplots(figsize=(7, 3.8), constrained_layout=True)
    bars = ax.bar(list(accs.keys()), list(accs.values()),
                  color=[COLOR_BLUE, COLOR_RED],
                  edgecolor=COLOR_LIGHT_GRAY, linewidth=0.8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Accuracy")
    ax.set_title("Test accuracy on plain vs negated examples")
    for bar, v in zip(bars, accs.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.3f}", ha="center", va="bottom", fontsize=11,
                color=COLOR_TEXT, weight="bold")
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return accs


def fig_confusion(out: dict, out_path: Path) -> np.ndarray:
    Xte, yte = out["Xte"], out["yte"]
    model = out["model"]
    model.eval()
    with torch.no_grad():
        preds = model(Xte).argmax(1).numpy()
    yte_np = yte.numpy()
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(yte_np, preds):
        cm[t, p] += 1
    norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(5.4, 4.5), constrained_layout=True)
    ax.imshow(norm, cmap=CMAP_BLUE, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            color = "white" if norm[i, j] > 0.4 else COLOR_TEXT
            ax.text(j, i, f"{cm[i, j]}\n({norm[i, j]:.2%})",
                    ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["pred neg", "pred pos"])
    ax.set_yticklabels(["true neg", "true pos"])
    ax.set_title("Confusion matrix on the test set")
    ax.grid(False)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return cm


def fig_attention_demo(out: dict, out_path: Path) -> None:
    """
    Visualize the attention pattern from layer-0 head-0 on a few negated examples.
    We hook into the encoder layer to grab attention weights.
    """
    model = out["model"]
    vocab = out["vocab"]
    max_len = out["max_len"]
    inv_vocab = {i: w for w, i in vocab.items()}
    test = out["test"]

    # Pick 3 negated examples (one positive, one negative, one of either).
    examples = [(t, l) for t, l, s in test if s == "negated"][:3]

    # Manual attention extraction by replacing the attn module's forward.
    layer = model.encoder.layers[0]
    captured = {}

    def hook(module, inputs, output):
        # Re-run the attention module with need_weights=True to capture maps.
        x = inputs[0]
        attn_mask = inputs[1] if len(inputs) > 1 else None
        kpm = inputs[2] if len(inputs) > 2 else None
        # Recreate same attn call, but ask for weights.
        attn_out, attn_w = module.self_attn(
            x, x, x, attn_mask=attn_mask,
            key_padding_mask=kpm,
            need_weights=True, average_attn_weights=False,
        )
        captured["weights"] = attn_w  # (B, n_heads, T, T)

    handle = layer.register_forward_pre_hook(lambda mod, inp: None)
    handle.remove()  # placeholder — actual capture is via layer.self_attn directly below

    fig, axes = plt.subplots(1, len(examples), figsize=(4.5 * len(examples), 4.2),
                             constrained_layout=True)
    if len(examples) == 1:
        axes = [axes]

    model.eval()
    for ax, (text, lab) in zip(axes, examples):
        ids = torch.tensor([
            [vocab.get(tok, vocab["<unk>"]) for tok in text.split()][:max_len]
            + [vocab["<pad>"]] * (max_len - len(text.split()))
        ], dtype=torch.long)
        with torch.no_grad():
            B, T = ids.shape
            positions = torch.arange(T).unsqueeze(0).expand(B, T)
            x = model.tok_emb(ids) + model.pos_emb(positions)
            pad_mask = (ids == model.pad_idx)
            _, attn = model.encoder.layers[0].self_attn(
                x, x, x, key_padding_mask=pad_mask,
                need_weights=True, average_attn_weights=True,
            )
        # `attn`: (B, T, T) when average_attn_weights=True
        attn = attn[0].numpy()

        toks = [inv_vocab[int(i)] for i in ids[0]]
        n_real = sum(1 for t in toks if t != "<pad>")
        attn = attn[:n_real, :n_real]
        toks = toks[:n_real]

        im = ax.imshow(attn, cmap=CMAP_BLUE, vmin=0, vmax=attn.max())
        ax.set_xticks(range(len(toks)))
        ax.set_yticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(toks, fontsize=8)
        ax.set_title(f"{'positive' if lab == 1 else 'negative'}",
                     color=COLOR_BLUE if lab == 1 else COLOR_RED)
        ax.grid(False)

    fig.suptitle("Layer-0 attention — averaged over heads (negated examples)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------- main ----
def main() -> None:
    out = train_model(epochs=8)

    cfg = DataConfig()
    train, _ = generate(cfg)

    Path("results").mkdir(exist_ok=True)
    cm = None
    accs_by_struct = None

    assets = Path("assets"); assets.mkdir(exist_ok=True)
    fig_samples_table(assets / "01_samples.png", train)
    fig_curves(out["history"], assets / "02_curves.png")
    accs_by_struct = fig_per_struct_acc(out, assets / "03_struct_acc.png")
    cm = fig_confusion(out, assets / "04_confusion.png")
    fig_attention_demo(out, assets / "05_attention.png")

    summary = {
        "epochs": 8,
        "vocab_size": len(out["vocab"]),
        "max_len": out["max_len"],
        "final_train_acc": float(out["history"]["train_acc"][-1]),
        "final_test_acc": float(out["history"]["test_acc"][-1]),
        "history": out["history"],
        "test_accuracy_by_structure": accs_by_struct,
        "confusion_matrix": cm.tolist(),
    }
    with open("results/metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nFinal test acc: {summary['final_test_acc']:.3f}")
    print(f"Plain  test acc: {accs_by_struct['plain']:.3f}")
    print(f"Negated test acc: {accs_by_struct['negated']:.3f}")
    print(f"Figures saved to: {assets.resolve()}")


if __name__ == "__main__":
    main()
