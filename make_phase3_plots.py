"""
Generate Phase 3 figures from the captured A40 run data.

Run with:  python make_phase3_plots.py

Writes:
    figures/phase3_loss.png        — slice + LoRA training-loss curves
    figures/phase3_expert_load.png — per-layer, per-mode expert activation %s

These numbers are hard-coded from the actual A40 run logs
(phase3_train_slice.log, phase3_train_lora.log, phase3_eval_slice.log,
phase3_eval_lora.log). Re-run the training scripts to refresh the data
and update the constants at the top of this file.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------- training loss data ---------------------------------------
EPOCHS = [1, 2]
SLICE_LOSS = [10.886, 8.411]
LORA_LOSS = [2.374, 2.359]


# ---------------- expert load data (4 layers × 4 experts) ------------------
# Percentages from the train_moe.py "Expert load after training" lines.
SLICE_LOAD_AFTER = np.array([
    [17, 22, 16, 45],   # layer 0
    [48, 26, 13, 14],   # layer 1
    [40, 46,  6,  7],   # layer 2
    [19, 50, 23,  7],   # layer 3
])
LORA_LOAD_AFTER = np.array([
    [25, 21, 34, 21],   # layer 0
    [30, 11, 21, 38],   # layer 1
    [18, 33, 20, 29],   # layer 2
    [31, 25, 33, 11],   # layer 3
])


def make_loss_plot(out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(EPOCHS, SLICE_LOSS, "o-", color="#d6604d",
            label=f"Slice (router only, 131K params)")
    ax.plot(EPOCHS, LORA_LOSS, "s-", color="#4393c3",
            label=f"LoRA (router + adapters, 2.23M params)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss (cross-entropy, nats)")
    ax.set_title("Phase 3: training loss for both MoE init modes")
    ax.set_xticks(EPOCHS)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="center right")
    # annotate endpoint values
    for x, y in zip(EPOCHS, SLICE_LOSS):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(8, -2), fontsize=9, color="#d6604d")
    for x, y in zip(EPOCHS, LORA_LOSS):
        ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                    xytext=(8, -10), fontsize=9, color="#4393c3")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def make_expert_load_plot(out_path: str) -> None:
    """Grouped bar chart: 4 layers × 4 experts, slice vs lora side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)

    expert_labels = [f"E{i}" for i in range(4)]
    layers = ["Layer 0", "Layer 1", "Layer 2", "Layer 3"]
    bar_w = 0.18
    x = np.arange(len(layers))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

    for ax, data, title in [
        (axes[0], SLICE_LOAD_AFTER, "Slice mode (after fine-tuning)"),
        (axes[1], LORA_LOAD_AFTER, "LoRA mode (after fine-tuning)"),
    ]:
        for i in range(4):
            ax.bar(x + (i - 1.5) * bar_w, data[:, i], bar_w,
                   color=colors[i], label=expert_labels[i])
        ax.axhline(25.0, color="grey", ls="--", lw=0.8, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.set_title(title)
        ax.set_ylabel("Routings (%)")
        ax.set_ylim(0, 60)
        ax.grid(True, axis="y", alpha=0.25)

    # one shared legend on the right
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right",
               bbox_to_anchor=(1.04, 0.5), frameon=False, title="Expert")
    fig.suptitle(
        "Phase 3: per-layer expert activation distribution "
        "(dashed line = uniform 25 %)", y=1.02
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs("figures", exist_ok=True)
    make_loss_plot("figures/phase3_loss.png")
    make_expert_load_plot("figures/phase3_expert_load.png")
    print("Wrote figures/phase3_loss.png")
    print("Wrote figures/phase3_expert_load.png")


if __name__ == "__main__":
    main()
