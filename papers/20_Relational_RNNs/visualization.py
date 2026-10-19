"""
Day 20: Relational RNNs - Visualization
=======================================

- Attention heatmaps: where each memory slot looks, at every time step. With the paper's Eq. 2 the
  matrix is N x (N+1): N memory slots ask (rows), N slots plus the input row answer (columns). The
  last column is how much a slot reads the new input - the column the paper's Fig. 3 calls "input".
- Accuracy curves: RMC vs LSTM on the Nth Farthest task.

Run `train_minimal.py --visualize --save_plot` to generate both.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch

SERIES_COLORS = ['#2a78d6', '#eb6834', '#1baf7a']   # fixed categorical order: RMC, LSTM, (third model)
INK, MUTED = '#0b0b0b', '#52514e'


# [PATCH 2026-10-18] N x (N+1) attention with a labelled input column; optional reference-step highlight
def plot_attention_heatmap(attention_weights, save_path="attention_map.png", highlight_step=None, title=None):
    """Plot the RMC's attention over time, averaged over heads.

    Args:
        attention_weights: (seq_len, num_heads, mem_slots, mem_slots + 1) - or (..., mem_slots) for a
            square matrix. One sequence, e.g. RelationalRNN(..., return_attention=True)[1][0].
        save_path: where to save the figure.
        highlight_step: optional time step to mark (e.g. when the reference vector m arrived).
        title: optional figure title.
    """
    if isinstance(attention_weights, torch.Tensor):
        attention_weights = attention_weights.detach().cpu().numpy()
    seq_len, num_heads, slots, cols = attention_weights.shape
    avg_attention = np.mean(attention_weights, axis=1)      # mean over the heads axis -> (seq_len, slots, cols)
    plot_steps = min(seq_len, 10)
    col_labels = [str(j) for j in range(slots)] + (['in'] if cols == slots + 1 else [])

    fig, axes = plt.subplots(1, plot_steps, figsize=(1.9 * plot_steps + 0.6, 2.9), squeeze=False)
    axes = axes[0]
    for t in range(plot_steps):
        ax = axes[t]
        im = ax.imshow(avg_attention[t], cmap='Blues', vmin=0, vmax=1)   # single-hue sequential: weight in [0, 1]
        ax.set_xticks(range(cols))
        ax.set_xticklabels(col_labels, fontsize=7, color=MUTED)
        ax.set_yticks(range(slots))
        ax.set_yticklabels([str(i) for i in range(slots)] if t == 0 else [], fontsize=7, color=MUTED)
        mark = t == highlight_step
        ax.set_title(f"t={t}" + (" (m)" if mark else ""), fontsize=9,
                     color='#e34948' if mark else INK, fontweight='bold' if mark else 'normal')
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("slot (query)", fontsize=8, color=MUTED)
    fig.colorbar(im, ax=list(axes), orientation='vertical', fraction=0.02, pad=0.02).set_label(
        "attention weight", fontsize=8, color=MUTED)
    fig.suptitle(title or "RMC attention per step (mean over heads); columns: slots, then the input row",
                 fontsize=10, color=INK)
    fig.text(0.5, 0.02, "attended row (key): memory slots 0..N-1, 'in' = the projected input x~",
             ha='center', fontsize=8, color=MUTED)
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved attention heatmap to {save_path}")


# [PATCH 2026-10-18] accuracy curves for any number of models (was: two MSE loss lists)
def plot_comparison(histories, save_path="comparison.png", baselines=None, marker_step=None):
    """Accuracy curves on the Nth Farthest task: relational questions (left), all questions (right).

    Args:
        histories: {model name: {'step': [...], 'acc': [...], 'acc_relational': [...]}} as returned
            by train_minimal.train_model(...)['history'].
        save_path: where to save the figure.
        baselines: {'acc_relational' or 'acc': {label: accuracy}}, dashed reference lines per panel.
        marker_step: optional step to mark with a vertical line (e.g. the end of the curriculum).
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    panels = [('acc_relational', 'Relational questions (n < K-1)'), ('acc', 'All questions')]
    for ax, (key, name) in zip(axes, panels):
        for k, (color, (model, h)) in enumerate(zip(SERIES_COLORS, histories.items())):
            ax.plot(h['step'], h[key], color=color, linewidth=2, label=model)
            ax.annotate(model, (h['step'][-1], h[key][-1]), xytext=(4, 6 - 12 * k), textcoords='offset points',
                        fontsize=8, color=INK, va='center')   # staggered so labels of equal curves don't collide
        ax.set_title(name, fontsize=10, color=INK)
        ax.set_xlabel("training step", color=MUTED)
        ax.grid(True, alpha=0.25)
        ax.set_ylim(0, 1)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        if marker_step:
            ax.axvline(marker_step, color=MUTED, linestyle=':', linewidth=1)
            ax.annotate("all n from here", (marker_step, 0.02), xytext=(3, 0), textcoords='offset points',
                        fontsize=8, color=MUTED)
        for label, value in (baselines or {}).get(key, {}).items():
            ax.axhline(value, color=MUTED, linestyle='--', linewidth=1)
            ax.annotate(label, (0, value), xytext=(2, 3), textcoords='offset points', fontsize=8, color=MUTED)
    axes[0].set_ylabel("eval accuracy", color=MUTED)
    axes[1].legend(frameon=False, fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close(fig)
    print(f"Saved comparison plot to {save_path}")
