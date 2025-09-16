#!/usr/bin/env python3
# evaluation.py
#
# Read true labels + predicted probabilities, then:
#  - Print overall ROC/PR AUCs
#  - Sweep thresholds and print TP/FP/TN/FN + metrics
#  - Save confusion matrices (counts + normalized), ROC, PR plots
#  - Save classification reports per threshold
#  - Save threshold_sweep.csv
#
# Example:
#   python3 evaluation.py \
#       --csv results/predictions.csv \
#       --y_true_col churn_true \
#       --y_proba_col churn_proba \
#       --outdir results/plots \
#       --thresholds 0.20 0.30 0.40 0.50

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, precision_recall_curve, auc, precision_score,
    recall_score, f1_score, accuracy_score
)

# -----------------------------
# CLI
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="CSV containing true labels and predicted probabilities.")
    p.add_argument("--y_true_col", required=True, help="Column name: true labels (0/1).")
    p.add_argument("--y_proba_col", required=True, help="Column name: predicted probability for positive class (1).")
    p.add_argument("--outdir", default="results/plots", help="Directory to save plots and reports.")
    p.add_argument("--thresholds", type=float, nargs="*", default=[0.20, 0.30, 0.40, 0.50],
                   help="Decision thresholds to evaluate.")
    p.add_argument("--pos_label", type=int, default=1, help="Positive label value (default 1).")
    p.add_argument("--neg_label", type=int, default=0, help="Negative label value (default 0).")
    p.add_argument("--title_prefix", default="Churn (positive=1)", help="Title prefix for plots.")
    return p.parse_args()

# -----------------------------
# Plot helpers
# -----------------------------
def plot_confusion_matrix(cm, class_names, title, savepath, normalize=False):
    """Confusion matrix (counts or row-normalized %) with annotations."""
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    data = cm.astype(float)
    if normalize:
        row_sums = data.sum(axis=1, keepdims=True)
        data = np.divide(data, row_sums, out=np.zeros_like(data), where=row_sums != 0)
    im = ax.imshow(data, interpolation="nearest")

    ax.set_title(title, pad=12)
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    # Annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if normalize:
                txt = f"{cm[i, j]:.0f}\n({data[i, j] * 100:.1f}%)"
            else:
                txt = f"{cm[i, j]:.0f}"
            ax.text(j, i, txt, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(savepath, dpi=200)
    plt.close(fig)

def plot_roc_pr(y_true, y_proba, outdir: Path, title_prefix: str):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(fpr, tpr, label=f"ROC AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{title_prefix} — ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(outdir / "roc_curve.png", dpi=200)
    plt.close(fig)

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    ax.plot(recall, precision, label=f"PR AUC = {pr_auc:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{title_prefix} — Precision-Recall Curve")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(outdir / "pr_curve.png", dpi=200)
    plt.close(fig)

    return roc_auc, pr_auc

# -----------------------------
# Metrics at a threshold
# -----------------------------
def metrics_for_threshold(y_true, y_proba, thr, pos_label=1):
    y_pred = (y_proba >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0, pos_label=pos_label)
    rec = recall_score(y_true, y_pred, pos_label=pos_label)
    f1 = f1_score(y_true, y_pred, pos_label=pos_label)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "threshold": thr,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "accuracy": acc, "precision": prec, "recall": rec, "specificity": spec, "f1": f1
    }, cm

# -----------------------------
# Main
# -----------------------------
def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(args.csv)
    if args.y_true_col not in df.columns or args.y_proba_col not in df.columns:
        raise ValueError(
            f"CSV must contain '{args.y_true_col}' and '{args.y_proba_col}'. "
            f"Found: {list(df.columns)}"
        )

    y_true = df[args.y_true_col].astype(int).to_numpy()
    y_proba = df[args.y_proba_col].astype(float).to_numpy()

    # Curves + AUCs
    roc_auc_curve, pr_auc_curve = plot_roc_pr(y_true, y_proba, outdir, args.title_prefix)

    # Console header
    print("\n================= EVALUATION SUMMARY =================")
    print(f"Inputs: csv='{args.csv}', y_true='{args.y_true_col}', y_proba='{args.y_proba_col}'")
    print(f"Saved plots to: {outdir.resolve()}")
    print(f"Overall ROC AUC: {roc_auc_curve:.4f}")
    print(f"Overall PR  AUC: {pr_auc_curve:.4f}")
    print("======================================================\n")

    # Threshold sweep
    class_names = [f"Non-Churn ({args.neg_label})", f"Churn ({args.pos_label})"]
    rows = []
    for thr in args.thresholds:
        row, cm = metrics_for_threshold(y_true, y_proba, thr, pos_label=args.pos_label)
        rows.append(row)

        # Print to console (counts + normalized)
        tn, fp, fn, tp = cm.ravel()
        total = cm.sum()
        print(f"--- Threshold = {thr:.2f} ---")
        print(f"TP={tp}  FP={fp}  TN={tn}  FN={fn}  (Total={total})")
        print(f"Accuracy={row['accuracy']:.4f}  Precision={row['precision']:.4f}  "
              f"Recall={row['recall']:.4f}  Specificity={row['specificity']:.4f}  F1={row['f1']:.4f}")
        print()

        # Save plots
        title = f"{args.title_prefix} — Confusion Matrix @ threshold={thr:.2f}"
        plot_confusion_matrix(cm, class_names, title, outdir / f"cm_counts_thr_{thr:.2f}.png", normalize=False)
        plot_confusion_matrix(cm, class_names, title + " (normalized)", outdir / f"cm_norm_thr_{thr:.2f}.png", normalize=True)

        # Save a per-threshold classification report
        y_pred = (y_proba >= thr).astype(int)
        report = classification_report(y_true, y_pred, digits=4, target_names=["Non-Churn", "Churn"])
        with open(outdir / f"classification_report_thr_{thr:.2f}.txt", "w") as f:
            f.write(report)

    # Save sweep CSV
    sweep = pd.DataFrame(rows).sort_values("threshold")
    sweep_path = outdir / "threshold_sweep.csv"
    sweep.to_csv(sweep_path, index=False)

    # Quick peek
    print("=== threshold_sweep.csv (head) ===")
    print(sweep.head().to_string(index=False))
    print(f"\n[OK] Wrote: {sweep_path.resolve()}")
    print(f"[OK] Plots saved under: {outdir.resolve()}\n")

if __name__ == "__main__":
    main()
