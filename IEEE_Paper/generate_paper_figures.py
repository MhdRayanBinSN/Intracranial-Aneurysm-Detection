"""
Generate all evaluation figures for the IEEE paper.
Uses the 15-series evaluation from evaluation_output/evaluation_results.json
No model loading or re-inference needed.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay
)
from pathlib import Path

# ── Load real data from evaluation_results.json ───────────────────────────
data_path = Path(__file__).parent.parent / \
    "Code/Pretrained detection/evaluation_output/evaluation_results.json"

with open(data_path) as f:
    data = json.load(f)

series = data["per_series_results"]

# Build arrays
y_true  = np.array([1 if s["is_positive"] else 0 for s in series])
y_prob  = np.array([s["aneurysm_probability"] for s in series])
y_pred  = np.array([1 if s["predicted_positive"] else 0 for s in series])

# Series-level raw metrics
TP = int(data["series_metrics"]["true_positives"])
TN = int(data["series_metrics"]["true_negatives"])
FP = int(data["series_metrics"]["false_positives"])
FN = int(data["series_metrics"]["false_negatives"])
ACC  = data["series_metrics"]["accuracy"]
PREC = data["series_metrics"]["precision"]
REC  = data["series_metrics"]["recall"]
F1   = data["series_metrics"]["f1_score"]
SPEC = TN / (TN + FP)

# Location-level metrics
LOC_PREC = data["location_metrics"]["precision"]
LOC_REC  = data["location_metrics"]["recall"]
LOC_F1   = data["location_metrics"]["f1_score"]

print("=" * 55)
print("  nnU-Net Evaluation — 15-Series Test Set")
print("=" * 55)
print(f"  TP={TP}  TN={TN}  FP={FP}  FN={FN}")
print(f"  Accuracy        : {ACC*100:.1f}%")
print(f"  Sensitivity     : {REC*100:.1f}%")
print(f"  Specificity     : {SPEC*100:.1f}%")
print(f"  Precision       : {PREC*100:.1f}%")
print(f"  F1 Score        : {F1:.3f}")
print(f"  Location Prec   : {LOC_PREC*100:.1f}%")
print(f"  Location Recall : {LOC_REC*100:.1f}%")
print(f"  Location F1     : {LOC_F1:.3f}")

# ── ROC Curve ──────────────────────────────────────────────────────────────
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)
print(f"  ROC-AUC         : {roc_auc:.3f}")

# Precision-Recall Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
avg_prec = average_precision_score(y_true, y_prob)

# ── Figure 1: 2×2 metrics panel ────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle(
    "nnU-Net Blob Regression Model — Evaluation Results\n(15-Series Test Set, RSNA 2025 Dataset)",
    fontsize=12, fontweight='bold', y=0.98
)
plt.rcParams['font.family'] = 'DejaVu Sans'

# ── Panel A: Confusion Matrix ──────────────────────────────────────────────
ax = axes[0, 0]
cm = np.array([[TN, FP], [FN, TP]])
im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=9)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted\nNegative', 'Predicted\nPositive'], fontsize=9)
ax.set_yticklabels(['Actual\nNegative', 'Actual\nPositive'], fontsize=9)
for i in range(2):
    for j in range(2):
        val = cm[i, j]
        label = ['TN','FP','FN','TP'][i*2+j]
        ax.text(j, i, f'{val}\n({label})', ha='center', va='center',
                fontsize=13, fontweight='bold',
                color='white' if val > 5 else 'black')
ax.set_title('(a) Confusion Matrix', fontsize=10, fontweight='bold')
ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')

# ── Panel B: ROC Curve ────────────────────────────────────────────────────
ax = axes[0, 1]
ax.plot(fpr, tpr, color='#2563EB', lw=2.5,
        label=f'nnU-Net (AUC = {roc_auc:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
ax.fill_between(fpr, tpr, alpha=0.08, color='#2563EB')
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=9)
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=9)
ax.set_title('(b) ROC Curve', fontsize=10, fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.grid(True, alpha=0.3)
# mark operating point (threshold=0.5)
op_fpr = FP / (FP + TN)
op_tpr = TP / (TP + FN)
ax.plot(op_fpr, op_tpr, 'ro', ms=10, label='τ=0.50', zorder=5)
ax.annotate(f'  τ=0.50\n  ({op_tpr:.2f}, {1-op_fpr:.2f})',
            xy=(op_fpr, op_tpr), fontsize=8, color='red')

# ── Panel C: Metrics Bar Chart ────────────────────────────────────────────
ax = axes[1, 0]
metrics = {
    'Accuracy':    ACC,
    'Sensitivity\n(Recall)': REC,
    'Specificity': SPEC,
    'Precision':   PREC,
    'F1 Score':    F1,
    'ROC-AUC':     roc_auc,
}
colors = ['#1E40AF','#059669','#7C3AED','#D97706','#DC2626','#0891B2']
bars = ax.barh(list(metrics.keys()), [v*100 for v in metrics.values()],
               color=colors, edgecolor='white', height=0.55)
ax.set_xlim(0, 115)
ax.set_xlabel('Score (%)', fontsize=9)
ax.set_title('(c) Series-Level Performance Metrics', fontsize=10, fontweight='bold')
ax.axvline(x=50, color='gray', lw=0.8, linestyle='--', alpha=0.4)
for bar, val in zip(bars, metrics.values()):
    ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
            f'{val*100:.1f}%', va='center', fontsize=9, fontweight='bold')
ax.grid(axis='x', alpha=0.2)

# ── Panel D: Per-Location Recall ──────────────────────────────────────────
ax = axes[1, 1]
loc_data = {
    'Right MCA':            {'gt': 2, 'correct': 2},
    'Anterior Comm. Artery':{'gt': 1, 'correct': 1},
    'Left MCA':             {'gt': 1, 'correct': 1},
    'Right Supraclinoid ICA':{'gt':1, 'correct': 1},
    'Other Post. Circ.':    {'gt': 1, 'correct': 0},
    'R. Infraclinoid ICA':  {'gt': 1, 'correct': 0},
    'Right ACA':            {'gt': 1, 'correct': 0},
}
names   = list(loc_data.keys())
recalls = [v['correct']/v['gt']*100 for v in loc_data.values()]
colors_loc = ['#059669' if r == 100 else '#DC2626' for r in recalls]

bars2 = ax.barh(names, recalls, color=colors_loc, edgecolor='white', height=0.55)
ax.set_xlim(0, 130)
ax.set_xlabel('Recall (%)', fontsize=9)
ax.set_title('(d) Per-Location Detection Recall', fontsize=10, fontweight='bold')
ax.axvline(x=100, color='gray', lw=1, linestyle='--', alpha=0.4)
for bar, val in zip(bars2, recalls):
    ax.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
            f'{val:.0f}%', va='center', fontsize=9, fontweight='bold')

tp_patch = mpatches.Patch(color='#059669', label='Perfect Detection (100%)')
fn_patch = mpatches.Patch(color='#DC2626', label='Missed (0%)')
ax.legend(handles=[tp_patch, fn_patch], loc='lower right', fontsize=8)
ax.grid(axis='x', alpha=0.2)

plt.tight_layout(rect=[0, 0, 1, 0.96])
out1 = Path(__file__).parent / "figures" / "evaluation_panel.png"
out1.parent.mkdir(exist_ok=True)
plt.savefig(out1, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"\n  ✅ Saved: {out1}")

# ── Figure 2: Precision-Recall Curve ──────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(5, 4))
ax2.plot(recall_curve, precision_curve, color='#7C3AED', lw=2.5,
         label=f'nnU-Net (AP = {avg_prec:.3f})')
ax2.fill_between(recall_curve, precision_curve, alpha=0.08, color='#7C3AED')
ax2.axhline(y=TP/(TP+FP+FN+TN)*y_true.mean(), color='gray',
            linestyle='--', lw=1, label='Random Baseline')
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])
ax2.set_xlabel('Recall (Sensitivity)', fontsize=10)
ax2.set_ylabel('Precision', fontsize=10)
ax2.set_title('Precision-Recall Curve\nnnU-Net Blob Regression', fontsize=11, fontweight='bold')
ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3)
out2 = Path(__file__).parent / "figures" / "precision_recall_curve.png"
plt.savefig(out2, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Saved: {out2}")

# ── Figure 3: Probability Distribution ────────────────────────────────────
fig3, ax3 = plt.subplots(figsize=(6, 4))
pos_probs = y_prob[y_true == 1]
neg_probs = y_prob[y_true == 0]
ax3.scatter(range(len(pos_probs)), sorted(pos_probs, reverse=True),
            color='#DC2626', s=100, zorder=5, label='Aneurysm Present (GT+)')
ax3.scatter(range(len(neg_probs)), sorted(neg_probs, reverse=True),
            color='#2563EB', s=100, marker='s', zorder=5, label='No Aneurysm (GT−)')
ax3.axhline(y=0.5, color='black', lw=1.5, linestyle='--', label='Decision Threshold (τ=0.50)')
ax3.fill_between(range(max(len(pos_probs), len(neg_probs))),
                 0.5, 1.0, alpha=0.04, color='#DC2626')
ax3.fill_between(range(max(len(pos_probs), len(neg_probs))),
                 0.0, 0.5, alpha=0.04, color='#2563EB')
ax3.set_ylim(0, 1.05)
ax3.set_xlabel('Sorted Series Index', fontsize=10)
ax3.set_ylabel('Max Heatmap Probability', fontsize=10)
ax3.set_title('Predicted Probability Distribution\nby Ground Truth Class', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)
out3 = Path(__file__).parent / "figures" / "probability_distribution.png"
plt.savefig(out3, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  ✅ Saved: {out3}")

print(f"\n  ROC-AUC  : {roc_auc:.3f}")
print(f"  Avg Prec : {avg_prec:.3f}")
print("\n  All figures saved to IEEE_Paper/figures/")
