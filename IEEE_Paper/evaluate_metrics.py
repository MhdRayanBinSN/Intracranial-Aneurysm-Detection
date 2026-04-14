"""
=======================================================
 nnU-Net Evaluation Metrics — Instant (No Re-inference)
 Run in Jupyter or as plain Python
=======================================================
Uses the 15-series results already stored in EVALUATION_RESULTS.md.
No model loading needed — just sklearn + numpy.
"""

import numpy as np
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix,
    classification_report, roc_auc_score,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# ── 1. Ground truth and predictions (from EVALUATION_RESULTS.md) ──────────────
#    1 = aneurysm present, 0 = no aneurysm

series_ids = [
    "...10004044428023505",   # TN
    "...10004684224894397",   # TN
    "...10005158603912009",   # FN  ← missed (Other Posterior Circ.)
    "...10009383108068795",   # TN
    "...10012790035410518",   # FP  ← false alarm
    "...10014757658335054",   # TN
    "...10021411248005513",   # FP  ← false alarm
    "...10022688097731894",   # TN
    "...10022796280698534",   # TP  ✓ Right MCA (74.2%)
    "...10023411164590664",   # TP  ✓ Right MCA (56.2%)
    "...10030095840917973",   # FN  ← missed (R. Infraclinoid ICA)
    "...10030804647049037",   # TN
    "...10034081836061566",   # TP  ✓ Anterior Comm. (67.2%)
    "...10035643165968342",   # TP  ✓ Multi-aneurysm partial (67.4%)
    "...10035782880104673",   # TN
]

y_true = np.array([0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 0])
y_pred = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0])

# Peak heatmap probabilities (max across all 13 location channels)
y_prob = np.array([
    0.009, 0.218, 0.449, 0.154, 0.682,
    0.224, 0.525, 0.041, 0.742, 0.562,
    0.225, 0.249, 0.672, 0.674, 0.008
])

# ── 2. Core Metrics ────────────────────────────────────────────────────────────
print("=" * 55)
print("   nnU-Net Series-Level Evaluation Metrics")
print("=" * 55)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy    = accuracy_score(y_true, y_pred)
sensitivity = recall_score(y_true, y_pred)          # = TP / (TP + FN)
specificity = tn / (tn + fp)                         # TN / (TN + FP)
precision   = precision_score(y_true, y_pred, zero_division=0)
f1          = f1_score(y_true, y_pred)
auc         = roc_auc_score(y_true, y_prob)

print(f"\n  Confusion Matrix:")
print(f"    TP = {tp}   FP = {fp}")
print(f"    FN = {fn}   TN = {tn}")
print(f"\n  Accuracy        : {accuracy:.3f}  ({accuracy*100:.1f}%)")
print(f"  Sensitivity     : {sensitivity:.3f}  ({sensitivity*100:.1f}%)")
print(f"  Specificity     : {specificity:.3f}  ({specificity*100:.1f}%)")
print(f"  Precision       : {precision:.3f}  ({precision*100:.1f}%)")
print(f"  F1 Score        : {f1:.3f}")
print(f"  ROC-AUC         : {auc:.3f}")
print(f"\n  Full Report:")
print(classification_report(y_true, y_pred,
      target_names=["No Aneurysm", "Aneurysm"]))

# ── 3. Per-Location Metrics ────────────────────────────────────────────────────
print("=" * 55)
print("   Per-Location Detection (nnU-Net)")
print("=" * 55)

locations = {
    "Right MCA":             {"gt": 2, "det": 2, "correct": 2},
    "Anterior Comm. Art.":   {"gt": 1, "det": 1, "correct": 1},
    "Left MCA":              {"gt": 1, "det": 1, "correct": 1},
    "Right Supraclinoid ICA":{"gt": 1, "det": 1, "correct": 1},
    "Other Post. Circ.":     {"gt": 1, "det": 0, "correct": 0},
    "Right Infraclinoid ICA":{"gt": 1, "det": 0, "correct": 0},
    "Right ACA":             {"gt": 1, "det": 0, "correct": 0},
}

total_gt = total_det = total_correct = 0
print(f"\n  {'Location':<28} {'GT':>4} {'Det':>4} {'OK':>4} {'Recall':>8}")
print("  " + "-" * 52)
for loc, v in locations.items():
    recall = v["correct"] / v["gt"] if v["gt"] > 0 else 0
    print(f"  {loc:<28} {v['gt']:>4} {v['det']:>4} {v['correct']:>4} {recall*100:>7.0f}%")
    total_gt      += v["gt"]
    total_det     += v["det"]
    total_correct += v["correct"]

loc_precision = total_correct / total_det if total_det > 0 else 0
loc_recall    = total_correct / total_gt  if total_gt  > 0 else 0
loc_f1        = (2 * loc_precision * loc_recall / (loc_precision + loc_recall)
                 if (loc_precision + loc_recall) > 0 else 0)

print(f"\n  Location Precision : {loc_precision:.3f}  ({loc_precision*100:.1f}%)")
print(f"  Location Recall    : {loc_recall:.3f}  ({loc_recall*100:.1f}%)")
print(f"  Location F1 Score  : {loc_f1:.3f}")

# ── 4. Plots ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("nnU-Net Evaluation — 15-Series Test Set", fontsize=13, fontweight='bold')

# --- Confusion Matrix ---
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Aneurysm", "Aneurysm"])
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")

# --- Metrics Bar Chart ---
metric_names  = ["Accuracy", "Sensitivity", "Specificity", "Precision", "F1", "AUC"]
metric_values = [accuracy, sensitivity, specificity, precision, f1, auc]
colors = ['#4C72B0','#DD8452','#55A868','#C44E52','#8172B2','#937860']
bars = axes[1].barh(metric_names, [v*100 for v in metric_values],
                    color=colors, edgecolor='white', height=0.6)
axes[1].set_xlim(0, 100)
axes[1].set_xlabel("Score (%)")
axes[1].set_title("Series-Level Metrics")
for bar, val in zip(bars, metric_values):
    axes[1].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                 f"{val*100:.1f}%", va='center', fontsize=9)

# --- Per-Location Recall ---
loc_names   = [l[:18] for l in locations.keys()]
loc_recalls = [v["correct"]/v["gt"]*100 if v["gt"] > 0 else 0
               for v in locations.values()]
bar_colors  = ['#2ecc71' if r == 100 else '#e74c3c' for r in loc_recalls]
axes[2].barh(loc_names, loc_recalls, color=bar_colors, edgecolor='white', height=0.6)
axes[2].set_xlim(0, 110)
axes[2].set_xlabel("Recall (%)")
axes[2].set_title("Per-Location Recall")
axes[2].axvline(x=100, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("evaluation_metrics_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("\n  Plot saved → evaluation_metrics_plot.png")
