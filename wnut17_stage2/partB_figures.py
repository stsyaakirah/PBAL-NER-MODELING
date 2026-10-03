"""Section B figures. Black-and-grayscale only, no colored accent boxes (new style rule).
Reads results_partB/partB_analysis.json, results_partB/*.json, data_partB/split_stats.json,
and Section A results/ for the A-vs-B comparison. Writes report_figs_partB/*.png.
"""
import json, os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.edgecolor": "black",
                     "axes.labelcolor": "black", "text.color": "black", "xtick.color": "black",
                     "ytick.color": "black", "axes.titlecolor": "black", "figure.dpi": 150})
GRAY = "0.35"; LGRAY = "0.7"; DK = "0.15"
os.makedirs("report_figs_partB", exist_ok=True)

A = {m: json.load(open(f"results/{f}")) for m, f in [("CRF", "to_crf.json"), ("SVM", "to_svm.json")]}
import evalx
from wnut_data import read_conll
testA = read_conll  # not needed; use recorded
# Section A Bi-LSTM baseline F1/surface from prediction file
from evalx import evaluate as _ev
testA_seqs = None
def secA_bilstm():
    seqs = read_conll(os.path.join(__import__("wnut_data").DATA, "emerging.test.annotated"))
    pred = json.load(open("results/pred_to_bilstm_base_nobias_test.json"))
    e = _ev(seqs, pred); return round(e["entity"]["f1"] * 100, 2), round(e["surface"]["f1"] * 100, 2)
a_bl_f1, a_bl_surf = secA_bilstm()
secA = {
    "CRF": (round(A["CRF"]["test"]["entity"]["f1"] * 100, 2), round(A["CRF"]["test"]["surface"]["f1"] * 100, 2)),
    "SVM": (round(A["SVM"]["test"]["entity"]["f1"] * 100, 2), round(A["SVM"]["test"]["surface"]["f1"] * 100, 2)),
    "Bi-LSTM CRF": (a_bl_f1, a_bl_surf),
}
B = json.load(open("results_partB/partB_analysis.json"))
stats = json.load(open("data_partB/split_stats.json"))
models = [m for m in ("CRF", "SVM", "Bi-LSTM CRF") if m in B]

# --- Fig 1: entity F1, Section A vs B ---
fig, ax = plt.subplots(figsize=(7, 4))
x = np.arange(len(models)); w = 0.38
aF = [secA[m][0] for m in models]; bF = [B[m]["entity"]["f1"] for m in models]
ax.bar(x - w / 2, aF, w, color=LGRAY, edgecolor="black", label="Section A (official split)")
ax.bar(x + w / 2, bF, w, color=DK, edgecolor="black", label="Section B (pooled 80/10/10)")
for xi, v in zip(x - w / 2, aF): ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, bF): ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("F1 entity (test, %)")
ax.set_title("Entity F1: official split vs pooled split")
ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, max(bF) + 8)
fig.tight_layout(); fig.savefig("report_figs_partB/f_partB_f1_compare.png"); plt.close(fig)

# --- Fig 2: leakage ---
fig, ax = plt.subplots(figsize=(6, 4))
labels = ["Official test\n(vs train)", "Section B new dev", "Section B new test"]
vals = [2.0, stats["leakage"]["dev"]["leak_pct"], stats["leakage"]["test"]["leak_pct"]]
ax.bar(labels, vals, color=[LGRAY, GRAY, DK], edgecolor="black")
for i, v in enumerate(vals): ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=9)
ax.set_ylabel("Gold entity mentions also in train (%)")
ax.set_title("Entity leakage into training data")
ax.set_ylim(0, max(vals) + 4); fig.tight_layout()
fig.savefig("report_figs_partB/f_partB_leakage.png"); plt.close(fig)

# --- Fig 3: leaked vs unseen recall per model ---
fig, ax = plt.subplots(figsize=(7, 4))
groups = ["leaked (seen in new train)", "genuinely unseen"]
x = np.arange(len(models)); w = 0.38
lk = [B[m]["recall_leaked_vs_unseen"].get(groups[0], {}).get("recall") or 0 for m in models]
un = [B[m]["recall_leaked_vs_unseen"].get(groups[1], {}).get("recall") or 0 for m in models]
ax.bar(x - w / 2, lk, w, color=DK, edgecolor="black", label="leaked (surface seen in new train)")
ax.bar(x + w / 2, un, w, color=LGRAY, edgecolor="black", label="genuinely unseen")
for xi, v in zip(x - w / 2, lk): ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=9)
for xi, v in zip(x + w / 2, un): ax.text(xi, v + 1, f"{v:.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("Recall (%)")
ax.set_title("Where the gain comes from: recall on leaked vs unseen entities")
ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, 105)
fig.tight_layout(); fig.savefig("report_figs_partB/f_partB_leaked_recall.png"); plt.close(fig)

# --- Fig 4: per-type F1 (Section B) ---
fig, ax = plt.subplots(figsize=(8, 4))
types = ["person", "location", "corporation", "product", "creative-work", "group"]
x = np.arange(len(types)); w = 0.8 / len(models)
shades = ["0.15", "0.45", "0.7"]
for j, m in enumerate(models):
    vals = [B[m]["per_type"].get(t, {}).get("f1", 0) for t in types]
    ax.bar(x + j * w - 0.4 + w / 2, vals, w, color=shades[j % 3], edgecolor="black", label=m)
ax.set_xticks(x); ax.set_xticklabels(types, rotation=20, ha="right"); ax.set_ylabel("F1 (test, %)")
ax.set_title("Per-type F1 on the pooled split"); ax.legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig("report_figs_partB/f_partB_pertype.png"); plt.close(fig)

# --- Fig 5: confusion heatmaps (grayscale) ---
for m in models:
    Mx = np.array(B[m]["confusion"]["matrix"]); rows = B[m]["confusion"]["rows"]; cols = B[m]["confusion"]["cols"]
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Mx, cmap="Greys", aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    mx = Mx.max() if Mx.max() else 1
    for i in range(len(rows)):
        for j in range(len(cols)):
            if Mx[i, j]:
                ax.text(j, i, int(Mx[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if Mx[i, j] > 0.5 * mx else "black")
    ax.set_title(f"Entity confusion (SPURIOUS row) - {m}, pooled split")
    fig.tight_layout(); fig.savefig(f"report_figs_partB/f_partB_confusion_{m.split()[0].lower()}.png"); plt.close(fig)

# --- Fig 6: error breakdown ---
fig, ax = plt.subplots(figsize=(7, 4))
cats = ["benar", "salah_batas", "salah_tipe", "salah_batas_dan_tipe", "terlewat", "prediksi_palsu"]
labs = ["correct", "wrong boundary", "wrong type", "boundary+type", "missed", "spurious"]
x = np.arange(len(models)); bottom = np.zeros(len(models)); shades = np.linspace(0.15, 0.85, len(cats))
for c, lab, sh in zip(cats, labs, shades):
    vals = [B[m]["error_breakdown"].get(c, 0) for m in models]
    ax.bar(x, vals, 0.6, bottom=bottom, color=str(sh), edgecolor="black", label=lab)
    bottom += np.array(vals)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("gold entities / predictions")
ax.set_title("Error breakdown on the pooled split")
ax.legend(frameon=False, fontsize=8, ncol=2); fig.tight_layout()
fig.savefig("report_figs_partB/f_partB_errors.png"); plt.close(fig)

print("Section B figures written to report_figs_partB/:", sorted(os.listdir("report_figs_partB")), flush=True)
