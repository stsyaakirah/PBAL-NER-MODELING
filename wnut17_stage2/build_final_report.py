"""Comprehensive comparison report (English, black-only, no colored boxes, no em-dashes).
Compares three approaches on WNUT17:
  A. Official split (Twitter-only): CRF, SVM, Bi-LSTM CRF (+ ensemble experiments).
  B. Frozen RoBERTa enhancement of the Bi-LSTM CRF (official split).
  C. Section B pooled 80/10/10 random split (ablation).
Writes report_figs_final/*.png and NER_WNUT17_Comprehensive_Comparison.pdf.
"""
import os, json, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak)
from wnut_data import load, DATA
from evalx import evaluate

# ---------------- gather numbers ----------------
def ev_pred(split_seqs, path):
    return evaluate(split_seqs, json.load(open(path)))
testA = load("test")
def E(ev): return round(ev["entity"]["f1"] * 100, 2)
def S(ev): return round(ev["surface"]["f1"] * 100, 2)
def PR(ev): return (round(ev["entity"]["precision"] * 100, 2), round(ev["entity"]["recall"] * 100, 2))

A_crf = json.load(open("results/to_crf.json"))["test"]
A_svm = json.load(open("results/to_svm.json"))["test"]
A_bl = ev_pred(testA, "results/pred_to_bilstm_base_nobias_test.json")
A_ens = json.load(open("results/ensemble_6model.json"))["test_delta0"]
RB = json.load(open("results/bilstm_roberta.json"))
RB0, RBd = RB["test_delta0"], RB["test"]
B = json.load(open("results_partB/partB_analysis.json"))
st = json.load(open("data_partB/split_stats.json"))
SHARED = 41.86

# ---------------- figures (black / grayscale) ----------------
plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 11, "axes.edgecolor": "black",
                     "axes.labelcolor": "black", "text.color": "black", "xtick.color": "black",
                     "ytick.color": "black", "axes.titlecolor": "black", "figure.dpi": 150})
DK, GR, LG = "0.15", "0.45", "0.72"
os.makedirs("report_figs_final", exist_ok=True)

# Fig 1: improvement path on the official split (entity F1)
fig, ax = plt.subplots(figsize=(7.6, 4.2))
labels = ["Bi-LSTM\nbaseline", "6-model\nensemble", "+RoBERTa\n(delta 0)", "+RoBERTa\n(dev-tuned)"]
vals = [E(A_bl), E(A_ens), E(RB0), E(RBd)]
bars = ax.bar(range(4), vals, color=[LG, LG, DK, DK], edgecolor="black")
for i, v in enumerate(vals): ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=10)
ax.axhline(SHARED, ls="--", color="0.3", lw=1); ax.text(3.4, SHARED + 0.4, "shared-task best 41.86", ha="right", fontsize=8, color="0.3")
ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylabel("F1 entity (test, %)")
ax.set_title("Official split: the improvement path for the Bi-LSTM CRF")
ax.set_ylim(0, 60); fig.tight_layout(); fig.savefig("report_figs_final/f_path.png"); plt.close(fig)

# Fig 2: entity vs surface across the headline methods
fig, ax = plt.subplots(figsize=(8.2, 4.2))
methods = ["CRF (A)", "SVM (A)", "Bi-LSTM (A)", "Bi-LSTM ens (A)", "Bi-LSTM+RoBERTa (A)"]
ent = [E(A_crf), E(A_svm), E(A_bl), E(A_ens), E(RBd)]
sur = [S(A_crf), S(A_svm), S(A_bl), S(A_ens), S(RBd)]
x = np.arange(len(methods)); w = 0.38
ax.bar(x - w / 2, ent, w, color=DK, edgecolor="black", label="F1 entity")
ax.bar(x + w / 2, sur, w, color=LG, edgecolor="black", label="F1 surface")
for xi, v in zip(x - w / 2, ent): ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, sur): ax.text(xi, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(methods, rotation=15, ha="right"); ax.set_ylabel("F1 (test, %)")
ax.set_title("Entity and surface F1 on the official split"); ax.legend(frameon=False, fontsize=9)
ax.set_ylim(0, 60); fig.tight_layout(); fig.savefig("report_figs_final/f_entsurf.png"); plt.close(fig)

# Fig 3: three-way per model (A official vs B pooled), plus RoBERTa marker
fig, ax = plt.subplots(figsize=(8, 4.2))
models = ["CRF", "SVM", "Bi-LSTM CRF"]
aF = [E(A_crf), E(A_svm), E(A_bl)]
bF = [B["CRF"]["entity"]["f1"], B["SVM"]["entity"]["f1"], B["Bi-LSTM CRF"]["entity"]["f1"]]
x = np.arange(3); w = 0.38
ax.bar(x - w / 2, aF, w, color=LG, edgecolor="black", label="Section A (official split)")
ax.bar(x + w / 2, bF, w, color=GR, edgecolor="black", label="Section B (pooled split)")
ax.plot([2], [E(RBd)], marker="*", markersize=16, color="black", label="A + RoBERTa (Bi-LSTM)")
ax.text(2, E(RBd) + 1, f"{E(RBd):.1f}", ha="center", fontsize=9)
for xi, v in zip(x - w / 2, aF): ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
for xi, v in zip(x + w / 2, bF): ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("F1 entity (test, %)")
ax.set_title("Per model: official split, pooled split, and the RoBERTa enhancement")
ax.legend(frameon=False, fontsize=9); ax.set_ylim(0, 60)
fig.tight_layout(); fig.savefig("report_figs_final/f_threeway.png"); plt.close(fig)

# ---------------- PDF ----------------
BLACKC = colors.black; GRIDG = colors.Color(0.6, 0.6, 0.6)
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=BLACKC, fontName="Times-Bold", fontSize=16, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=BLACKC, fontName="Times-Bold", fontSize=12.5, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], textColor=BLACKC, fontName="Times-Roman", fontSize=10.3, leading=14, spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=9, leading=12, textColor=colors.Color(0.25, 0.25, 0.25))
CAP = ParagraphStyle("CAP", parent=SMALL, alignment=1)

def tbl(data, widths):
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([("FONT", (0, 0), (-1, -1), "Times-Roman", 9),
                           ("GRID", (0, 0), (-1, -1), 0.4, GRIDG), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("ALIGN", (1, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3), ("BACKGROUND", (0, 0), (-1, 0), colors.black),
                           ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONT", (0, 0), (-1, 0), "Times-Bold", 9)]))
    return t

def img(path, w):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize(); return Image(path, width=w, height=w * ih / iw)

el = []
el.append(Paragraph("WNUT17 Named Entity Recognition: Comparative Report", H1))
el.append(Paragraph("Kelompok 1, Pemrosesan Bahasa Alami Lanjut. This report compares three approaches to NER on the "
                    "WNUT17 emerging-entities corpus: (A) the three required models on the official domain-separated split, "
                    "(B) a frozen-RoBERTa enhancement of the Bi-LSTM CRF on that same split, and (C) a pooled 80/10/10 "
                    "random-split ablation. Section A is the primary, honest result. All models use seed 42; every tuning "
                    "decision is made on dev; test is scored once.", BODY))

el.append(Paragraph("1. Dataset and evaluation", H2))
el.append(Paragraph("WNUT17 has six entity types (person, location, corporation, product, creative-work, group) in IOB2. "
                    "The official split is domain-separated on purpose: train is Twitter 2010, dev is YouTube 2017, test is "
                    "Reddit, StackExchange and Twitter 2017. About 98 percent of test entity surfaces never appear in train, "
                    "so the task rewards handling novel entities rather than memorizing. Two official metrics, both entity level: "
                    "strict entity F1 (seqeval, exact span and type) and surface F1 (unique lowercased surface-form and type "
                    "pairs). We also report a 2000-resample bootstrap interval on the test set.", BODY))

el.append(Paragraph("2. Section A: the official split (primary result)", H2))
el.append(Paragraph("Three models with the finalized settings, trained on the Twitter train file only. The CRF and SVM are "
                    "lexical-feature models; the Bi-LSTM CRF adds frozen GloVe Twitter 200d, a char-CNN and a capitalization "
                    "embedding. Because dev and test are different domains, dev-tuned single changes did not transfer; only an "
                    "ensemble of six seed-42 variants rose on both dev and test.", BODY))
el.append(tbl([["Model", "P", "R", "F1 entity", "F1 surface"],
               ["CRF", f"{PR(A_crf)[0]:.1f}", f"{PR(A_crf)[1]:.1f}", f"{E(A_crf):.1f}", f"{S(A_crf):.1f}"],
               ["SVM", f"{PR(A_svm)[0]:.1f}", f"{PR(A_svm)[1]:.1f}", f"{E(A_svm):.1f}", f"{S(A_svm):.1f}"],
               ["Bi-LSTM CRF (baseline)", f"{PR(A_bl)[0]:.1f}", f"{PR(A_bl)[1]:.1f}", f"{E(A_bl):.1f}", f"{S(A_bl):.1f}"],
               ["Bi-LSTM CRF (6-model ensemble)", f"{A_ens['entity']['precision']*100:.1f}", f"{A_ens['entity']['recall']*100:.1f}", f"{E(A_ens):.1f}", f"{S(A_ens):.1f}"]],
              [6.2 * cm, 2.0 * cm, 2.0 * cm, 2.6 * cm, 2.6 * cm]))
el.append(Paragraph("Ensemble experiments (all Twitter-only, seed-based): enlarging the ensemble from 6 to 9 structural "
                    "variants gave 41.11, and a 5-seed baseline ensemble gave 41.14; surface F1 stayed 39.63 in every case. "
                    "Variance-reduction ensembling has plateaued. The lexical CRF and SVM sit far lower (16.8 and 29.8), which "
                    "is expected: with 98 percent novel surfaces, word-identity features rarely fire.", BODY))

el.append(Paragraph("3. The RoBERTa enhancement (same official split)", H2))
el.append(Paragraph("The one lever that moved both metrics is contextual embeddings. We add a frozen RoBERTa-base vector "
                    "(768d, last hidden state, subwords mean-pooled to each token, extracted per sentence so context is used) "
                    "as an extra input to the same Bi-LSTM CRF. RoBERTa is not fine-tuned and no labels touch it; the model is "
                    "still a Bi-LSTM CRF, still Twitter-only, seed 42, with the O-bias delta chosen on dev.", BODY))
el.append(tbl([["Bi-LSTM CRF + RoBERTa", "P", "R", "F1 entity", "F1 surface", "entity 95% CI"],
               ["decode delta 0", f"{PR(RB0)[0]:.1f}", f"{PR(RB0)[1]:.1f}", f"{E(RB0):.1f}", f"{S(RB0):.1f}", "[49.2, 55.7]"],
               ["dev-tuned delta 1.0", f"{PR(RBd)[0]:.1f}", f"{PR(RBd)[1]:.1f}", f"{E(RBd):.1f}", f"{S(RBd):.1f}", "[51.0, 57.2]"]],
              [5.0 * cm, 1.7 * cm, 1.7 * cm, 2.4 * cm, 2.4 * cm, 3.0 * cm]))
el.append(Paragraph("Both metrics rise about 12 to 13 points over the ensemble. The bootstrap lower bound alone (51.0) is far "
                    "above the previous best (41.35), so the gain is real, not noise. Per type (dev-tuned): person 69.2, "
                    "location 64.5, group 42.8, creative-work 32.9, corporation 30.9, product 26.7. Honest framing: this clears "
                    "the internal target and passes the shared-task best of 41.86, but RoBERTa is a 2019 contextual model the "
                    "2017 shared-task systems did not have, so it is not a like-for-like win over them. It is the expected effect "
                    "of contextual embeddings, reported as a labeled enhancement to the Bi-LSTM CRF, not a fourth model.", BODY))
el.append(img("report_figs_final/f_path.png", 12 * cm))
el.append(Paragraph("Figure 1. The improvement path on the official split. Ensembling plateaus near the shared-task line; "
                    "frozen RoBERTa features clear it.", CAP))

el.append(PageBreak())
el.append(img("report_figs_final/f_entsurf.png", 13 * cm))
el.append(Paragraph("Figure 2. Entity and surface F1 together, official split. RoBERTa lifts both, not just entity F1.", CAP))

el.append(Paragraph("4. Section B: pooled 80/10/10 split (ablation)", H2))
lt = st["leakage"]["test"]["leak_pct"]
el.append(Paragraph("Section B asks a different question: if we ignore the domain design and shuffle all three files into a "
                    "conventional random split, how much easier does the task get. Mixing was authorized for this section only. "
                    "The split is document-level (seed 42), conservation is exact (5,690 sequences, 101,857 tokens, 3,890 "
                    "entities) and BIO violations are zero. Same models, same hyperparameters, no re-search.", BODY))
el.append(tbl([["Model", "A official F1", "B pooled F1", "A surface", "B surface", "test leakage"],
               ["CRF", f"{E(A_crf):.1f}", f"{B['CRF']['entity']['f1']:.1f}", f"{S(A_crf):.1f}", f"{B['CRF']['surface_f1']:.1f}", f"2% -> {lt}%"],
               ["SVM", f"{E(A_svm):.1f}", f"{B['SVM']['entity']['f1']:.1f}", f"{S(A_svm):.1f}", f"{B['SVM']['surface_f1']:.1f}", f"2% -> {lt}%"],
               ["Bi-LSTM CRF", f"{E(A_bl):.1f}", f"{B['Bi-LSTM CRF']['entity']['f1']:.1f}", f"{S(A_bl):.1f}", f"{B['Bi-LSTM CRF']['surface_f1']:.1f}", f"2% -> {lt}%"]],
              [3.2 * cm, 2.8 * cm, 2.8 * cm, 2.3 * cm, 2.3 * cm, 2.6 * cm]))
el.append(Paragraph("Every model jumps, but this is not better NER. Entity leakage into train rises from about 2 percent "
                    "(official test) to 21.1 percent (pooled test), and recall concentrates on leaked entities (for example the "
                    "CRF recalls 64.8 percent of leaked entities but only 23.8 percent of genuinely unseen ones). The gain is "
                    "leakage plus feature coverage, not modeling. Section B is therefore not comparable to the shared-task "
                    "leaderboard, whose premise is the novelty-guaranteed split. It is an ablation that measures how much of the "
                    "difficulty is entity novelty. The test set here is small (256 entities), so its intervals are wide.", BODY))

el.append(Paragraph("5. All three approaches together", H2))
el.append(img("report_figs_final/f_threeway.png", 13 * cm))
el.append(Paragraph("Figure 3. Per model: official split (Section A), pooled split (Section B), and the RoBERTa enhancement.", CAP))
el.append(Paragraph("The most useful comparison is the Bi-LSTM CRF. On the hard official split it scores about 40 to 41. "
                    "Removing the novelty challenge by pooling raises it to 51.8, almost entirely through leakage. Adding RoBERTa "
                    "on the hard split reaches 54.1, matching and even exceeding the easy-split number while keeping the honest "
                    "split intact. In other words, contextual embeddings genuinely solve much of what makes WNUT17 hard, whereas "
                    "the pooled split only hides it.", BODY))

el.append(Paragraph("6. Conclusion", H2))
el.append(Paragraph("Section A is the primary result and the honest measure of the three required models: CRF 16.8, SVM 29.8, "
                    "Bi-LSTM CRF 40.4, ensemble 41.35 entity F1. Ensembling cannot push past this because it only reduces "
                    "variance. Frozen RoBERTa features lift the Bi-LSTM CRF to 54.1 entity and 52.9 surface F1, a real gain of "
                    "about 13 points on both metrics, achieved without leaving the domain-separated split and without a fourth "
                    "model. The Section B ablation shows why the official numbers are low in the first place: most of WNUT17's "
                    "difficulty is entity novelty, and a random split hides that difficulty behind 21 percent leakage rather than "
                    "solving it. The recommended headline for this project is Section A as the honest baseline and the RoBERTa "
                    "enhancement as the improvement, with Section B reported as an ablation and clearly marked as not comparable "
                    "to the shared task.", BODY))

SimpleDocTemplate("NER_WNUT17_Comprehensive_Comparison.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.8 * cm, bottomMargin=1.8 * cm).build(el)
print("NER_WNUT17_Comprehensive_Comparison.pdf written", flush=True)
print("numbers:",
      "A CRF", E(A_crf), "SVM", E(A_svm), "BL", E(A_bl), "ens", E(A_ens),
      "| RoBERTa d0", E(RB0), "devd", E(RBd),
      "| B CRF", B["CRF"]["entity"]["f1"], "SVM", B["SVM"]["entity"]["f1"], "BL", B["Bi-LSTM CRF"]["entity"]["f1"], flush=True)
