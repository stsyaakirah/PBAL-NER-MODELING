"""Section B addendum PDF (black-only, no colored boxes, no em-dashes).
Reads results_partB/*.json, data_partB/split_stats.json, Section A results/, and figures.
Writes NER_WNUT17_PartB_MixedSplit.pdf.
"""
import json, os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak)

BLACK = colors.black; GRIDG = colors.Color(0.6, 0.6, 0.6); HDR = colors.black
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], textColor=BLACK, fontName="Times-Bold", fontSize=16, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=BLACK, fontName="Times-Bold", fontSize=12.5, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("BODY", parent=ss["BodyText"], textColor=BLACK, fontName="Times-Roman", fontSize=10.3, leading=14, spaceAfter=6)
SMALL = ParagraphStyle("SMALL", parent=BODY, fontSize=9, leading=12, textColor=colors.Color(0.25, 0.25, 0.25))
CAP = ParagraphStyle("CAP", parent=SMALL, alignment=1)

st = json.load(open("data_partB/split_stats.json"))
B = json.load(open("results_partB/partB_analysis.json"))
A = {m: json.load(open(f"results/{f}")) for m, f in [("CRF", "to_crf.json"), ("SVM", "to_svm.json")]}
from wnut_data import read_conll, DATA
from evalx import evaluate
_bl_seqs = read_conll(os.path.join(DATA, "emerging.test.annotated"))
_bl = evaluate(_bl_seqs, json.load(open("results/pred_to_bilstm_base_nobias_test.json")))
secA = {
    "CRF": (A["CRF"]["test"]["entity"]["precision"] * 100, A["CRF"]["test"]["entity"]["recall"] * 100, A["CRF"]["test"]["entity"]["f1"] * 100, A["CRF"]["test"]["surface"]["f1"] * 100),
    "SVM": (A["SVM"]["test"]["entity"]["precision"] * 100, A["SVM"]["test"]["entity"]["recall"] * 100, A["SVM"]["test"]["entity"]["f1"] * 100, A["SVM"]["test"]["surface"]["f1"] * 100),
    "Bi-LSTM CRF": (_bl["entity"]["precision"] * 100, _bl["entity"]["recall"] * 100, _bl["entity"]["f1"] * 100, _bl["surface"]["f1"] * 100),
}
models = [m for m in ("CRF", "SVM", "Bi-LSTM CRF") if m in B]

def tbl(data, widths, header=True):
    t = Table(data, colWidths=widths)
    style = [("FONT", (0, 0), (-1, -1), "Times-Roman", 9),
             ("GRID", (0, 0), (-1, -1), 0.4, GRIDG),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ALIGN", (1, 0), (-1, -1), "CENTER"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), HDR), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONT", (0, 0), (-1, 0), "Times-Bold", 9)]
    t.setStyle(TableStyle(style)); return t

def img(path, w):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(path).getSize(); return Image(path, width=w, height=w * ih / iw)

E = []
E.append(Paragraph("WNUT17 NER, Part 2 Section B: Pooled 80/10/10 Split", H1))
E.append(Paragraph("Kelompok 1, Pemrosesan Bahasa Alami Lanjut. Addendum to the Section A report (official WNUT17 split). "
                   "This experiment pools the three official files and re-splits them at random. Mixing was authorized for this section only. "
                   "The result below is an ablation, not a new leaderboard number.", BODY))

E.append(Paragraph("1. What Section B does and why", H2))
E.append(Paragraph("Section A keeps the official WNUT17 design: train is Twitter 2010, dev is YouTube, test is Reddit, StackExchange and Twitter 2017. "
                   "That separation is deliberate. It forces every model to tag entities it has never seen, which is the whole difficulty of the task. "
                   "Section B asks the opposite question: if we ignore the design and shuffle everything into a conventional random split, how much easier does the task get, and why. "
                   "We build the pooled split, run the same three models with the same finalized hyperparameters, and read the gap against Section A.", BODY))

E.append(Paragraph("2. Building the split", H2))
d = st["documents"]; c = st["counts"]; p = st["achieved_pct"]
E.append(Paragraph(f"The split unit is the document, keyed as (source file, document id) so a tweet's sequences never cross splits and no document from one original file collides with another. "
                   f"We shuffle the {d['total']} documents with numpy RandomState(42) and cut 80/10/10 by document count "
                   f"({d['train']}/{d['dev']}/{d['test']} documents). Three new CoNLL files go to data_partB/.", BODY))
cons = tbl([["Split", "Sequences", "Tokens", "Entities", "Seq %", "Token %", "Ent %"]] +
           [[s.capitalize(), c[s]["seqs"], c[s]["tokens"], c[s]["entities"], p[s]["seqs"], p[s]["tokens"], p[s]["entities"]] for s in ("train", "dev", "test")] +
           [["Total", sum(c[s]["seqs"] for s in c), sum(c[s]["tokens"] for s in c), sum(c[s]["entities"] for s in c), 100.0, 100.0, 100.0]],
           [2.4 * cm, 2.3 * cm, 2.0 * cm, 2.0 * cm, 1.6 * cm, 1.9 * cm, 1.6 * cm])
E.append(cons)
E.append(Paragraph(f"Conservation check passes exactly: the three new files hold 5,690 sequences, 101,857 tokens and 3,890 entities, identical to the pooled originals. "
                   f"BIO violations are zero in all three files. Because documents vary in size, the achieved entity percentages deviate from 10 percent for dev and test; "
                   f"this is measured, not assumed, and is reported above.", BODY))

E.append(Paragraph("3. Entity leakage, the main driver", H2))
ld, lt = st["leakage"]["dev"], st["leakage"]["test"]
E.append(Paragraph(f"Before training anything, we measure how many gold entity mentions in the new dev and test also appear as an entity in the new train. "
                   f"On the official split this figure is about 2 percent for test against train, by design. On the pooled split it rises to "
                   f"{ld['leak_pct']} percent for the new dev and {lt['leak_pct']} percent for the new test. "
                   f"{st['entities_in_multiple_splits']} distinct entity surfaces now appear in at least two splits, led by frequent names such as "
                   f"{', '.join(w for w, _ in st['top_repeated_multi_split'][:6])}. This leakage number sits next to every F1 in this section, because it is the main reason the numbers move.", BODY))
E.append(img("report_figs_partB/f_partB_leakage.png", 11 * cm))
E.append(Paragraph("Figure 1. Entity leakage into the training data: official test vs the pooled split.", CAP))

E.append(PageBreak())
E.append(Paragraph("4. Results: Section A vs Section B", H2))
rows = [["Model", "Split", "P", "R", "F1 entity", "F1 surface", "Test leakage"]]
for m in models:
    pa = secA[m]; rows.append([m, "A official", f"{pa[0]:.1f}", f"{pa[1]:.1f}", f"{pa[2]:.1f}", f"{pa[3]:.1f}", "~2%"])
    bb = B[m]; rows.append([m, "B pooled", f"{bb['entity']['precision']:.1f}", f"{bb['entity']['recall']:.1f}",
                            f"{bb['entity']['f1']:.1f}", f"{bb['surface_f1']:.1f}", f"{lt['leak_pct']}%"])
E.append(tbl(rows, [3.0 * cm, 2.2 * cm, 1.5 * cm, 1.5 * cm, 2.2 * cm, 2.3 * cm, 2.2 * cm]))
E.append(Paragraph("All Section B test scores carry a 2000-resample bootstrap interval: " +
                   "; ".join(f"{m} {B[m]['bootstrap_test_f1']['mean']:.1f} (95% CI {B[m]['bootstrap_test_f1']['lo']:.1f} to {B[m]['bootstrap_test_f1']['hi']:.1f})" for m in models) + ".", BODY))
E.append(img("report_figs_partB/f_partB_f1_compare.png", 12 * cm))
E.append(Paragraph("Figure 2. Entity F1 on the official split (Section A) and the pooled split (Section B).", CAP))

E.append(Paragraph("5. Why the numbers move, and what they do not mean", H2))
E.append(Paragraph("The pooled split is easier for two separate reasons, and they must not be confused with a better model. "
                   "First, leakage: entities that recur across the corpus now show up in both train and test, so the model can memorize them. "
                   "Second, feature coverage: with a random 80 percent of all documents in train, the CRF and SVM simply see more vocabulary, more word shapes and more context patterns than the Twitter-only train file offered. "
                   "Neither effect is available on the official split, and neither reflects better handling of novel entities.", BODY))
E.append(img("report_figs_partB/f_partB_leaked_recall.png", 12 * cm))
E.append(Paragraph("Figure 3. Recall split by whether the test entity leaked into the new train or is still genuinely unseen. "
                   "The gain concentrates on leaked entities; recall on genuinely unseen entities stays close to the Section A regime.", CAP))
E.append(Paragraph("This is why the Section B result is not comparable to the WNUT17 shared task. The leaderboard, and the 41.86 shared-task best, are defined on the domain-separated split whose premise is that test entities are novel. "
                   "Reshuffling removes that premise. Section B should be read as an ablation that isolates how much of the task's difficulty comes from novel and rare entities specifically, rather than from general NER difficulty. "
                   "On this evidence, most of the difficulty is the novelty: remove it and even the lexical CRF, which scored 16.8 on the official split, moves into the 40s.", BODY))

E.append(PageBreak())
E.append(Paragraph("6. Per-type, confusion and errors", H2))
E.append(img("report_figs_partB/f_partB_pertype.png", 12 * cm))
E.append(Paragraph("Figure 4. Per-type F1 on the pooled split.", CAP))
E.append(img("report_figs_partB/f_partB_errors.png", 11 * cm))
E.append(Paragraph("Figure 5. Error breakdown on the pooled split.", CAP))
for m in models:
    fn = f"report_figs_partB/f_partB_confusion_{m.split()[0].lower()}.png"
    if os.path.exists(fn):
        E.append(img(fn, 10.5 * cm)); E.append(Paragraph(f"Figure. Entity confusion with a SPURIOUS row, {m}.", CAP))

E.append(Paragraph("7. Conclusion", H2))
best = max(models, key=lambda m: B[m]["entity"]["f1"])
E.append(Paragraph(f"On the pooled 80/10/10 split every model improves sharply, with {best} reaching F1 {B[best]['entity']['f1']:.1f} on the new test. "
                   f"The improvement tracks a leakage rate of {lt['leak_pct']} percent, up from about 2 percent on the official split, and extra feature coverage for the classical models. "
                   f"It is an ablation of the dataset's difficulty, not a better NER system, and it does not compare to the shared-task leaderboard. Section A remains the primary, honest result for this assignment.", BODY))

SimpleDocTemplate("NER_WNUT17_PartB_MixedSplit.pdf", pagesize=A4,
                  leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm).build(E)
print("NER_WNUT17_PartB_MixedSplit.pdf written", flush=True)
