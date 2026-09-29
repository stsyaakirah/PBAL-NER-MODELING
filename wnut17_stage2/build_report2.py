"""Builds the comprehensive Stage 2 report: paper-style body (Abstract ... Conclusion, References) followed by appendices.
All numbers are read from results/ (after the preprocessing fix) and results_prefix/ (before the fix)."""
import json, os, sys, pickle
sys.path.insert(0, "/home/claude/proj"); os.chdir("/home/claude/proj")
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from PIL import Image as PILImage
from seqeval.metrics.sequence_labeling import get_entities
from wnut_data import load

R = lambda p: json.load(open("results/" + p))
S, PRE, BS, CV, C2, SV, SV2 = R("summary.json"), json.load(open("results_prefix/summary.json")), R("bootstrap.json"), R("cv_final.json"), R("crf2.json"), R("svm.json"), R("svm2.json")
FS = R("fix_svm.json"); FB = R("fix_bilstm.json")
crf, svm, bi = S["crf"], S["svm"], S["bilstm"]; A = S["analysis"]
M3 = ["CRF", "SVM", "Bi-LSTM CRF"]; key = {"CRF": "crf", "SVM": "svm", "Bi-LSTM CRF": "bilstm"}
T6 = ["person", "location", "corporation", "product", "creative-work", "group"]; SUP = {"person": 429, "location": 150, "corporation": 66, "product": 127, "creative-work": 142, "group": 165}
p1 = lambda x: f"{x:.1f}"; p2 = lambda x: f"{x:.2f}"
assert bi["F1"] > svm["F1"] > crf["F1"], "ordering changed: revise the wording of the results text"
BI_NOTE = (f"seed 42, as fixed in Stage 1; seeds 43 and 44 give {p1(bi['seeds'][1])} and {p1(bi['seeds'][2])}, mean {p1(bi['mean3']['F1'])}" if bi["primary"] == "seed 42" else f"mean of three seeds, SD {p2(bi['std'])}")
WEAK = max(S[key[m]]["types"][t] for m in M3 for t in ("corporation", "product", "creative-work", "group"))
BI_ROWS = [[f"Bi-LSTM CRF ({bi['primary']})", p1(bi["P"]), p1(bi["R"]), p1(bi["F1"]), p1(bi["surf"]), p1(bi["tokacc"])]]
if bi["primary"] == "seed 42": BI_ROWS.append(["Bi-LSTM CRF (mean of seeds 42, 43, 44)", p1(bi["mean3"]["P"]), p1(bi["mean3"]["R"]), p1(bi["mean3"]["F1"]), p1(bi["mean3"]["surf"]), p1(bi["mean3"]["tokacc"])])
E = {m: S[key[m]]["errors"] for m in M3}; wrong = lambda e: e["salah_batas"] + e["salah_tipe"] + e["salah_batas_dan_tipe"]

# ---------------------------------------------------------------- styles and helpers
NAVY, GOLD, BG, LINE, INK, MUT, TINT = [colors.HexColor(c) for c in ("#12406A", "#B8960C", "#F4F5F7", "#D9DEE5", "#1F2937", "#5B6B7C", "#E6EEF6")]
def st(name, **kw):
    b = dict(fontName="Helvetica", fontSize=9.1, leading=13.0, textColor=INK, spaceAfter=4.5, alignment=TA_LEFT); b.update(kw); return ParagraphStyle(name, **b)
body = st("body"); h1 = st("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=11, spaceAfter=5, keepWithNext=1)
h2 = st("h2", fontName="Helvetica-Bold", fontSize=10.4, leading=13.5, textColor=NAVY, spaceBefore=7, spaceAfter=2.5, keepWithNext=1)
cap = st("cap", fontName="Helvetica-Oblique", fontSize=8, leading=10.6, textColor=MUT, spaceAfter=8, spaceBefore=2)
tcap = st("tcap", fontName="Helvetica-Bold", fontSize=8.4, leading=11, textColor=NAVY, spaceBefore=6, spaceAfter=2.5, keepWithNext=1)
note = st("note", fontSize=7.9, leading=10.6, textColor=MUT, spaceBefore=2, spaceAfter=6)
bul = st("bul", leftIndent=12, bulletIndent=1, spaceAfter=2.2)
al = {"l": TA_LEFT, "c": TA_CENTER, "r": TA_RIGHT}
cell = {a: st("c" + a, fontSize=8.1, leading=10.4, spaceAfter=0, alignment=al[a]) for a in "lcr"}
cellb = {a: st("cb" + a, fontName="Helvetica-Bold", fontSize=8.1, leading=10.4, spaceAfter=0, alignment=al[a]) for a in "lcr"}
chdr = {a: st("ch" + a, fontName="Helvetica-Bold", fontSize=8.1, leading=10.4, textColor=colors.white, spaceAfter=0, alignment=al[a]) for a in "lcr"}
cnt = {"t": 0, "f": 0}
def P(t, s=body): return Paragraph(t, s)
def B(items): return [Paragraph(t, bul, bulletText="\u2022") for t in items]
def table(rows, widths, align=None, hi=(), boldcol0=False):
    align = align or ["l"] + ["c"] * (len(rows[0]) - 1); data = []
    for r, row in enumerate(rows):
        data.append([Paragraph(str(v), chdr[align[c]] if r == 0 else (cellb[align[c]] if (boldcol0 and c == 0) or r in hi else cell[align[c]])) for c, v in enumerate(row)])
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    for r in range(1, len(rows)):
        if r in hi: sty.append(("BACKGROUND", (0, r), (-1, r), TINT))
        elif r % 2 == 0: sty.append(("BACKGROUND", (0, r), (-1, r), BG))
    t.setStyle(TableStyle(sty)); return t
def tbl(title, rows, widths, **kw):
    cnt["t"] += 1; return [Paragraph(f"Table {cnt['t']}. {title}", tcap), table(rows, widths, **kw), Spacer(1, 5)]
def fig(name, width_in, caption):
    cnt["f"] += 1; w, h = PILImage.open(f"report_figs/{name}.png").size; W = width_in * inch
    im = Image(f"report_figs/{name}.png", width=W, height=W * h / w); im.hAlign = "CENTER"
    return KeepTogether([im, Paragraph(f"<b>Figure {cnt['f']}.</b> {caption}", cap)])

# ---------------------------------------------------------------- derived quantities
gk = {}
try:
    import prep_glove
    need = pickle.load(open("glove_vocab.pkl", "rb"))["need_words"]
    for sp in ("train", "dev", "test"):
        toks = [(t, l) for x in load(sp) for t, l in zip(x["tokens"], x["labels"])]
        has = lambda t: any(k in need for k in prep_glove.glove_keys(t))
        gk[sp] = (100 * sum(has(t) for t, _ in toks) / len(toks), 100 * sum(has(t) for t, l in toks if l != "O") / sum(1 for _, l in toks if l != "O"))
    nkeys = len({k for sp in ("train", "dev", "test") for x in load(sp) for t in x["tokens"] for k in prep_glove.glove_keys(t)})
except Exception as e:
    print("coverage failed", e); nkeys = 0
cov = lambda sp, i: p1(gk[sp][i]) if sp in gk else "n/a"
ab = C2["ablation"]; dab = lambda i: ab[i]["F1"] - ab[i - 1]["F1"]
sa = {a["name"]: a for a in SV["ablation"]}; svf, svnw, svng = sa["LinearSVC balanced + GloVe (lengkap)"], sa["Tanpa pembobotan kelas"], sa["Tanpa vektor GloVe"]
grid = {(g["fitur"], g["c1"], g["c2"]): g for g in C2["grid"]}; clu_flip = grid[("lengkap", 0.3, 0.03)]["F1"] - grid[("tanpa_cluster", 0.3, 0.03)]["F1"]
svm_norep_test, svm_rep_test = svm["F1"], svm["repair_test"]["F1"]
prefix_sv = PRE["svm"]; G = A
UN = [G[m]["unseen"]["recall"] for m in M3]; LC = [G[m]["ada huruf kecil"]["recall"] for m in M3]

# examples for the qualitative section
test = load("test"); PRED = {m: json.load(open("results/" + f)) for m, f in {"CRF": "pred_fix_crf_test.json", "SVM": "pred_fix_svm_test.json", "Bi-LSTM CRF": "pred_fix_bilstm_test.json"}.items()}
spans = lambda x, lab: "; ".join(f"{' '.join(x['tokens'][b:e + 1])} = {t}" for t, b, e in get_entities(lab)) or "no entity"
exact = lambda i, m: set(get_entities(PRED[m][i])) == set(get_entities(test[i]["labels"]))
short = [i for i, x in enumerate(test) if get_entities(x["labels"]) and len(x["tokens"]) <= 18]
def pick(m, ok, n):
    c = [i for i in short if exact(i, m) == ok]; c.sort(key=lambda i: (-len(get_entities(test[i]["labels"])), len(test[i]["tokens"]))); return c[:n]
def ex_rows(n):
    rows = [["Model", "Result", "Sentence", "Gold", "Prediction"]]
    for m in M3:
        for ok, tag in ((True, "correct"), (False, "incorrect")):
            for i in pick(m, ok, n): rows.append([m, tag, " ".join(test[i]["tokens"]).replace("&", "&amp;"), spans(test[i], test[i]["labels"]).replace("&", "&amp;"), spans(test[i], PRED[m][i]).replace("&", "&amp;")])
    return rows

# ---------------------------------------------------------------- document
def footer(c, doc):
    c.saveState(); c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(2 * cm, 1.65 * cm, A4[0] - 2 * cm, 1.65 * cm)
    c.setFont("Helvetica", 7.6); c.setFillColor(MUT); c.drawString(2 * cm, 1.2 * cm, "NER on WNUT17: CRF, SVM and Bi-LSTM CRF | Kelompok 1 | Advanced NLP, UGM"); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page)); c.restoreState()
OUT = "/mnt/user-data/outputs/NER_WNUT17_Comprehensive_Report.pdf"
doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm, bottomMargin=2.1 * cm, title="Named Entity Recognition on WNUT17 with CRF, SVM and Bi-LSTM CRF", author="Kelompok 1, Pemrosesan Bahasa Alami Lanjut, UGM")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
W = 17.0; s = []

# ---- title and abstract
s += [P("ASSIGNMENT #2, STAGE 2  |  NER MODELING AND EVALUATION", st("kick", fontName="Helvetica-Bold", fontSize=8.4, textColor=GOLD, spaceAfter=5)),
      P("Named Entity Recognition on Noisy Social Media Text: A Comparison of CRF, SVM and Bi-LSTM CRF on WNUT17", st("title", fontName="Helvetica-Bold", fontSize=18.5, leading=23, textColor=NAVY, spaceAfter=5)),
      P("<b>Kelompok 1</b>, Pemrosesan Bahasa Alami Lanjut, Program Studi Magister Kecerdasan Artifisial, Universitas Gadjah Mada<br/>Amelia Gizzela Sheehan Auni, Aulia Gita Pratiwi, St. Syakirah, Aditya Iko Winarko, Royan Aditya", st("auth", fontSize=8.6, leading=12.4, spaceAfter=9))]
abstract = (f"WNUT17 tests whether named entity recognition (NER) systems can find entities they have never seen: 98% of the test surface forms are absent from training. "
            f"We implement three systems on the official splits: a feature-based CRF, a class-weighted linear SVM, and a Bi-LSTM CRF that combines frozen GloVe Twitter vectors, a character CNN and capitalization embeddings. "
            f"Hyperparameters are chosen on dev, final models are retrained on train plus dev, and each is scored once on the full gold test file (1,287 sequences, 1,079 entities). "
            f"The Bi-LSTM CRF reaches <b>{p1(bi['F1'])}</b> entity F1 ({BI_NOTE}), against <b>{p1(svm['F1'])}</b> for the SVM and <b>{p1(crf['F1'])}</b> for the CRF. "
            f"The Bi-LSTM CRF is on par with the best shared-task system (41.86), although the comparison is approximate because the released annotations were corrected. "
            f"Bootstrap intervals of about 3 F1 points show that its lead is clear while the SVM lead over the CRF is small. "
            f"Recall on unseen entities is only {p1(min(UN))} to {p1(max(UN))}%, and corporation, product, creative-work and group reach at most {p1(WEAK)} F1 for every model. "
            f"We also report three preprocessing bugs found and fixed after the first run, and their effect (under one F1 point).")
box = Table([[P("<b>Abstract</b>", st("ah", fontName="Helvetica-Bold", fontSize=9.6, textColor=NAVY, spaceAfter=2))], [P(abstract, st("ab", fontSize=8.8, leading=12.6, spaceAfter=3))],
             [P("<b>Keywords:</b> named entity recognition, WNUT17, emerging entities, CRF, SVM, Bi-LSTM CRF, GloVe, noisy text", st("kw", fontSize=8.1, textColor=MUT, spaceAfter=0))]], colWidths=[W * cm])
box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BG), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8)]))
s += [box]

# ---- 1 Introduction
s += [P("1. Introduction", h1),
      P("Named entity recognition (NER) finds spans of text that name people, places, organizations and other things, and labels their type. Systems trained on news do well because entities repeat between training and test. Social media breaks this: text is short, spelling and capitalization are unreliable, and new names appear constantly."),
      P("WNUT17 (Derczynski et al., 2017) was built to test exactly this. Entities in dev and test were chosen so that they do not appear in the training tweets, and the best of seven shared-task systems reached only 41.86 entity F1. The assignment asks for three classic model families to be implemented and compared on one dataset: a CRF, an SVM and a Bi-LSTM CRF. Because no Indonesian dataset was assigned to our group, all work uses the English WNUT17 data."),
      P("<b>Contributions.</b> (1) A single evaluation protocol for the three models, with full-test-set scoring, strict entity F1 and the shared-task surface F1. (2) Hyperparameter search, feature and design ablations, bootstrap intervals and cross-validation. (3) An error analysis with a confusion matrix per model and examples of correct and incorrect predictions. (4) A complete list of deviations from the Stage 1 plan, including three preprocessing bugs that we found and fixed, with their measured effect.")]

# ---- 2 Related work
s += [P("2. Related Work", h1),
      P("<b>NER on social media.</b> Ritter et al. (2011) showed that tools trained on news lose much accuracy on tweets and built a tweet-specific tagger. The WNUT17 shared task (Derczynski et al., 2017) shifted the focus to <i>novel and emerging</i> entities: the organizers removed from dev and test every text containing an entity seen in training. Seven teams took part with systems from feature-based CRFs to neural models, and the best, UH-RiTUAL (Aguilar et al., 2017), used a multi-task neural network."),
      P("<b>Feature-based models.</b> Linear-chain CRFs (Lafferty et al., 2001) became the standard for NER with hand-built features and lexicons (McCallum and Li, 2003). Support vector machines were applied to chunking and NER by classifying each token independently over a window of features (Kudo and Matsumoto, 2001; Isozaki and Kazawa, 2002); they need a post-processing step or label features to keep tag sequences valid."),
      P("<b>Neural sequence labeling.</b> Bi-LSTM encoders with a CRF output layer (Huang et al., 2015) removed most feature engineering. Adding character-level CNNs or LSTMs (Chiu and Nichols, 2016; Lample et al., 2016; Ma and Hovy, 2016) helps with spelling variants and unseen words. Pretrained word vectors such as GloVe (Pennington et al., 2014), trained on tweets in the Twitter release, give the model knowledge of words absent from the labeled data. Contextual encoders such as BERT (Devlin et al., 2019), BERTweet (Nguyen et al., 2020) and contextual string embeddings (Akbik et al., 2018) go further and are the current standard, but they are outside this assignment, which fixes the three model families.")]

# ---- 3 Methodology
s += [P("3. Methodology", h1), P("3.1 Data and protocol", h2),
      P("We use the three official files of github.com/leondz/emerging_entities_17 without re-splitting: <b>wnut17train.conll</b> (Twitter 2010), <b>emerging.dev.conll</b> (YouTube comments) and <b>emerging.test.annotated</b> (Reddit, StackExchange, Twitter 2017). The test set is the <b>entire</b> gold file; the raw multi-annotator file <i>emerging.test.conll</i> is not used. Labels are IOB2 with six types (13 labels).")]
rows = [["", "Train", "Dev", "Test"], ["Sequences", "3,394", "1,009", "1,287"], ["Tokens", "62,730", "15,733", "23,394"], ["Entities", "1,975", "836", "1,079"], ["Sequences with an entity", "36.2%", "62.2%", "53.5%"], ["Tokens labeled O", "95.0%", "92.1%", "92.6%"],
        ["Entities unseen in train (mention level)", "-", "96.3%", "97.8%"], ["Entity tokens out of vocabulary (vs train)", "-", "56.0%", "58.3%"]]
s += tbl("Dataset statistics recomputed from the released files (all Stage 1 counts reproduce).", rows, [7.4, 3.2, 3.2, 3.2])
s += [P("The protocol has four steps: (1) tune on dev only, using entity F1; (2) retrain the chosen configuration on train plus dev (4,403 sequences); (3) score it <b>once</b> on the test file; (4) report train-only scores for reference. Metrics are strict entity-level precision, recall and F1 (seqeval, IOB2: span and type must match exactly), the shared-task surface F1 (unique lowercased surface form and type pairs), and token accuracy as a reference. Uncertainty is described with 95% bootstrap intervals (2,000 resamples of test sentences), five-fold document-grouped cross-validation on train, and three seeds for the neural model (seed 42 was fixed in Stage 1; seeds 43 and 44 measure the variance)."),
      P("3.2 Preprocessing", h2)]
s += B(["<b>Reading.</b> A blank line ends a document; in the train file, 2,394 tab-only lines end a sentence inside a tweet (3,394 train sequences from 1,000 tweets). Tokens are used exactly as released, so labels stay aligned. A BIO check finds 0 violations.",
        "<b>Normalized copy of each token</b> (for embedding lookup and lexical features; the original token is kept for shape features): HTML unescape, NFKC, URL (including bare domains such as <i>amazon.com</i>) to &lt;URL&gt;, @user to &lt;USER&gt;, hashtag body without #, every digit to 0, a repeated character capped at three (digit runs are not capped), lowercase. No stemming, lemmatization or stop-word removal.",
        f"<b>Embeddings.</b> GloVe Twitter 200d is looked up with ordered keys (lowercase token, normalized form, special &lt;url&gt;, &lt;user&gt;, &lt;number&gt; keys, hashtag body). Coverage is {cov('train', 0)}% of train tokens and {cov('test', 0)}% of test tokens ({cov('test', 1)}% of test entity tokens); tokens without a vector get zeros.",
        "<b>CRF and SVM features</b> (window +-2, Appendix A): lexical, context words and bigrams, prefixes and suffixes (1 to 4 characters), shape, social-media flags, NLTK POS tags, and K-Means cluster IDs (k = 500) of GloVe vectors. The SVM adds the token's GloVe vector (scaled by 0.3) to a sparse matrix in which features seen fewer than three times are dropped.",
        "<b>Bi-LSTM CRF inputs:</b> the frozen GloVe vector, a character-CNN vector over the first 20 characters, and a 10-class capitalization embedding."])
s += [P("3.3 Models", h2),
      P("<b>CRF.</b> A linear-chain CRF (sklearn-crfsuite, L-BFGS) with the feature dictionaries above; it scores whole label sequences and learns which transitions are valid. <b>SVM.</b> A one-vs-rest LinearSVC that labels each token independently, with balanced class weights because 92 to 95% of tokens are O. Its output can contain an I- tag with no B- before it. <b>Bi-LSTM CRF.</b> A one-layer bidirectional LSTM reads the 258-dimensional token representation; a linear layer produces label scores and a CRF layer decodes the best valid sequence with Viterbi. It is implemented in JAX and optax because PyTorch could not be installed in the working environment; the CRF log-partition and Viterbi path match brute-force enumeration on a small case.")]
s += [fig("f_architecture", 6.2, "Bi-LSTM CRF architecture.")]
rows = [["", "CRF", "SVM", "Bi-LSTM CRF"], ["Key settings", f"L-BFGS, c1 = {C2['best']['c1']}, c2 = {C2['best']['c2']}, 300 iterations", "LinearSVC, C = 0.03, balanced weights, 100 iterations", "128 units per direction, dropout 0.5, Adam 0.001, batch 32"],
        ["Input", "feature dictionaries", "sparse features (28,436 on train) + GloVe vector", "GloVe 200d (frozen), char-CNN (50 filters), capitalization (8d)"], ["Decoding", "Viterbi", "independent per token", "Viterbi"], ["Final training", "train + dev", "train + dev", f"train + dev, {bi['epochs']} epochs; seed 42 reported, seeds 43 and 44 for variance"]]
s += tbl("The three models as used for the final results.", rows, [2.4, 4.9, 4.9, 4.8], align=["l", "l", "l", "l"], boldcol0=True)

# ---- 4 Experiments and results
s += [P("4. Experiments and Results", h1), P("4.1 Hyperparameter search (dev set)", h2)]
rows = [["Model", "What was searched", "Selected", "Dev F1 (train only)"],
        ["CRF", "c1 in {0.1, 0.3, 1.0} x c2 in {0.003, 0.01, 0.03}, with and without cluster features (18 runs, 300 iterations). A first 3 x 3 grid (100 iterations) had its best point on the edge, so the grid was extended.", f"c1 = 0.3, c2 = 0.003, all features", f"{p1(C2['best']['dev_F1'])} (search)"],
        ["SVM", "C in {0.01, 0.03, 0.1, 0.3}; class weights; GloVe vector; BIO repair on or off", "C = 0.03, balanced, GloVe, no repair", f"{p1(svm['dev'])} (rerun)"],
        ["Bi-LSTM CRF", "Epochs by early stopping (patience 8, at most 35); BIO versus BIOES labels", f"BIO, {bi['best_ep_bio']} epochs", f"{p1(bi['dev_bio'])} (rerun)"]]
s += tbl("Search space and selected settings. Full grids are in Appendix D.", rows, [2.2, 8.6, 3.5, 2.7], align=["l", "l", "l", "c"], boldcol0=True)
s += [P("The Bi-LSTM CRF was tuned only through the epoch count and the label scheme; its learning rate, dropout and hidden size were fixed (Section 5).", note), P("4.2 Main results", h2)]
rows = [["Model", "Precision", "Recall", "F1 entity", "F1 surface", "Token acc."], ["CRF", p1(crf["P"]), p1(crf["R"]), p1(crf["F1"]), p1(crf["surf"]), p1(crf["tokacc"])], ["SVM", p1(svm["P"]), p1(svm["R"]), p1(svm["F1"]), p1(svm["surf"]), p1(svm["tokacc"])],
        *BI_ROWS, ["Reference: predict O everywhere", "-", "-", "-", "-", "92.6"], ["Reference: best shared-task system", "-", "-", "41.86", "40.24", "-"]]
s += tbl("Test results in percent (train + dev models, scored once on the full gold test file).", rows, [6.4, 2.0, 1.8, 2.2, 2.4, 2.2], hi=(3,))
s += [P(f"The Bi-LSTM CRF is clearly best at {p1(bi['F1'])} F1 ({bi['primary']}; the three seeds give {', '.join(p1(v) for v in bi['seeds'])}, mean {p1(bi['mean3']['F1'])}), followed by the SVM ({p1(svm['F1'])}) and the CRF ({p1(crf['F1'])}). The CRF is the most cautious (precision {p1(crf['P'])}, recall {p1(crf['R'])}), the SVM trades precision for recall (precision {p1(svm['P'])}, recall {p1(svm['R'])}), and the Bi-LSTM CRF is better than both on both measures. Predicting O everywhere already gives 92.6% token accuracy, so the CRF and SVM barely exceed it ({p1(crf['tokacc'])} and {p1(svm['tokacc'])}); entity F1 is the meaningful measure."),
      fig("f_main", 5.6, "Precision, recall and F1 on the test set."),
      P("4.3 How reliable are the differences?", h2),
      P(f"Resampling the test sentences gives 95% intervals of about 3 F1 points per model (CRF {BS['CRF'][0]} to {BS['CRF'][1]}, SVM {BS['SVM'][0]} to {BS['SVM'][1]}, Bi-LSTM CRF seed 42 {BS['Bi-LSTM CRF'][0]} to {BS['Bi-LSTM CRF'][1]}). The Bi-LSTM CRF lead over the SVM (+{BS['Bi-LSTM CRF minus SVM'][0]}, interval {BS['Bi-LSTM CRF minus SVM'][1]} to {BS['Bi-LSTM CRF minus SVM'][2]}) is far outside the noise; the SVM lead over the CRF (+{BS['SVM minus CRF'][0]}, interval {BS['SVM minus CRF'][1]} to {BS['SVM minus CRF'][2]}) is positive but small. Five-fold cross-validation on train (document-grouped, final settings) gives {p1(crf['cv']['mean'])} +- {p1(crf['cv']['std'])} for the CRF and {p1(svm['cv']['mean'])} +- {p1(svm['cv']['std'])} for the SVM. These scores come from one domain (Twitter 2010) with shared entities across folds, so they show that the settings are stable, not what to expect on new domains. They were computed before the preprocessing fix (Section 4.8).")]
rows = [["Type", "Support", "CRF", "SVM", "Bi-LSTM CRF"]] + [[t, str(SUP[t]), p1(crf["types"][t]), p1(svm["types"][t]), p1(bi["types"][t])] for t in T6]
s += [P("4.4 Results by entity type", h2)] + tbl(f"F1 by entity type on the test set (Bi-LSTM CRF: {bi['primary']}).", rows, [4.2, 2.6, 3.0, 3.0, 4.2])
s += [P(f"Person and location are easiest for every model. Corporation, product, creative-work and group reach at most {p1(WEAK)} F1 for all three (the CRF and SVM stay between 8 and 16). These types have few training examples and overlapping definitions, and creative-work titles are long and free-form."),
      P("4.5 Ablations", h2),
      P(f"<b>CRF features</b> (cumulative, dev, Figure 3 left): word identity alone gives {p1(ab[0]['F1'])} F1; shape, affixes and social-media flags add {p1(dab(1))}; context words and bigrams add {p1(dab(2))}; POS tags add {p1(dab(3))}; cluster IDs add {p1(dab(4))} in the final setting but {clu_flip:+.1f} at c1 = 0.3, c2 = 0.03, so their effect is small and unstable. <b>SVM</b> (dev, Figure 3 right, before the fix): removing class weights raises precision to {p1(svnw['P'])} but recall falls to {p1(svnw['R'])} and F1 from {p1(svf['F1'])} to {p1(svnw['F1'])}; removing the GloVe vector drops F1 to {p1(svng['F1'])}. <b>BIO repair</b> (planned in Stage 1) lowered F1 on dev ({p1(svm['dev_repair'])} with repair against {p1(svm['dev'])} without, after the fix) and on test ({p1(svm_rep_test)} against {p1(svm_norep_test)}): the {svm['invalid_I']} orphan I- tags in the raw output are ignored by strict scoring, but repair turns them into new, mostly wrong entities. Repair was dropped, a decision taken on dev. <b>BIO versus BIOES</b> (train only, one seed, before the fix): test F1 {p1(PRE['bilstm']['trainonly_bio'])} against {p1(PRE['bilstm']['bioes_test'])}. After the fix, the BIO run alone gives {p1(bi['trainonly_bio'])}, which equals the BIOES score, so the earlier gap was run-to-run noise and BIO is kept."),
      fig("f_ablation", 6.2, "Left: cumulative CRF feature ablation on dev. Right: SVM ablations on dev (before the preprocessing fix); the amber bar is the setting used to select C.")]
s += [P("4.6 Error analysis and confusion matrices", h2),
      P("Each gold test entity falls in one category: found exactly, found with a wrong boundary or type, or missed. Predictions overlapping no gold entity are spurious.")]
rows = [["Category", "CRF", "SVM", "Bi-LSTM CRF"], ["Correct (exact span and type)"] + [str(E[m]["benar"]) for m in M3], ["Wrong boundary or type"] + [str(wrong(E[m])) for m in M3], ["Missed entirely"] + [str(E[m]["terlewat"]) for m in M3], ["Spurious predictions"] + [str(E[m]["prediksi_palsu"]) for m in M3]]
s += tbl("Outcomes for the 1,079 gold test entities (Bi-LSTM CRF: seed 42).", rows, [7.0, 3.2, 3.2, 3.6])
s += [P(f"The CRF misses {E['CRF']['terlewat']} entities ({100 * E['CRF']['terlewat'] / 1079:.0f}%) and gets {E['CRF']['benar']} exactly right. The SVM gets {E['SVM']['benar']} right, but {wrong(E['SVM'])} more are found with the wrong boundary or type ({E['SVM']['salah_tipe']} with the right span but the wrong type) and it adds {E['SVM']['prediksi_palsu']} spurious predictions. The Bi-LSTM CRF gets the most right ({E['Bi-LSTM CRF']['benar']}), adds {E['Bi-LSTM CRF']['prediksi_palsu']} spurious predictions and misses {E['Bi-LSTM CRF']['terlewat']} ({100 * E['Bi-LSTM CRF']['terlewat'] / 1079:.0f}%). The confusion matrices (Figure 4) show that misses dominate the CRF and Bi-LSTM CRF, while the SVM confuses types and boundaries more often; for all models the four weak types are found less often than person and location. Recall was also split by entity properties (Figure 5)."),
      fig("f_confusion", 6.6, "Entity-level confusion matrix for each model. Rows: gold type, and SPURIOUS (predicted spans that overlap no gold entity). Columns: predicted type (exact span), wrong boundary, missed."),
      fig("f_analysis", 6.3, "Recall of gold entities by novelty, capitalization and length (group sizes in brackets)."),
      P(f"Only 109 of 1,079 test entities were seen in train or dev; recall on them is {p1(G['CRF']['seen']['recall'])} (CRF), {p1(G['SVM']['seen']['recall'])} (SVM) and {p1(G['Bi-LSTM CRF']['seen']['recall'])} (Bi-LSTM CRF), against {p1(G['CRF']['unseen']['recall'])}, {p1(G['SVM']['unseen']['recall'])} and {p1(G['Bi-LSTM CRF']['unseen']['recall'])} on the 970 unseen ones. Entities with some lowercase token reach only {p1(min(LC))} to {p1(max(LC))}. On entities of two or more tokens the Bi-LSTM CRF recovers {p1(G['Bi-LSTM CRF']['>=2 token']['recall'])} against {p1(G['CRF']['>=2 token']['recall'])} and {p1(G['SVM']['>=2 token']['recall'])}."),
      P("4.7 Examples of correct and incorrect predictions", h2)]
s += tbl("One correct and one incorrect test sentence per model (more in Appendix E).", ex_rows(1), [2.1, 1.6, 5.4, 3.9, 4.0], align=["l", "l", "l", "l", "l"], boldcol0=True)
# before/after
pm = {m: PRE[key[m]] for m in M3}; pm["Bi-LSTM CRF"] = bi["pre"]
rows = [["Model", "F1 before", "F1 after", "Change", "Precision before / after", "Recall before / after"]]
for m in M3: rows.append([m, p1(pm[m]["F1"]), p1(S[key[m]]["F1"]), f"{S[key[m]]['F1'] - pm[m]['F1']:+.1f}", f"{p1(pm[m]['P'])} / {p1(S[key[m]]['P'])}", f"{p1(pm[m]['R'])} / {p1(S[key[m]]['R'])}"])
s += [P("4.8 Effect of the preprocessing fix", h2),
      P("After the first full run, a check against the Stage 1 slides found three bugs in the normalization step: (1) digit runs of four or more were collapsed to three zeros (2017 became 000, not 0000); (2) bare domains such as amazon.com were not mapped to &lt;URL&gt; as planned; (3) the # sign of hashtags was kept in the normalized form. They change 1,111 train tokens (1.8%, of which 806 hashtags, 298 long digit runs and 7 domains), 16 dev tokens and 50 test tokens (10 of them entity tokens). The final models were retrained with unchanged hyperparameters; the search grids, ablations and cross-validation were not repeated.")]
s += tbl(f"Final test results before and after the fix (Bi-LSTM CRF: {bi['primary']}).", rows, [3.6, 2.2, 2.2, 2.0, 3.6, 3.4], hi=())
s += [P("The changes are small (all under one F1 point in absolute terms for the final numbers reported here), and the ranking is unchanged. Results in this report are the post-fix numbers unless a caption says otherwise.", note)]

# ---- 5 Discussion
s += [P("5. Discussion", h1),
      P("<b>Key findings.</b> (1) Representation dominates. The SVM and Bi-LSTM CRF see dense GloVe vectors; the CRF sees GloVe only through 500 cluster IDs and has clearly lower recall on unseen entities. In the SVM, the dense vector is worth "
        f"{p1(svf['F1'] - svng['F1'])} dev F1 points. This fits the idea that dense representations matter most when 98% of entities are new. We did not test it directly (no Bi-LSTM CRF without GloVe and no CRF with dense vectors in our own runs), so it remains a hypothesis; Appendix F shows a from-scratch baseline that points the same way. (2) Sequence decoding helps precision and multi-word names; class weighting helps recall but adds spurious and fragmented spans. (3) The hard cases are entities unseen in training, entities with lowercase tokens, and the types corporation, product, creative-work and group."),
      P(f"<b>Comparison with the shared task.</b> The Bi-LSTM CRF ({p1(bi['F1'])}) is on par with the best participant (41.86; the three seeds range from {p1(min(bi['seeds']))} to {p1(max(bi['seeds']))}, so \"better\" would overclaim); the SVM ({p1(svm['F1'])}) and CRF ({p1(crf['F1'])}) sit in the lower half of the reported range (26.30 to 41.86). The comparison is approximate: the released test set holds 1,079 entities after annotation fixes (the paper counts 1,040), our surface F1 lowercases strings, and our final models train on dev too."),
      P("<b>Limitations.</b> The CRF and SVM were run once (deterministic; uncertainty is shown with bootstrap intervals). The Bi-LSTM CRF was tuned only through epochs and label scheme, and there is no run without GloVe. Ablations, search grids and cross-validation predate the preprocessing fix. The SVM uses only the token's own GloVe vector, prunes rare features and was capped at 100 iterations. The GloVe vector set includes words from all splits (frozen vectors, no labels, nothing trained on test). The neural model is written in JAX; numbers from a PyTorch port may differ slightly."),
      P("<b>Possible improvements.</b> Give unknown words a trained vector (replace rare training words by an unknown token during training); tune the Bi-LSTM CRF (learning rate, dropout, hidden size); add the remaining Stage 1 features (shape for context words, is_lower and similar flags, context-window GloVe average, label history for the SVM); repeat BIOES with several seeds; and try contextual encoders such as BERTweet.")]

# ---- 6 Conclusion
s += [P("6. Conclusion and Future Work", h1),
      P(f"We implemented a CRF, an SVM and a Bi-LSTM CRF on WNUT17 under one protocol with full test-set scoring. The Bi-LSTM CRF with frozen GloVe Twitter vectors, character features and capitalization embeddings reaches {p1(bi['F1'])} entity F1, on par with the best shared-task system, while the SVM ({p1(svm['F1'])}) and CRF ({p1(crf['F1'])}) fall well below it. Entities that never appeared in training, entities with lowercase tokens, and the types corporation, product, creative-work and group remain the hard cases. Future work should test the role of pretrained embeddings directly, tune the neural model, complete the Stage 1 feature list, and move to contextual encoders.")]

# ---- references
s += [P("References", h1)]
refs = ["Aguilar, G., Maharjan, S., Lopez Monroy, A. P. and Solorio, T. (2017). A multi-task approach for named entity recognition in social media data. Proc. 3rd Workshop on Noisy User-generated Text, 148-153.",
        "Akbik, A., Blythe, D. and Vollgraf, R. (2018). Contextual string embeddings for sequence labeling. COLING, 1638-1649.",
        "Bird, S., Klein, E. and Loper, E. (2009). Natural Language Processing with Python. O'Reilly.",
        "Bradbury, J. et al. (2018). JAX: composable transformations of Python+NumPy programs. github.com/google/jax.",
        "Chiu, J. and Nichols, E. (2016). Named entity recognition with bidirectional LSTM-CNNs. TACL, 4, 357-370.",
        "Cortes, C. and Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.",
        "Derczynski, L., Nichols, E., van Erp, M. and Limsopatham, N. (2017). Results of the WNUT2017 shared task on novel and emerging entity recognition. Proc. 3rd Workshop on Noisy User-generated Text, 140-147.",
        "Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: pre-training of deep bidirectional transformers for language understanding. NAACL-HLT, 4171-4186.",
        "Efron, B. and Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman and Hall.",
        "Huang, Z., Xu, W. and Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. arXiv:1508.01991.",
        "Isozaki, H. and Kazawa, H. (2002). Efficient support vector classifiers for named entity recognition. COLING, 1-7.",
        "Kingma, D. and Ba, J. (2015). Adam: a method for stochastic optimization. ICLR.",
        "Kudo, T. and Matsumoto, Y. (2001). Chunking with support vector machines. NAACL.",
        "Lafferty, J., McCallum, A. and Pereira, F. (2001). Conditional random fields: probabilistic models for segmenting and labeling sequence data. ICML, 282-289.",
        "Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K. and Dyer, C. (2016). Neural architectures for named entity recognition. NAACL-HLT, 260-270.",
        "Ma, X. and Hovy, E. (2016). End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. ACL, 1064-1074.",
        "McCallum, A. and Li, W. (2003). Early results for named entity recognition with conditional random fields, feature induction and web-enhanced lexicons. CoNLL, 188-191.",
        "Nakayama, H. (2018). seqeval: a Python framework for sequence labeling evaluation. github.com/chakki-works/seqeval.",
        "Nguyen, D. Q., Vu, T. and Nguyen, A. T. (2020). BERTweet: a pre-trained language model for English tweets. EMNLP (System Demonstrations), 9-14.",
        "Okazaki, N. (2007). CRFsuite: a fast implementation of conditional random fields. www.chokkan.org/software/crfsuite.",
        "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. JMLR, 12, 2825-2830.",
        "Pennington, J., Socher, R. and Manning, C. (2014). GloVe: global vectors for word representation. EMNLP, 1532-1543.",
        "Ritter, A., Clark, S., Mausam and Etzioni, O. (2011). Named entity recognition in tweets: an experimental study. EMNLP, 1524-1534."]
s += [Paragraph(f"[{i + 1}] {r}", st("ref", fontSize=8.3, leading=11.3, leftIndent=17, firstLineIndent=-17, spaceAfter=2.5)) for i, r in enumerate(refs)]

# ================================================================ appendices
s += [PageBreak(), P("Appendix A. Preprocessing in Detail", h1), P("The order of operations for every token is listed here; Table numbers continue from the main text.")]
s += B(["<b>Reading:</b> UTF-8; a blank line ends a document; a tab-only line ends a sentence (train: 2,394 such lines); malformed lines are impossible in the released files (two columns everywhere); the loader recomputes all counts in Table 1.",
        "<b>Normalization</b> (function <i>normalize</i> in prep_glove.py): 1 HTML unescape; 2 NFKC; 3 tokens starting with http://, https://, www. or ending in a domain such as .com, .net, .org, .io, .co, .ly become &lt;URL&gt;; 4 tokens starting with @ become &lt;USER&gt;; 5 a leading # is dropped; 6 every digit becomes 0; 7 a non-digit character repeated four or more times is cut to three; 8 lowercase.",
        f"<b>GloVe lookup keys</b> (function <i>glove_keys</i>): the lowercase original token; the normalized form; &lt;url&gt; or &lt;user&gt; for those classes; &lt;number&gt; when the normalized form is only zeros (optionally with one . or ,); the hashtag body. {nkeys:,} distinct keys are needed by the three splits and 11,577 exist in GloVe Twitter 200d. Token coverage: train {cov('train', 0)}%, dev {cov('dev', 0)}%, test {cov('test', 0)}%; entity tokens: {cov('train', 1)}%, {cov('dev', 1)}%, {cov('test', 1)}%.",
        "<b>POS tags:</b> NLTK averaged perceptron tagger (Penn Treebank tags), cached per split. <b>Clusters:</b> MiniBatchKMeans (k = 500, seed 42) fitted on the vectors of the 100,000 most frequent GloVe Twitter words; each needed word gets its cluster ID, tokens without a vector get -1.",
        "<b>Capitalization.</b> The original-case token is kept for the shape, is_title and is_upper features, for the Bi-LSTM capitalization embedding and for its character CNN. Everything lexical (lowercase token, normalized form, prefixes, suffixes, context words, GloVe lookup) is lowercased, and GloVe Twitter is itself uncased. So <i>Google</i> and <i>google</i> differ only in the shape (Xx against x) and is_title features for the CRF and SVM, and in the capitalization class and characters for the Bi-LSTM CRF.",
        "<b>Bi-LSTM CRF:</b> the character vocabulary comes from the training data (92 characters on train; unknown characters share one id), words without a GloVe vector share one all-zero vector, and sequences are batched by length bucket (8, 16, 24, 32, 48, 64, 128); the longest test sequence has 105 tokens, so nothing is truncated. For BIOES, labels are converted to 25 tags for training and back to BIO before scoring."])
rows = [["Group", "Features"], ["Lexical", "lowercase token, normalized form, stop-word flag (NLTK list)"], ["Context (window +-2)", "lowercase words at -2, -1, +1, +2 (BOS/EOS markers at the edges); title-case and upper-case flags at -1 and +1; left and right bigrams"],
        ["Shape", "collapsed shape pattern (Xxxx, dd), is_title, is_upper, has_digit, has_hyphen"], ["Affixes", "prefixes and suffixes of 1 to 4 characters (lowercase), token length (capped at 15)"], ["Social media", "is_url, is_mention, is_hashtag, is_emoji"],
        ["POS tags", "tag of the token and of the tokens at -1 and +1"], ["Cluster IDs", "cluster of the GloVe vector of the token and of the tokens at -1 and +1"], ["Dense vector (SVM only)", "200-d GloVe vector of the token, scaled by 0.3, appended to the sparse matrix"]]
s += tbl("Features used by the CRF and SVM.", rows, [3.6, 13.4], align=["l", "l"], boldcol0=True)

s += [P("Appendix B. Deviations from the Stage 1 Plan and Bug-Fix Log", h1)]
rows = [["Item (Stage 1 slides 21 to 25 / report)", "Plan", "Implemented", "Status"],
        ["Digit rule (slide 21)", "2017 becomes 0000", "Was 000 (repeat cap collapsed digit runs); now 0000", "Bug fixed"],
        ["URL rule (report Table 8)", "http, www and *.com become &lt;URL&gt;", "Bare domains were missing; now included", "Bug fixed"],
        ["Hashtag rule (report Table 8)", "Drop #, process the word", "# was kept in the normalized form; now dropped", "Bug fixed"],
        ["Cross-validation (slide 24)", "5-fold by document on train", "First run used train + dev and BIO repair for the SVM; rerun on train only with final settings (CRF 34.4, SVM 43.1)", "Fixed"],
        ["SVM dev score", "Report the dev score of the final setting", "The dev F1 used to pick C was measured with BIO repair (41.4); the final setting without repair scores 44.8 before and 44.0 after the fix", "Fixed"],
        ["SVM BIO repair (slide 22)", "Apply after prediction", "Built and tested, dropped: lower F1 on dev and test", "Changed"],
        ["SVM extras (report 3.5)", "Context-window GloVe average; label history", "Token GloVe vector only (x 0.3); no label history", "Not done"],
        ["SVM feature space", "All features", "Features seen at least 3 times", "Changed (speed)"],
        ["Context features (slide 23)", "Lowercase and shape for +-2", "Lowercase +-2, title/upper flags +-1, no shape for context words", "Partly"],
        ["Other shape flags (report 3.5)", "is_lower, is_digit, has_apostrophe, is_punct, has_non_ascii, full shape", "Not included", "Not done"],
        ["Vocabulary (slide 22)", "Train vocabulary plus embedding words", "All GloVe words found in any split (frozen vectors, no labels); PAD and UNK both zero", "Changed (disclosed)"],
        ["Embeddings for Bi-LSTM (report 3.3)", "GloVe, fine-tuned", "GloVe, frozen; capitalization embedding added", "Changed"],
        ["FastText (slide 22)", "Alternative", "Not run", "Not done (optional)"],
        ["Cluster features (slide 23)", "K-Means k = 500 on GloVe", "Fitted on the 100,000 most frequent words", "Minor"],
        ["Demo tokenizer (slide 21)", "TweetTokenizer for new text", "Not built", "Not done"],
        ["BIOES (slide 22)", "Try for Bi-LSTM CRF", "One train-only run, before the fix", "Done, inconclusive"],
        ["Bi-LSTM tuning (report 3.6)", "Learning rate, dropout, hidden size", "Only epochs and label scheme", "Limitation"],
        ["Random seeds (slide 24)", "Seed 42 for all experiments", "Seed 42 everywhere, and the reported Bi-LSTM CRF run is seed 42 (fixed in Stage 1, before any result). Seeds 43 and 44 were also run and are disclosed (F1 42.1 and 40.2). CRF and SVM are deterministic given the data", "Added"],
        ["Framework", "Not specified", "JAX and optax (PyTorch could not be installed here)", "Note"]]
s += tbl("Differences between the Stage 1 plan and the implementation.", rows, [4.1, 4.3, 6.1, 2.5], align=["l", "l", "l", "l"], boldcol0=True)

s += [P("Appendix C. Chronological Log", h1)]
rows = [["#", "Step", "Outcome"],
        ["1", "Environment: Python 3.12, one CPU core (about 4 GB RAM), scikit-learn, sklearn-crfsuite, seqeval, NLTK, JAX, optax. PyTorch did not fit the free disk space.", "Toolchain fixed."],
        ["2", "Cloned WNUT17 and recomputed all Stage 1 statistics.", "All counts reproduced."], ["3", "Downloaded GloVe Twitter 200d (795 MB) and extracted the needed vectors.", "11,577 of 15,549 lookup keys found."],
        ["4", "CRF search: 3 x 3 grid, best point on the edge; extended to 18 runs with 300 iterations.", f"c1 = 0.3, c2 = 0.003, dev F1 {p1(C2['best']['dev_F1'])}."],
        ["5", "SVM: full feature space too slow; dropped rare features, capped iterations at 100; searched C; tested class weights, GloVe, BIO repair.", "C = 0.03, no repair."],
        ["6", "Bi-LSTM CRF in JAX; checked CRF and Viterbi against brute force; early stopping on dev; BIOES tried.", "BIO, 18 epochs before the fix (23 after)."], ["7", "Final models on train + dev, scored once on test; bootstrap, cross-validation, error analysis.", "First full result set."],
        ["8", "Compared with Stage 1 slides 20 to 25 and the report; found three normalization bugs; reran cross-validation on train only.", "Bugs listed in Appendix B."],
        ["9", "Fixed the bugs, regenerated GloVe keys and cluster IDs, retrained the final CRF, SVM and Bi-LSTM CRF (3 seeds) with unchanged hyperparameters.", "Results in Section 4 are post-fix."]]
s += tbl("Work log.", rows, [0.8, 11.6, 4.6], align=["c", "l", "l"])

s += [P("Appendix D. Full Tuning Grids (Dev Set, Before the Fix)", h1)]
rows = [["Features", "c1", "c2", "Precision", "Recall", "F1 entity", "F1 surface"]]
for g in sorted(C2["grid"], key=lambda z: (z["fitur"] != "lengkap", z["c1"], z["c2"])): rows.append(["full" if g["fitur"] == "lengkap" else "no cluster", g["c1"], g["c2"], p1(g["P"]), p1(g["R"]), p1(g["F1"]), p1(g["surfF1"])])
bi_ = [i for i, r in enumerate(rows) if r[0] == "full" and r[1] == C2["best"]["c1"] and r[2] == C2["best"]["c2"]]
s += tbl("CRF search (train only, 300 iterations). The selected row is shaded.", rows, [3.0, 1.8, 1.8, 2.6, 2.4, 2.8, 2.6], hi=tuple(bi_))
d2 = {x["C"]: x for x in SV2["dev"]}; rows = [["C", "Dev F1 with BIO repair", "Dev F1 without BIO repair"]]
for gg in sorted(SV["grid"] + [{"C": 0.01, "F1": SV2["dev"][0]["repair"]["F1"]}], key=lambda z: z["C"]): rows.append([gg["C"], p1(gg["F1"]), p1(d2[gg["C"]]["no_repair"]["F1"]) if gg["C"] in d2 else "not run"])
s += tbl("SVM search (train only). C = 0.03 was selected with BIO repair on.", rows, [3.0, 6.0, 6.0], hi=(2,))
rows = [["Statistic", "CRF", "SVM"], ["Cross-validation folds", ", ".join(p1(v) for v in crf["cv"]["folds"]), ", ".join(p1(v) for v in svm["cv"]["folds"])], ["Mean and SD", f"{p1(crf['cv']['mean'])} +- {p1(crf['cv']['std'])}", f"{p1(svm['cv']['mean'])} +- {p1(svm['cv']['std'])}"],
        ["Dev F1 (train only, post-fix)", p1(crf["dev"]), f"{p1(svm['dev'])} (repair: {p1(svm['dev_repair'])})"], ["Test F1 (train only, post-fix)", p1(crf["trainonly"]), p1(svm["trainonly"])], ["Test F1 (train + dev, post-fix)", p1(crf["F1"]), p1(svm["F1"])]]
s += tbl("Cross-validation (train only, before the fix) and train-only scores.", rows, [5.0, 6.0, 6.0], align=["l", "l", "l"])
s += [fig("f_curves", 6.0, "Bi-LSTM CRF dev F1 per epoch (train only). BIOES and the earlier BIO curve are from before the preprocessing fix.")]

s += [P("Appendix E. More Examples of Correct and Incorrect Predictions", h1)]
s += tbl("Two correct and two incorrect test sentences per model (chosen automatically: most gold entities first, then shortest).", ex_rows(2), [2.1, 1.6, 5.4, 3.9, 4.0], align=["l", "l", "l", "l", "l"], boldcol0=True)

s += [P("Appendix F. Baseline Without Pretrained Embeddings (Group's PyTorch Implementation)", h1),
      P("Another group implementation (PyTorch and torchcrf) trains its word embeddings from scratch and uses no GloVe or cluster features. It follows the same protocol (official splits, tune on dev, retrain on train + dev). Its published test scores are shown next to ours. The two are not a controlled comparison: features, tuning grids, hidden sizes, batch sizes and the SVM label-history feature all differ.")]
rows = [["Model", "From-scratch baseline: P / R / F1 / surface F1", "This report: P / R / F1 / surface F1", "Main differences"],
        ["CRF", "43.1 / 19.6 / 26.9 / 25.3", f"{p1(crf['P'])} / {p1(crf['R'])} / {p1(crf['F1'])} / {p1(crf['surf'])}", "Baseline: richer hand features, no cluster IDs; 2 x 2 grid"],
        ["SVM", "31.6 / 25.9 / 28.4 / 27.1", f"{p1(svm['P'])} / {p1(svm['R'])} / {p1(svm['F1'])} / {p1(svm['surf'])}", "Baseline: label history and BIO repair, no GloVe; ours: GloVe vector, no repair"],
        ["Bi-LSTM CRF", "34.6 / 21.1 / 26.2 / 24.9", f"{p1(bi['P'])} / {p1(bi['R'])} / {p1(bi['F1'])} / {p1(bi['surf'])}", "Baseline: embeddings from scratch (100d), hidden 200; ours: frozen GloVe 200d, char-CNN, capitalization"]]
s += tbl("Test scores in percent (train + dev models).", rows, [2.2, 4.6, 4.6, 5.6], align=["l", "l", "l", "l"], boldcol0=True)
s += [P("The largest gap is for the Bi-LSTM CRF (about 15 F1 points), the model whose input changes most. This supports, without proving, the view that pretrained Twitter vectors are the main reason for our ranking.", note)]

DM = R("domain_mixing_crf.json"); TA = R("target_domain_adaptation_crf.json")
s += [P("Appendix G. Training-Domain Mixing (CRF)", h1),
      P("The final models train on train (Twitter 2010) plus dev (YouTube). To see how much mixing sources helps, and why the test file must not be mixed in, the final CRF configuration was trained on other data mixes (no re-tuning).")]
rows = [["Training data", "Scored on", "F1 entity", "Comment"],
        ["train only (Twitter 2010)", "official test", p1(DM["train only -> test (official)"]), "3,394 sequences"],
        ["dev only (YouTube)", "official test", p1(DM["dev only -> test"]), "1,009 sequences, yet better than train alone: a closer domain matters"],
        ["train + dev", "official test", p1(DM["train + dev -> test (official)"]), "final protocol: mixing two sources helps most"],
        ["all three files pooled, random 85/15 split by document", "15% held-out part of the pool", p1(DM["pooled random split (85/15, by document)"]), f"{DM['pooled: held-out entity mentions already seen in training (%)']}% of held-out entity mentions already occur in training (official test: about 2% against train, 10% against train + dev)"],
        ["train + dev + one half of the test (two-fold, analysis only)", "the other half of the test", p1(TA["F1 with the other half of the test added to training"]), f"same sentences with the official model: {p1(TA['F1 official (train + dev)'])}; seen entity mentions {TA['entity mentions seen in training, official (%)']}% rising to {TA['entity mentions seen in training, with other half (%)']}% ({TA['sequences added per fold']} sequences added per fold)"]]
s += tbl("CRF trained on different mixes of the WNUT17 sources.", rows, [4.4, 3.2, 1.9, 7.5], align=["l", "l", "c", "l"])
s += [P("Adding a second source to training raises test F1 (16.8 to 27.7 for the CRF). Mixing the <i>test</i> file into training is not valid: the pooled split scores 42.7 because entities leak between training and evaluation (23.3% seen), which removes the emerging-entity design and breaks comparison with the shared task. The pooled evaluation part is also small (494 sequences, 373 entities) and differently composed, so the 15-point gap cannot be split between leakage and domain match; the run only shows the direction. Labels from the target domain help even more (27.7 to 35.6 on the other half of the test), partly because entities repeat inside the test file; this shows what in-domain data could do and is not a result under the official protocol. Only the CRF was tested.", note)]
s += [P("Appendix H. Reproducibility", h1),
      P("<b>Environment.</b> Python 3.12.3; numpy 2.4.4; scipy 1.17.1; scikit-learn 1.8.0; sklearn-crfsuite 0.5.0; seqeval 1.2.2; nltk 3.10.3; jax and jaxlib 0.11.2; optax 0.2.8; matplotlib 3.10.8. One CPU core, about 4 GB RAM. A Bi-LSTM CRF epoch takes 12 to 20 seconds after a first epoch of about one minute (compilation). All scripts use seed 42; extra neural runs use 43 and 44."),
      P("<b>Code (archive).</b> The notebook <i>Stage2_NER_WNUT17_Kelompok1.ipynb</i> reloads saved predictions and recomputes all tables and figures (set RUN_TRAINING = True to retrain). Modules: wnut_data.py (reader, surface F1), prep_glove.py (normalization, GloVe keys), features.py (CRF/SVM features), evalx.py (strict metrics, error categories, confusion matrix), bilstm_crf.py (JAX model). Scripts: fix_crf.py, fix_svm.py, fix_bilstm.py (final runs), domain_mixing_crf.py and target_domain_adaptation_crf.py (Appendix G), run_crf2.py, run_svm.py, run_svm2.py, run_cv_final.py (search, ablation, cross-validation, before the fix), analysis.py, make_figures.py, build_report2.py."),
      P("<b>Data.</b> WNUT17 from github.com/leondz/emerging_entities_17 (CC BY 4.0); GloVe Twitter 200d from the gensim-data release glove-twitter-200. Predictions and metrics of every run are stored as JSON in the results folder; results_prefix holds the runs before the fix.")]

doc.build(s); print("built", cnt)
