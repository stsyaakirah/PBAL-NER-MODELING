"""Report for the TWITTER-ONLY protocol (train on train only, seed 42) with the Bi-LSTM CRF improvement study.
All numbers come from results/ (to_crf.json, to_svm.json, improve_bilstm.json, analysis_to.json, bootstrap_to.json, errors_to.json)."""
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
IB, TC, TS, AN, BS, ERR, C2, SV, SV2, CV, PREV = R("improve_bilstm.json"), R("to_crf.json"), R("to_svm.json"), R("analysis_to.json")["group_recall"], R("bootstrap_to.json"), R("errors_to.json"), R("crf2.json"), R("svm.json"), R("svm2.json"), R("cv_final.json"), R("summary.json")
M4 = ["CRF", "SVM", "Bi-LSTM CRF (baseline)", "Bi-LSTM CRF (ensemble)"]; T6 = ["person", "location", "corporation", "product", "creative-work", "group"]; SUP = {"person": 429, "location": 150, "corporation": 66, "product": 127, "creative-work": 142, "group": 165}
p1 = lambda x: f"{x:.1f}"; p2 = lambda x: f"{x:.2f}"; pct = lambda x: round(100 * x, 2)
V = IB["variants"]; v0 = V[0]; VN = {v["name"][:2]: v for v in V}; fin = IB["final"]; base = IB["base"]; sw = IB["sweep"]; fv = next(v for v in V if v["name"] == IB["final_variant"]); d_fin = fin["delta"]; d_base = base["delta"]
def r(e): return {"P": pct(e["entity"]["precision"]), "R": pct(e["entity"]["recall"]), "F1": pct(e["entity"]["f1"]), "surf": pct(e["surface"]["f1"]), "tok": pct(e["token_acc"]), "types": {t: pct(e["per_type"][t]["f1-score"]) for t in T6}}
IBE = R("ensemble_bilstm.json")
MD = {"CRF": r(TC["test"]), "SVM": r(TS["test"]), "Bi-LSTM CRF (baseline)": r(base["test_nobias"]), "Bi-LSTM CRF (ensemble)": r(IBE["test_delta0"])}; SEL = r(fin["test"])
DEV = {"CRF": pct(TC["dev"]["entity"]["f1"]), "SVM": pct(TS["dev"]["entity"]["f1"]), "Bi-LSTM CRF (baseline)": v0["dev_F1"], "Bi-LSTM CRF (ensemble)": IBE["dev_delta0"]}
crf, svm, bb, bi = MD["CRF"], MD["SVM"], MD["Bi-LSTM CRF (baseline)"], MD["Bi-LSTM CRF (ensemble)"]
def sweep_at(name, d): return next(x for x in sw[name]["rows"] if x["delta"] == d)
gain_test = bi["F1"] - bb["F1"]; REF = PREV["bilstm"]["mean3"]["F1"]; REF42 = PREV["bilstm"]["seed42"]["F1"]; SEEDS = PREV["bilstm"]["seeds"]
pb = BS["Bi-LSTM CRF (ensemble) minus Bi-LSTM CRF (baseline)"]; SIG = pb[1] > 0
SG = [VN[k]["dev_F1"] - v0["dev_F1"] for k in ("V1", "V2", "V3", "V4")]; TG = [VN[k]["test"]["F1"] - v0["test"]["F1"] for k in ("V1", "V2", "V3", "V4")]
ER0 = IBE["rows"][0]; ERB = next(x for x in IBE["rows"] if x["delta"] == IBE["best_delta"]); bsel = BS["V3 + shift (dev choice) minus Bi-LSTM CRF (baseline)"]
sn = lambda k: {"V0": "baseline", "V1": "word dropout", "V2": "fine-tuned GloVe", "V3": "hidden size 256", "V4": "POS tags", "V5": "combination"}[k]
E = ERR; wrong = lambda e: e["salah_batas"] + e["salah_tipe"] + e["salah_batas_dan_tipe"]
UN = [AN[m]["unseen"]["recall"] for m in M4]; LC = [AN[m]["ada huruf kecil"]["recall"] for m in M4]
WEAK = max(MD[m]["types"][t] for m in M4 for t in ("corporation", "product", "creative-work", "group"))
ab = C2["ablation"]; dab = lambda i: ab[i]["F1"] - ab[i - 1]["F1"]; sa = {a["name"]: a for a in SV["ablation"]}; svf, svnw, svng = sa["LinearSVC balanced + GloVe (lengkap)"], sa["Tanpa pembobotan kelas"], sa["Tanpa vektor GloVe"]
grid = {(g["fitur"], g["c1"], g["c2"]): g for g in C2["grid"]}; clu_flip = grid[("lengkap", 0.3, 0.03)]["F1"] - grid[("tanpa_cluster", 0.3, 0.03)]["F1"]
gk = {}
try:
    import prep_glove
    need = pickle.load(open("glove_vocab.pkl", "rb"))["need_words"]
    for sp in ("train", "dev", "test"):
        toks = [(t, l) for x in load(sp) for t, l in zip(x["tokens"], x["labels"])]; has = lambda t: any(k in need for k in prep_glove.glove_keys(t))
        gk[sp] = (100 * sum(has(t) for t, _ in toks) / len(toks), 100 * sum(has(t) for t, l in toks if l != "O") / sum(1 for _, l in toks if l != "O"))
    nkeys = len({k for sp in ("train", "dev", "test") for x in load(sp) for t in x["tokens"] for k in prep_glove.glove_keys(t)})
except Exception as e:
    print("coverage failed", e); nkeys = 0
cov = lambda sp, i: p1(gk[sp][i]) if sp in gk else "n/a"

test = load("test"); PF = {"CRF": "pred_to_crf_test.json", "SVM": "pred_to_svm_test.json", "Bi-LSTM CRF": "pred_to_bilstm_final_test.json"}; PRED = {m: json.load(open("results/" + f)) for m, f in PF.items()}
M3 = list(PRED); spans = lambda x, lab: "; ".join(f"{' '.join(x['tokens'][b:e + 1])} = {t}" for t, b, e in get_entities(lab)) or "no entity"
exact = lambda i, m: set(get_entities(PRED[m][i])) == set(get_entities(test[i]["labels"])); short = [i for i, x in enumerate(test) if get_entities(x["labels"]) and len(x["tokens"]) <= 18]
def pick(m, ok, n):
    c = [i for i in short if exact(i, m) == ok]; c.sort(key=lambda i: (-len(get_entities(test[i]["labels"])), len(test[i]["tokens"]))); return c[:n]
def ex_rows(n):
    rows = [["Model", "Result", "Sentence", "Gold", "Prediction"]]
    for m in M3:
        for ok, tag in ((True, "correct"), (False, "incorrect")):
            for i in pick(m, ok, n): rows.append([m, tag, " ".join(test[i]["tokens"]).replace("&", "&amp;"), spans(test[i], test[i]["labels"]).replace("&", "&amp;"), spans(test[i], PRED[m][i]).replace("&", "&amp;")])
    return rows

# ---------------------------------------------------------------- styles and helpers
NAVY, GOLD, BG, LINE, INK, MUT, TINT = [colors.HexColor(c) for c in ("#12406A", "#B8960C", "#F4F5F7", "#D9DEE5", "#1F2937", "#5B6B7C", "#E6EEF6")]
def st(name, **kw):
    b = dict(fontName="Helvetica", fontSize=9.1, leading=13.0, textColor=INK, spaceAfter=4.5, alignment=TA_LEFT); b.update(kw); return ParagraphStyle(name, **b)
body = st("body"); h1 = st("h1", fontName="Helvetica-Bold", fontSize=13.5, leading=17, textColor=NAVY, spaceBefore=11, spaceAfter=5, keepWithNext=1)
h2 = st("h2", fontName="Helvetica-Bold", fontSize=10.4, leading=13.5, textColor=NAVY, spaceBefore=7, spaceAfter=2.5, keepWithNext=1)
cap = st("cap", fontName="Helvetica-Oblique", fontSize=8, leading=10.6, textColor=MUT, spaceAfter=8, spaceBefore=2); tcap = st("tcap", fontName="Helvetica-Bold", fontSize=8.4, leading=11, textColor=NAVY, spaceBefore=6, spaceAfter=2.5, keepWithNext=1)
note = st("note", fontSize=7.9, leading=10.6, textColor=MUT, spaceBefore=2, spaceAfter=6); bul = st("bul", leftIndent=12, bulletIndent=1, spaceAfter=2.2)
al = {"l": TA_LEFT, "c": TA_CENTER, "r": TA_RIGHT}
cell = {a: st("c" + a, fontSize=8.1, leading=10.4, spaceAfter=0, alignment=al[a]) for a in "lcr"}; cellb = {a: st("cb" + a, fontName="Helvetica-Bold", fontSize=8.1, leading=10.4, spaceAfter=0, alignment=al[a]) for a in "lcr"}
chdr = {a: st("ch" + a, fontName="Helvetica-Bold", fontSize=8.1, leading=10.4, textColor=colors.white, spaceAfter=0, alignment=al[a]) for a in "lcr"}
cnt = {"t": 0, "f": 0}
P = lambda t, s=body: Paragraph(t, s)
def B(items): return [Paragraph(t, bul, bulletText="\u2022") for t in items]
def table(rows, widths, align=None, hi=(), boldcol0=False):
    align = align or ["l"] + ["c"] * (len(rows[0]) - 1); data = []
    for ri, row in enumerate(rows): data.append([Paragraph(str(v), chdr[align[c]] if ri == 0 else (cellb[align[c]] if (boldcol0 and c == 0) or ri in hi else cell[align[c]])) for c, v in enumerate(row)])
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 2.6), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.6), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("BACKGROUND", (0, 0), (-1, 0), NAVY)]
    for ri in range(1, len(rows)):
        if ri in hi: sty.append(("BACKGROUND", (0, ri), (-1, ri), TINT))
        elif ri % 2 == 0: sty.append(("BACKGROUND", (0, ri), (-1, ri), BG))
    t.setStyle(TableStyle(sty)); return t
def tbl(title, rows, widths, **kw): cnt["t"] += 1; return [Paragraph(f"Table {cnt['t']}. {title}", tcap), table(rows, widths, **kw), Spacer(1, 5)]
def fig(name, width_in, caption):
    cnt["f"] += 1; w, h = PILImage.open(f"report_figs_to/{name}.png").size; W = width_in * inch; im = Image(f"report_figs_to/{name}.png", width=W, height=W * h / w); im.hAlign = "CENTER"
    return KeepTogether([im, Paragraph(f"<b>Figure {cnt['f']}.</b> {caption}", cap)])

def footer(c, doc):
    c.saveState(); c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(2 * cm, 1.65 * cm, A4[0] - 2 * cm, 1.65 * cm); c.setFont("Helvetica", 7.6); c.setFillColor(MUT)
    c.drawString(2 * cm, 1.2 * cm, "NER on WNUT17, Twitter-only training, seed 42 | Kelompok 1 | Advanced NLP, UGM"); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page)); c.restoreState()
OUT = "/mnt/user-data/outputs/NER_WNUT17_Comprehensive_Report_TwitterOnly_Seed42.pdf"
doc = BaseDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.9 * cm, bottomMargin=2.1 * cm, title="NER on WNUT17 with CRF, SVM and Bi-LSTM CRF trained on Twitter only", author="Kelompok 1, Pemrosesan Bahasa Alami Lanjut, UGM")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
W = 17.0; s = []

# ---------------------------------------------------------------- title, abstract
s += [P("ASSIGNMENT #2, STAGE 2  |  NER MODELING AND EVALUATION  |  TWITTER-ONLY TRAINING, SEED 42", st("kick", fontName="Helvetica-Bold", fontSize=8.2, textColor=GOLD, spaceAfter=5)),
      P("CRF, SVM and Bi-LSTM CRF Trained on Twitter Only (WNUT17): What Raised the F1 and What Did Not", st("title", fontName="Helvetica-Bold", fontSize=18, leading=22.5, textColor=NAVY, spaceAfter=5)),
      P("<b>Kelompok 1</b>, Pemrosesan Bahasa Alami Lanjut, Program Studi Magister Kecerdasan Artifisial, Universitas Gadjah Mada<br/>Amelia Gizzela Sheehan Auni, Aulia Gita Pratiwi, St. Syakirah, Aditya Iko Winarko, Royan Aditya", st("auth", fontSize=8.6, leading=12.4, spaceAfter=9))]
cmp_ref = f"close to the {p1(REF)} obtained earlier when YouTube data were added to training"
abstract = (f"WNUT17 tests whether named entity recognition (NER) systems can find entities they have never seen. We train a CRF, a class-weighted SVM and a Bi-LSTM CRF on the <b>Twitter training file only</b> (no YouTube or Reddit data in training), "
            f"with dev used only to choose settings and stop training, seed 42 for every run, and one final scoring on the full gold test file (1,287 sequences, 1,079 entities). "
            f"The baseline Bi-LSTM CRF (frozen GloVe Twitter vectors, character CNN, capitalization embedding) scores <b>{p1(bb['F1'])}</b> entity F1. We tried to raise it in five ways. Word dropout, fine-tuned embeddings, a larger hidden layer and POS-tag input each looked better on dev ({min(SG):+.1f} to {max(SG):+.1f} F1) but none raised the test F1 ({min(TG):+.1f} to {max(TG):+.1f}), and a precision-recall shift at decoding tuned on dev lowered it; the variant chosen by dev scored {p1(SEL['F1'])} on test. "
            f"An ensemble of the six seed-42 models, defined in advance, was the only change that rose on both dev and test: <b>{p1(bi['F1'])}</b> ({gain_test:+.1f}; paired bootstrap interval {pb[1]} to {pb[2]}), {cmp_ref}. The SVM reaches {p1(svm['F1'])} and the CRF {p1(crf['F1'])}. "
            f"Recall on unseen entities is only {p1(min(UN))} to {p1(max(UN))}%, and corporation, product, creative-work and group reach at most {p1(WEAK)} F1 for every model.")
box = Table([[P("<b>Abstract</b>", st("ah", fontName="Helvetica-Bold", fontSize=9.6, textColor=NAVY, spaceAfter=2))], [P(abstract, st("ab", fontSize=8.8, leading=12.6, spaceAfter=3))],
             [P("<b>Keywords:</b> named entity recognition, WNUT17, emerging entities, CRF, SVM, Bi-LSTM CRF, GloVe, word dropout, noisy text", st("kw", fontSize=8.1, textColor=MUT, spaceAfter=0))]], colWidths=[W * cm])
box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BG), ("LEFTPADDING", (0, 0), (-1, -1), 10), ("RIGHTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, 0), 8), ("BOTTOMPADDING", (0, -1), (-1, -1), 8)])); s += [box]

# ---------------------------------------------------------------- 1 Introduction
s += [P("1. Introduction", h1),
      P("Named entity recognition (NER) finds spans of text that name people, places, organizations and other things, and labels their type. Systems trained on news do well because entities repeat between training and test. Social media breaks this: text is short, spelling and capitalization are unreliable, and new names appear constantly."),
      P("WNUT17 (Derczynski et al., 2017) was built to test exactly this. Entities in dev and test were chosen so that they do not appear in the training tweets, and the best of seven shared-task systems reached only 41.86 entity F1. The assignment asks for a CRF, an SVM and a Bi-LSTM CRF to be implemented and compared. Because no Indonesian dataset was assigned to our group, all work uses the English WNUT17 data."),
      P(f"<b>This report differs from our earlier run in two ways.</b> First, training uses <b>Twitter data only</b>: the YouTube dev file is no longer added to the training data, so the models must generalize from tweets to YouTube, Reddit and StackExchange text. Second, only seed 42 is used, and we try to improve the best model. Because the earlier run trained on train plus dev, its {p1(REF)} F1 is not the right starting point: the comparable baseline is the Twitter-only Bi-LSTM CRF at {p1(bb['F1'])}, a gap of {p1(REF - bb['F1'])} points to close."),
      P("<b>Contributions.</b> (1) One evaluation protocol for three models, with full-test-set scoring and strict entity F1. (2) A step-by-step attempt to raise the Bi-LSTM CRF F1, with every choice made on dev, including the changes that did not carry over to test. (3) Bootstrap intervals, ablations, an error analysis with a confusion matrix per model, and examples. (4) A complete list of deviations from the Stage 1 plan, including preprocessing bugs found and fixed.")]

# ---------------------------------------------------------------- 2 Related work
s += [P("2. Related Work", h1),
      P("<b>NER on social media.</b> Ritter et al. (2011) showed that tools trained on news lose much accuracy on tweets and built a tweet-specific tagger. The WNUT17 shared task (Derczynski et al., 2017) focused on <i>novel and emerging</i> entities: the organizers removed from dev and test every text containing an entity seen in training. Seven teams took part, from feature-based CRFs to neural models, and the best, UH-RiTUAL (Aguilar et al., 2017), used a multi-task neural network."),
      P("<b>Feature-based models.</b> Linear-chain CRFs (Lafferty et al., 2001) became the standard for NER with hand-built features and lexicons (McCallum and Li, 2003). Support vector machines were applied to chunking and NER by classifying each token independently over a window of features (Kudo and Matsumoto, 2001; Isozaki and Kazawa, 2002)."),
      P("<b>Neural sequence labeling.</b> Bi-LSTM encoders with a CRF output layer (Huang et al., 2015) removed most feature engineering; character-level CNNs or LSTMs (Chiu and Nichols, 2016; Lample et al., 2016; Ma and Hovy, 2016) help with spelling variants and unseen words. Pretrained vectors such as GloVe (Pennington et al., 2014), trained on tweets in the Twitter release, give the model knowledge of words absent from the labeled data. Dropout (Srivastava et al., 2014) is the standard regularizer, and replacing rare training words by an unknown token (Lample et al., 2016) teaches a model to cope with words it has not seen; our word dropout follows the same idea. Contextual encoders such as BERT (Devlin et al., 2019), BERTweet (Nguyen et al., 2020) and contextual string embeddings (Akbik et al., 2018) go further but are outside this assignment, which fixes the three model families.")]

# ---------------------------------------------------------------- 3 Methodology
s += [P("3. Methodology", h1), P("3.1 Data and protocol (Twitter-only training)", h2),
      P("We use the three official files of github.com/leondz/emerging_entities_17 without re-splitting: <b>wnut17train.conll</b> (Twitter 2010), <b>emerging.dev.conll</b> (YouTube comments) and <b>emerging.test.annotated</b> (Reddit, StackExchange, Twitter 2017). The test set is the <b>entire</b> gold file; the raw file <i>emerging.test.conll</i> is not used. Labels are IOB2 with six types (13 labels).")]
rows = [["", "Train (Twitter)", "Dev (YouTube)", "Test (Reddit, SE, Twitter 2017)"], ["Role in this report", "the only training data", "choose settings, stop training", "final scoring, once"], ["Sequences", "3,394", "1,009", "1,287"], ["Tokens", "62,730", "15,733", "23,394"], ["Entities", "1,975", "836", "1,079"],
        ["Tokens labeled O", "95.0%", "92.1%", "92.6%"], ["Entities unseen in train (mention level)", "-", "96.3%", "97.8%"], ["Entity tokens out of vocabulary (vs train)", "-", "56.0%", "58.3%"]]
s += tbl("Dataset statistics recomputed from the released files (all Stage 1 counts reproduce).", rows, [6.0, 3.4, 3.6, 4.0])
s += [P("<b>Protocol.</b> (1) Train every model on the train file only; dev is used to select settings and to stop training, never to update weights. (2) Choose settings and improvements by dev entity F1 only. (3) Score the chosen models on the test file. Test scores of the intermediate improvement steps are printed for transparency but were never used to choose. (4) Seed 42 for every run. Metrics: strict entity precision, recall and F1 (seqeval, IOB2), the shared-task surface F1 and token accuracy. Uncertainty: 95% bootstrap intervals from 2,000 resamples of test sentences."),
      P("3.2 Preprocessing", h2)]
s += B(["<b>Reading.</b> A blank line ends a document; in train, 2,394 tab-only lines end a sentence inside a tweet (3,394 sequences from 1,000 tweets). Tokens are used as released, so labels stay aligned; the BIO check finds 0 violations.",
        "<b>Normalized copy of each token</b> (for lookup and lexical features; the original token is kept for shape features): HTML unescape, NFKC, URL (including bare domains) to &lt;URL&gt;, @user to &lt;USER&gt;, hashtag body without #, every digit to 0, repeated characters capped at three (digit runs are not capped), lowercase. No stemming, lemmatization or stop-word removal.",
        f"<b>GloVe Twitter 200d</b> lookup with ordered keys (lowercase token, normalized form, special keys, hashtag body). Coverage: {cov('train', 0)}% of train tokens, {cov('test', 0)}% of test tokens; for entity tokens {cov('train', 1)}% in train but only {cov('test', 1)}% in test. Tokens without a vector get zeros.",
        "<b>CRF and SVM features</b> (window +-2): lexical, context words and bigrams, prefixes and suffixes, shape, social-media flags, NLTK POS tags, K-Means cluster IDs (k = 500) of GloVe vectors; the SVM also gets the token's GloVe vector (x 0.3) in a sparse matrix that drops features seen fewer than three times. Capitalization: shape and is_title features keep <i>Google</i> and <i>google</i> apart; everything lexical is lowercased.",
        "<b>Bi-LSTM CRF inputs:</b> frozen GloVe vector, a character-CNN vector (first 20 characters) and a 10-class capitalization embedding."])
s += [P("3.3 Models", h2),
      P("<b>CRF</b>: linear-chain CRF (sklearn-crfsuite, L-BFGS, c1 = 0.3, c2 = 0.003, 300 iterations). <b>SVM</b>: one-vs-rest LinearSVC, C = 0.03, balanced class weights, no BIO repair. <b>Bi-LSTM CRF baseline</b>: one bidirectional LSTM layer (128 units per direction) over the 258-dimensional token representation, a linear layer and a CRF layer decoded with Viterbi; dropout 0.5, Adam 0.001, batch 32, early stopping on dev (patience 8, at most 35 epochs). It is written in JAX and optax because PyTorch could not be installed in the working environment; the CRF and Viterbi were checked against brute-force enumeration."),
      fig("f_architecture", 5.8, "Bi-LSTM CRF baseline architecture."),
      P("3.4 Methods tried to raise the Bi-LSTM CRF F1", h2)]
s += B(["<b>Word dropout (V1).</b> During training the GloVe vector of a random 10% of tokens is set to zero, exactly what happens to a word without a vector. In test, "
        f"{p1(100 - gk['test'][1]) if 'test' in gk else '15'}% of entity tokens have no vector, against {p1(100 - gk['train'][1]) if 'train' in gk else '6'}% in train, so the baseline rarely learns to rely on characters and context when the word vector is missing.",
        "<b>Fine-tuned GloVe (V2).</b> The frozen vector table becomes trainable with an effective learning rate ten times smaller than the other weights, so the vectors can adapt to entity roles without drifting on 3,394 sentences; the vector of unknown words also becomes trainable.",
        "<b>Larger hidden layer (V3).</b> 256 units per direction give the encoder more capacity for context.",
        "<b>POS-tag input (V4).</b> The NLTK tag of each token enters as a 12-dimensional embedding. Proper-noun tags (NNP) flag names, and the Reddit part of the test set was itself selected with a proper-noun tagger.",
        "<b>Combination (V5).</b> Rule fixed in advance: keep every single change that gains at least 0.3 dev F1 and train them together.",
        "<b>Precision-recall shift at decoding.</b> A constant delta is subtracted from the score of the O tag before Viterbi decoding. It needs no retraining; delta is chosen on dev among 0, 0.5, ..., 3.0 (ties go to the smaller value).",
        "<b>Ensemble.</b> Defined before its scores were seen: average the emission scores and the CRF start, end and transition scores of the six seed-42 models (V0 to V5), then decode with Viterbi. No variant has to be chosen; the members differ in hidden size, embedding training, word dropout and POS input, not in the seed."])

# ---------------------------------------------------------------- 4 Results
s += [P("4. Experiments and Results", h1), P("4.1 Search on dev (Twitter-only training)", h2)]
rows = [["Model", "Searched", "Selected", "Dev F1"],
        ["CRF", "c1 in {0.1, 0.3, 1.0} x c2 in {0.003, 0.01, 0.03}, with and without cluster features (18 runs)", "c1 = 0.3, c2 = 0.003, all features", p1(DEV["CRF"])],
        ["SVM", "C in {0.01, 0.03, 0.1, 0.3}; class weights; GloVe vector; BIO repair on or off", "C = 0.03, balanced, GloVe, no repair", p1(DEV["SVM"])],
        ["Bi-LSTM CRF", "four single changes, their combination, an ensemble of the six models, and a decoding shift (Section 4.3)", "baseline reported; dev choice among single models: V3; ensemble of V0 to V5", p1(DEV["Bi-LSTM CRF (ensemble)"]) + " (ensemble)"]]
s += tbl("Search space and selected settings (CRF and SVM searches were run before the preprocessing fix, Appendix B).", rows, [2.2, 8.2, 4.6, 2.0], align=["l", "l", "l", "c"], boldcol0=True)
s += [P("4.2 Main results", h2)]
prevf = {"CRF": PREV["crf"]["F1"], "SVM": PREV["svm"]["F1"], "Bi-LSTM CRF (baseline)": REF42, "Bi-LSTM CRF (ensemble)": REF42}
rows = [["Model (train file only)", "Precision", "Recall", "F1 entity", "F1 surface", "Token acc.", "Earlier train+dev F1 (reference)"]]
for m in M4: rows.append([m, p1(MD[m]["P"]), p1(MD[m]["R"]), p1(MD[m]["F1"]), p1(MD[m]["surf"]), p1(MD[m]["tok"]), p1(prevf[m]) if m != "Bi-LSTM CRF (ensemble)" else "-"])
rows.append(["Bi-LSTM CRF (V3 + shift, the single model chosen by dev)", p1(SEL["P"]), p1(SEL["R"]), p1(SEL["F1"]), p1(SEL["surf"]), p1(SEL["tok"]), "-"])
rows += [["Reference: predict O everywhere", "-", "-", "-", "-", "92.6", "-"], ["Reference: best shared-task system", "-", "-", "41.86", "40.24", "-", "-"]]
s += tbl("Test results in percent (models trained on the Twitter train file only, seed 42, scored once).", rows, [4.6, 1.8, 1.6, 1.9, 1.9, 1.9, 3.3], hi=(4,))
s += [P(f"Bi-LSTM CRF baseline: {p1(bb['F1'])}; ensemble of six variants: <b>{p1(bi['F1'])}</b> ({gain_test:+.1f}); the single variant chosen by dev, with its dev-chosen shift: {p1(SEL['F1'])} ({SEL['F1'] - bb['F1']:+.1f}); SVM {p1(svm['F1'])}; CRF {p1(crf['F1'])}. The ensemble is {p1(REF - bi['F1'])} below the {p1(REF)} of the earlier train+dev run (mean of three seeds; seed 42 alone gave {p1(REF42)}) although it never sees YouTube data in training. "
         f"Removing the dev data from training costs the CRF {p1(PREV['crf']['F1'] - crf['F1'])} F1 points and the SVM {p1(PREV['svm']['F1'] - svm['F1'])}, and costs the baseline Bi-LSTM CRF {p1(REF42 - bb['F1'])} (seed 42 against seed 42). Token accuracy is close to the 92.6% of predicting O everywhere for the CRF and SVM, so entity F1 is the meaningful measure.")]

# ---- 4.3 trying to raise the F1
s += [P("4.3 Trying to raise the Bi-LSTM CRF F1", h2), P("Every change was trained on train only (seed 42) and stopped on dev. Table 5 lists dev F1 (used to choose) and test F1 (recorded, not used to choose). The dev gains did not carry over to test, and this is the main finding of this section.")]
desc = {"V0": "Baseline: frozen GloVe, char-CNN, capitalization, 128 units", "V1": "Word dropout 0.1 on the GloVe vector", "V2": "Fine-tuned GloVe (0.1 x learning rate)", "V3": "Hidden size 256", "V4": "POS-tag embedding (12 dims)", "V5": "All four changes together"}
rows = [["Step", "Change", "Dev F1", "Dev change", "Test F1", "Test change"]]
for v in V:
    k = v["name"][:2]; rows.append([k, desc[k], p1(v["dev_F1"]), "-" if k == "V0" else f"{v['dev_F1'] - v0['dev_F1']:+.1f}", p1(v["test"]["F1"]), "-" if k == "V0" else f"{v['test']['F1'] - v0['test']['F1']:+.1f}"])
sb0 = sweep_at(base["variant"], d_base); sbf = sweep_at(fin["variant"], d_fin)
rows.append(["E", "Ensemble of V0 to V5 (average scores, no shift)", p1(ER0["dev"]["F1"]), f"{ER0['dev']['F1'] - v0['dev_F1']:+.1f}", p1(ER0["test"]["F1"]), f"{ER0['test']['F1'] - v0['test']['F1']:+.1f}"])
rows.append(["V0+", f"Baseline with shift delta {d_base} (dev choice)", p1(sb0["dev"]["F1"]), f"{sb0['dev']['F1'] - v0['dev_F1']:+.1f}", p1(sb0["test"]["F1"]), f"{sb0['test']['F1'] - v0['test']['F1']:+.1f}"])
rows.append(["V3+", f"V3 with shift delta {d_fin} (single model chosen by dev)", p1(sbf["dev"]["F1"]), f"{sbf['dev']['F1'] - v0['dev_F1']:+.1f}", p1(sbf["test"]["F1"]), f"{sbf['test']['F1'] - v0['test']['F1']:+.1f}"])
rows.append(["E+", f"Ensemble with shift delta {IBE['best_delta']} (dev choice)", p1(ERB["dev"]["F1"]), f"{ERB['dev']['F1'] - v0['dev_F1']:+.1f}", p1(ERB["test"]["F1"]), f"{ERB['test']['F1'] - v0['test']['F1']:+.1f}"])
s += tbl("Everything tried on the Bi-LSTM CRF (seed 42, Twitter-only training). Changes are against V0.", rows, [1.2, 7.4, 1.8, 2.0, 1.8, 2.0], align=["c", "l", "c", "c", "c", "c"], hi=(7,))
s += [fig("f_ladder", 6.0, "Dev F1 (light) and test F1 (dark) for every model tried. Only the dev bars were used to choose.")]
s += B([f"<b>Word dropout</b> (V1). Aim: teach the model to use characters and context when the word vector is missing, which is common for test entities. Dev {SG[0]:+.1f}, test {TG[0]:+.1f}.",
        f"<b>Fine-tuned GloVe</b> (V2). Aim: adapt the vectors to entity roles. Dev {SG[1]:+.1f}, test {TG[1]:+.1f}.",
        f"<b>Hidden size 256</b> (V3). Aim: more capacity for context. It had the largest dev gain ({SG[2]:+.1f}) and the largest test loss ({TG[2]:+.1f}), so it is the clearest sign that dev choices did not transfer.",
        f"<b>POS-tag input</b> (V4). Aim: an explicit proper-noun signal. Dev {SG[3]:+.1f}, test {TG[3]:+.1f}.",
        f"<b>All four together</b> (V5). Dev {VN['V5']['dev_F1'] - v0['dev_F1']:+.1f}, test {VN['V5']['test']['F1'] - v0['test']['F1']:+.1f}: the only combination whose test score did not fall, and still within noise."])
s += [P(f"<b>Why the single changes did not transfer.</b> Dev is YouTube comments, test is Reddit, StackExchange and Twitter 2017, and with Twitter-only training there is no test-like data to tune on. Dev gains of one or two points on 836 entities are of the same size as the seed spread we measured earlier (about 2 points), so choosing among the variants by dev is not reliable. The variant chosen by dev (V3) is significantly worse than the baseline in a paired bootstrap ({bsel[0]:+.1f}, interval {bsel[1]} to {bsel[2]})."),
      P(f"<b>Why the decoding shift did not transfer.</b> On dev the baseline has precision {p1(v0['dev_P'])} and recall {p1(v0['dev_R'])}, so subtracting a constant from the O score raised dev F1 (delta {d_base}: {sb0['dev']['F1']:.1f} against {v0['dev_F1']:.1f}). On test the same shift moves precision from {p1(v0['test']['P'])} to {p1(sb0['test']['P'])} but recall only from {p1(v0['test']['R'])} to {p1(sb0['test']['R'])}, so test F1 falls ({sb0['test']['F1']:.1f} against {v0['test']['F1']:.1f}). Test entities are harder (more unseen names), so a shift tuned on the easier dev set over-predicts entities; the same happens for V3 and for the ensemble (Figure 6 in Appendix F)."),
      P(f"<b>What did work: the ensemble.</b> Averaging the six models raised dev F1 from {v0['dev_F1']:.1f} to {ER0['dev']['F1']:.1f} and test F1 from {v0['test']['F1']:.1f} to {ER0['test']['F1']:.1f}, with precision {p1(ER0['test']['P'])} against {p1(v0['test']['P'])} and recall {p1(ER0['test']['R'])} against {p1(v0['test']['R'])}. Averaging reduces the variance of each member, and the members differ in size and inputs. "
        f"Two cautions. First, under the strict rule of choosing the single model with the best dev F1 the winner is V3 ({VN['V3']['dev_F1']:.1f} against {ER0['dev']['F1']:.1f}), so the ensemble was not selected by dev; we report it as a separate method fixed before its scores were seen. Second, its test gain over the baseline is {pb[0]:+.1f} with a paired bootstrap interval of {pb[1]} to {pb[2]}, which includes zero: the gain is suggestive, not proven. With the shift chosen on dev (delta {IBE['best_delta']}) the ensemble scores {ERB['test']['F1']:.1f}, still {ERB['test']['F1'] - sb0['test']['F1']:+.1f} above the baseline with its shift.")]

# ---- 4.4 reliability
s += [P("4.4 How reliable are the differences?", h2),
      P(f"With one seed, the bootstrap over test sentences gives 95% intervals of about 3 points (CRF {BS['CRF'][0]} to {BS['CRF'][1]}, SVM {BS['SVM'][0]} to {BS['SVM'][1]}, baseline {BS['Bi-LSTM CRF (baseline)'][0]} to {BS['Bi-LSTM CRF (baseline)'][1]}, ensemble {BS['Bi-LSTM CRF (ensemble)'][0]} to {BS['Bi-LSTM CRF (ensemble)'][1]}). "
        f"The paired difference ensemble minus baseline is {pb[0]:+.1f} (interval {pb[1]} to {pb[2]}) and V3 with shift minus baseline is {bsel[0]:+.1f} (interval {bsel[1]} to {bsel[2]}). "
        f"The bootstrap does not cover training randomness: in an earlier three-seed run of the baseline architecture (train+dev protocol), seeds 42, 43 and 44 gave {', '.join(p1(v) for v in SEEDS)}, a spread of {max(SEEDS) - min(SEEDS):.1f} points, so a gain of about 2 points or less should be read as suggestive until several seeds are run. "
        f"The baseline Bi-LSTM CRF leads the SVM by {BS['Bi-LSTM CRF (baseline) minus SVM'][0]:+.1f} (interval {BS['Bi-LSTM CRF (baseline) minus SVM'][1]} to {BS['Bi-LSTM CRF (baseline) minus SVM'][2]}) and the SVM leads the CRF by {BS['SVM minus CRF'][0]:+.1f} (interval {BS['SVM minus CRF'][1]} to {BS['SVM minus CRF'][2]}). "
        f"Five-fold cross-validation on train (document-grouped, before the preprocessing fix) gives {p1(CV['crf_mean'])} +- {p1(CV['crf_std'])} for the CRF and {p1(CV['svm_mean'])} +- {p1(CV['svm_std'])} for the SVM; these come from one domain and only show that the settings are stable.")]
rows = [["Type", "Support", "CRF", "SVM", "Bi-LSTM baseline", "Bi-LSTM ensemble"]] + [[t, str(SUP[t]), p1(crf["types"][t]), p1(svm["types"][t]), p1(bb["types"][t]), p1(bi["types"][t])] for t in T6]
s += [P("4.5 Results by entity type", h2)] + tbl("F1 by entity type on the test set.", rows, [3.6, 2.0, 2.4, 2.4, 3.3, 3.3])
s += [P(f"Person and location are easiest. Corporation, product, creative-work and group reach at most {p1(WEAK)} F1 for every model: they have few training examples and overlapping definitions, and creative-work titles are long and free-form."),
      P("4.6 Ablations of the CRF and SVM", h2),
      P(f"<b>CRF features</b> (cumulative, dev, Figure 3 left; run before the preprocessing fix): word identity alone gives {p1(ab[0]['F1'])} F1; shape, affixes and social-media flags add {p1(dab(1))}; context words and bigrams add {p1(dab(2))}; POS tags add {p1(dab(3))}; cluster IDs add {p1(dab(4))} in the final setting but {clu_flip:+.1f} at c1 = 0.3, c2 = 0.03, so their effect is small and unstable. "
        f"<b>SVM</b> (dev, Figure 3 right): removing class weights raises precision to {p1(svnw['P'])} but recall falls to {p1(svnw['R'])} and F1 from {p1(svf['F1'])} to {p1(svnw['F1'])}; removing the GloVe vector drops F1 to {p1(svng['F1'])}. "
        f"<b>BIO repair</b> (planned in Stage 1) lowered the SVM F1 on dev and on test ({p1(pct(TS['test_repair']['entity']['f1']))} with repair against {p1(svm['F1'])} without, on test), because {TS['invalid_I']} orphan I- tags in the raw output are ignored by strict scoring while repair turns them into mostly wrong entities. Repair was dropped, a decision taken on dev."),
      fig("f_ablation", 5.3, "Left: cumulative CRF feature ablation on dev. Right: SVM ablations on dev (amber: setting used to select C).")]
s += [P("4.7 Error analysis and confusion matrices", h2), P("Each gold test entity falls in one category: found exactly, found with a wrong boundary or type, or missed. Predictions overlapping no gold entity are spurious.")]
rows = [["Category"] + M4] + [["Correct (exact span and type)"] + [str(E[m]["benar"]) for m in M4], ["Wrong boundary or type"] + [str(wrong(E[m])) for m in M4], ["Missed entirely"] + [str(E[m]["terlewat"]) for m in M4], ["Spurious predictions"] + [str(E[m]["prediksi_palsu"]) for m in M4]]
s += tbl("Outcomes for the 1,079 gold test entities.", rows, [4.6, 2.2, 2.2, 3.6, 3.6])
s += [P(f"The CRF misses {E['CRF']['terlewat']} entities ({100 * E['CRF']['terlewat'] / 1079:.0f}%). The SVM gets {E['SVM']['benar']} right, finds {wrong(E['SVM'])} more with the wrong boundary or type and adds {E['SVM']['prediksi_palsu']} spurious predictions. "
         f"The baseline Bi-LSTM CRF gets {E['Bi-LSTM CRF (baseline)']['benar']} right and the ensemble {E['Bi-LSTM CRF (ensemble)']['benar']}, with {E['Bi-LSTM CRF (baseline)']['prediksi_palsu']} and {E['Bi-LSTM CRF (ensemble)']['prediksi_palsu']} spurious predictions and {E['Bi-LSTM CRF (baseline)']['terlewat']} and {E['Bi-LSTM CRF (ensemble)']['terlewat']} misses: the ensemble mainly removes spurious predictions. "
         "The confusion matrices (Figure 4) show that misses and boundary errors dominate, while type-to-type confusion is small."),
      fig("f_confusion_to", 5.9, "Entity-level confusion matrices. Rows: gold type, and SPURIOUS (predicted spans overlapping no gold entity). Columns: predicted type (exact span), wrong boundary, missed."),
      fig("f_analysis_to", 5.6, "Recall of gold test entities by novelty (against the train file), capitalization and length (group sizes in brackets)."),
      P(f"Only {AN['CRF']['seen']['n']} of 1,079 test entities occur in the train file. On the {AN['CRF']['unseen']['n']} unseen ones, recall is {p1(AN['CRF']['unseen']['recall'])} (CRF), {p1(AN['SVM']['unseen']['recall'])} (SVM), {p1(AN['Bi-LSTM CRF (baseline)']['unseen']['recall'])} (baseline) and {p1(AN['Bi-LSTM CRF (ensemble)']['unseen']['recall'])} (ensemble). Entities with a lowercase token reach only {p1(min(LC))} to {p1(max(LC))}%. On entities of two or more tokens the ensemble Bi-LSTM CRF recovers {p1(AN['Bi-LSTM CRF (ensemble)']['>=2 token']['recall'])}% against {p1(AN['CRF']['>=2 token']['recall'])}% (CRF) and {p1(AN['SVM']['>=2 token']['recall'])}% (SVM)."),
      P("4.8 Examples of correct and incorrect predictions", h2)]
s += tbl("One correct and one incorrect test sentence per model (more in Appendix E; Bi-LSTM CRF is the improved model).", ex_rows(1), [2.1, 1.6, 5.4, 3.9, 4.0], align=["l", "l", "l", "l", "l"], boldcol0=True)

# ---------------------------------------------------------------- 5 Discussion
s += [P("5. Discussion", h1),
      P(f"<b>What worked and what did not.</b> Each of the four single changes looked better on dev ({min(SG):+.1f} to {max(SG):+.1f}) and none did better on test ({min(TG):+.1f} to {max(TG):+.1f}); the decoding shift tuned on dev lowered test F1 for the baseline, V3 and the ensemble. Only the ensemble rose on both dev and test. The lesson is about tuning: when dev (YouTube) and test (Reddit, StackExchange, Twitter 2017) differ, and training uses Twitter only, small dev gains are not evidence of test gains. This is a cost of not mixing sources in training and having no test-like data for tuning."),
      P(f"<b>Why the ranking of models.</b> The SVM and Bi-LSTM CRF see dense GloVe vectors; the CRF sees GloVe only through cluster IDs and has clearly lower recall on unseen entities ({p1(AN['CRF']['unseen']['recall'])}% against {p1(AN['SVM']['unseen']['recall'])}% for the SVM). In the SVM the dense vector is worth {p1(svf['F1'] - svng['F1'])} dev F1 points. We have not tested this directly for the CRF or the Bi-LSTM CRF, so it remains a hypothesis."),
      P(f"<b>Comparison with the shared task.</b> The ensemble ({p1(bi['F1'])}) and the baseline ({p1(bb['F1'])}) are close to the best participant (41.86); the SVM and CRF sit lower in the reported range (26.30 to 41.86). The comparison is approximate: the released test set holds 1,079 entities after annotation fixes (the paper counts 1,040) and our surface F1 lowercases strings. With one seed and a bootstrap interval of about 3 points we do not claim to beat the shared-task system, and we do not claim that the ensemble beats the baseline beyond noise."),
      P("<b>Limitations.</b> One seed for every model (the ensemble members share seed 42 and differ in configuration). CRF and SVM searches, ablations and cross-validation were run before the preprocessing fix; the Bi-LSTM CRF runs are after it. Improvement choices were made on a single dev set from a different source than test. There is no run without GloVe. The neural model is written in JAX. The GloVe vector set includes words from all splits (frozen or slowly fine-tuned vectors, no labels, no training on test text)."),
      P("<b>Not done yet, on purpose.</b> Training data are not mixed across sources. An earlier CRF-only analysis (not part of this protocol) suggested that adding YouTube data helps (test F1 16.8 with train only, 27.7 with train plus dev), and the earlier Bi-LSTM CRF run with train plus dev reached 41.6 (mean of three seeds); mixing sources is the natural next step.")]
s += [P("6. Conclusion and Future Work", h1),
      P(f"Trained on Twitter only with seed 42, the Bi-LSTM CRF baseline scores {p1(bb['F1'])} entity F1 on the WNUT17 test set. Four changes and a decoding shift that improved dev did not improve test, and the single model chosen by dev scored {p1(SEL['F1'])}. An ensemble of the six seed-42 models, defined in advance, raised test F1 to {p1(bi['F1'])} ({gain_test:+.1f}), {cmp_ref}, but this gain is within the noise of a single seed. The SVM ({p1(svm['F1'])}) and CRF ({p1(crf['F1'])}) stay lower. Unseen entities, entities with lowercase tokens and the types corporation, product, creative-work and group remain the hard cases. Next steps: run several seeds to confirm the ensemble gain, mix in sources beyond Twitter, tune on data that resemble the test domain, test the role of GloVe with a run without it, give the CRF dense embedding features, and try contextual encoders such as BERTweet.")]
s += [P("References", h1)]
refs = ["Aguilar, G., Maharjan, S., Lopez Monroy, A. P. and Solorio, T. (2017). A multi-task approach for named entity recognition in social media data. Proc. 3rd Workshop on Noisy User-generated Text, 148-153.", "Akbik, A., Blythe, D. and Vollgraf, R. (2018). Contextual string embeddings for sequence labeling. COLING, 1638-1649.",
        "Bird, S., Klein, E. and Loper, E. (2009). Natural Language Processing with Python. O'Reilly.", "Bradbury, J. et al. (2018). JAX: composable transformations of Python+NumPy programs. github.com/google/jax.", "Chiu, J. and Nichols, E. (2016). Named entity recognition with bidirectional LSTM-CNNs. TACL, 4, 357-370.",
        "Cortes, C. and Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.", "Derczynski, L., Nichols, E., van Erp, M. and Limsopatham, N. (2017). Results of the WNUT2017 shared task on novel and emerging entity recognition. Proc. 3rd Workshop on Noisy User-generated Text, 140-147.",
        "Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019). BERT: pre-training of deep bidirectional transformers for language understanding. NAACL-HLT, 4171-4186.", "Efron, B. and Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman and Hall.",
        "Huang, Z., Xu, W. and Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. arXiv:1508.01991.", "Isozaki, H. and Kazawa, H. (2002). Efficient support vector classifiers for named entity recognition. COLING, 1-7.", "Kingma, D. and Ba, J. (2015). Adam: a method for stochastic optimization. ICLR.",
        "Kudo, T. and Matsumoto, Y. (2001). Chunking with support vector machines. NAACL.", "Lafferty, J., McCallum, A. and Pereira, F. (2001). Conditional random fields: probabilistic models for segmenting and labeling sequence data. ICML, 282-289.",
        "Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K. and Dyer, C. (2016). Neural architectures for named entity recognition. NAACL-HLT, 260-270.", "Ma, X. and Hovy, E. (2016). End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. ACL, 1064-1074.",
        "McCallum, A. and Li, W. (2003). Early results for named entity recognition with conditional random fields, feature induction and web-enhanced lexicons. CoNLL, 188-191.", "Nakayama, H. (2018). seqeval: a Python framework for sequence labeling evaluation. github.com/chakki-works/seqeval.",
        "Nguyen, D. Q., Vu, T. and Nguyen, A. T. (2020). BERTweet: a pre-trained language model for English tweets. EMNLP (System Demonstrations), 9-14.", "Okazaki, N. (2007). CRFsuite: a fast implementation of conditional random fields. www.chokkan.org/software/crfsuite.",
        "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. JMLR, 12, 2825-2830.", "Pennington, J., Socher, R. and Manning, C. (2014). GloVe: global vectors for word representation. EMNLP, 1532-1543.",
        "Ritter, A., Clark, S., Mausam and Etzioni, O. (2011). Named entity recognition in tweets: an experimental study. EMNLP, 1524-1534.", "Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I. and Salakhutdinov, R. (2014). Dropout: a simple way to prevent neural networks from overfitting. JMLR, 15, 1929-1958."]
s += [Paragraph(f"[{i + 1}] {x}", st("ref", fontSize=8.3, leading=11.3, leftIndent=17, firstLineIndent=-17, spaceAfter=2.5)) for i, x in enumerate(refs)]

# ================================================================ appendices
s += [PageBreak(), P("Appendix A. Preprocessing in Detail", h1)]
s += B(["<b>Reading:</b> UTF-8; a blank line ends a document; a tab-only line ends a sentence (train: 2,394 such lines); the loader recomputes every count in Table 1.",
        "<b>Normalization</b> (function <i>normalize</i>): 1 HTML unescape; 2 NFKC; 3 tokens starting with http://, https://, www. or ending in .com, .net, .org, .io, .co, .ly become &lt;URL&gt;; 4 tokens starting with @ become &lt;USER&gt;; 5 a leading # is dropped; 6 every digit becomes 0; 7 a non-digit character repeated four or more times is cut to three; 8 lowercase.",
        f"<b>GloVe lookup keys</b> (function <i>glove_keys</i>): lowercase original token; normalized form; &lt;url&gt; or &lt;user&gt; for those classes; &lt;number&gt; when the normalized form is only zeros; the hashtag body. {nkeys:,} distinct keys are needed and 11,577 exist in GloVe Twitter 200d. Token coverage: train {cov('train', 0)}%, dev {cov('dev', 0)}%, test {cov('test', 0)}%; entity tokens {cov('train', 1)}%, {cov('dev', 1)}%, {cov('test', 1)}%.",
        "<b>POS tags:</b> NLTK averaged perceptron (Penn Treebank). <b>Clusters:</b> MiniBatchKMeans (k = 500, seed 42) on the 100,000 most frequent GloVe Twitter words; tokens without a vector get -1.",
        "<b>Capitalization.</b> The original-case token is kept for shape, is_title and is_upper features, for the capitalization embedding and for the character CNN. Lexical features, prefixes, suffixes, context words and the GloVe lookup are lowercased, and GloVe Twitter is uncased, so <i>Google</i> and <i>google</i> differ only in shape (Xx against x) and is_title for the CRF and SVM.",
        "<b>Bi-LSTM CRF:</b> character vocabulary from the training data (92 characters), words without a GloVe vector share one all-zero vector (trainable in V2), length buckets 8 to 128 (longest test sequence: 105 tokens, nothing truncated)."])
rows = [["Group", "Features"], ["Lexical", "lowercase token, normalized form, stop-word flag (NLTK list)"], ["Context (window +-2)", "lowercase words at -2, -1, +1, +2 (BOS/EOS markers); title-case and upper-case flags at -1 and +1; left and right bigrams"], ["Shape", "collapsed shape pattern (Xxxx, dd), is_title, is_upper, has_digit, has_hyphen"],
        ["Affixes", "prefixes and suffixes of 1 to 4 characters (lowercase), token length (capped at 15)"], ["Social media", "is_url, is_mention, is_hashtag, is_emoji"], ["POS tags", "tag of the token and of the tokens at -1 and +1"], ["Cluster IDs", "cluster of the GloVe vector of the token and of the tokens at -1 and +1"], ["Dense vector (SVM only)", "200-d GloVe vector of the token, scaled by 0.3"]]
s += tbl("Features used by the CRF and SVM.", rows, [3.6, 13.4], align=["l", "l"], boldcol0=True)
s += [fig("f_pipeline_to", 6.4, "Overall pipeline of the Twitter-only protocol.")]

s += [P("Appendix B. Deviations from the Stage 1 Plan, Protocol Changes and Bug-Fix Log", h1)]
rows = [["Item (Stage 1 slides 21 to 25 / report)", "Plan", "Implemented", "Status"],
        ["Training data", "Train on train + dev after tuning (slide 24)", "Train file only (Twitter); dev only for tuning and stopping", "Changed on request"],
        ["Random seeds (slide 24)", "Seed 42", "Seed 42 for every run; no other seeds in this report", "As planned"],
        ["Bi-LSTM CRF improvements", "Not planned", "Word dropout, fine-tuned GloVe, hidden 256, POS input, ensemble, decoding shift (Section 4.3)", "Added"],
        ["Digit rule (slide 21)", "2017 becomes 0000", "Was 000 (repeat cap collapsed digit runs); now 0000", "Bug fixed"], ["URL rule (report Table 8)", "http, www and *.com become &lt;URL&gt;", "Bare domains were missing; now included", "Bug fixed"], ["Hashtag rule (report Table 8)", "Drop #, process the word", "# was kept in the normalized form; now dropped", "Bug fixed"],
        ["Fix timing", "-", "Bi-LSTM CRF runs use the fixed preprocessing; CRF/SVM searches, ablations and cross-validation were run before it", "Disclosed"],
        ["Cross-validation (slide 24)", "5-fold by document on train", "Train only, final settings, before the fix (CRF 34.4, SVM 43.1)", "Done"], ["SVM BIO repair (slide 22)", "Apply after prediction", "Built and tested, dropped: lower F1 on dev and test", "Changed"],
        ["SVM extras (report 3.5)", "Context-window GloVe average; label history", "Token GloVe vector only (x 0.3); no label history", "Not done"], ["SVM feature space", "All features", "Features seen at least 3 times", "Changed (speed)"],
        ["Context features (slide 23)", "Lowercase and shape for +-2", "Lowercase +-2, title/upper flags +-1, no shape for context words", "Partly"], ["Other shape flags (report 3.5)", "is_lower, is_digit, has_apostrophe, is_punct, has_non_ascii, full shape", "Not included", "Not done"],
        ["Vocabulary (slide 22)", "Train vocabulary plus embedding words", "All GloVe words found in any split (no labels); PAD and UNK both zero", "Changed (disclosed)"], ["Embeddings for Bi-LSTM (report 3.3)", "GloVe, fine-tuned", "Frozen in the baseline; fine-tuned in V2", "Tested"],
        ["FastText, demo tokenizer (slides 21, 22)", "Alternative embedding; TweetTokenizer for new text", "Not run; not built", "Not done"], ["BIOES (slide 22)", "Try for Bi-LSTM CRF", "One train+dev-era run, no reliable gain; not repeated here", "Inconclusive"], ["Framework", "Not specified", "JAX and optax (PyTorch could not be installed here)", "Note"]]
s += tbl("Differences between the Stage 1 plan and this report.", rows, [4.1, 4.3, 6.1, 2.5], align=["l", "l", "l", "l"], boldcol0=True)

s += [P("Appendix C. Chronological Log", h1)]
rows = [["#", "Step", "Outcome"], ["1", "Environment: Python 3.12, one CPU core (about 4 GB RAM), scikit-learn, sklearn-crfsuite, seqeval, NLTK, JAX, optax. PyTorch did not fit the free disk space.", "Toolchain fixed."],
        ["2", "Cloned WNUT17, recomputed all Stage 1 statistics, downloaded GloVe Twitter 200d (795 MB) and kept the needed vectors.", "All counts reproduced."], ["3", "CRF search (extended grid, 18 runs) and SVM search (C, class weights, GloVe, BIO repair) on dev.", f"CRF dev {p1(C2['best']['dev_F1'])} (train only); SVM C = 0.03."],
        ["4", "Compared the implementation with Stage 1 slides 20 to 25; found three normalization bugs and fixed them.", "Bugs in Appendix B."], ["5", "Retrained the CRF and SVM on train only; trained the Bi-LSTM CRF baseline on train only with seed 42.", f"Test F1 {p1(crf['F1'])}, {p1(svm['F1'])}, {p1(bb['F1'])}."],
        ["6", "Trained four single changes and their combination, ran a decoding-shift sweep, and averaged the six models into an ensemble; chose by dev only.", f"Single model chosen by dev: V3 (test {p1(SEL['F1'])}); ensemble test {p1(bi['F1'])}."], ["7", "Bootstrap, error analysis, confusion matrices, examples.", "Section 4."]]
s += tbl("Work log.", rows, [0.8, 11.6, 4.6], align=["c", "l", "l"])

s += [P("Appendix D. Full Tuning Grids (Dev Set, Before the Preprocessing Fix)", h1)]
rows = [["Features", "c1", "c2", "Precision", "Recall", "F1 entity", "F1 surface"]]
for g in sorted(C2["grid"], key=lambda z: (z["fitur"] != "lengkap", z["c1"], z["c2"])): rows.append(["full" if g["fitur"] == "lengkap" else "no cluster", g["c1"], g["c2"], p1(g["P"]), p1(g["R"]), p1(g["F1"]), p1(g["surfF1"])])
bi_ = [i for i, x in enumerate(rows) if x[0] == "full" and x[1] == C2["best"]["c1"] and x[2] == C2["best"]["c2"]]
s += tbl("CRF search (train only, 300 iterations). The selected row is shaded.", rows, [3.0, 1.8, 1.8, 2.6, 2.4, 2.8, 2.6], hi=tuple(bi_))
d2 = {x["C"]: x for x in SV2["dev"]}; rows = [["C", "Dev F1 with BIO repair", "Dev F1 without BIO repair"]]
for gg in sorted(SV["grid"] + [{"C": 0.01, "F1": SV2["dev"][0]["repair"]["F1"]}], key=lambda z: z["C"]): rows.append([gg["C"], p1(gg["F1"]), p1(d2[gg["C"]]["no_repair"]["F1"]) if gg["C"] in d2 else "not run"])
s += tbl("SVM search (train only). C = 0.03 was selected with BIO repair on.", rows, [3.0, 6.0, 6.0], hi=(2,))

s += [P("Appendix E. More Examples of Correct and Incorrect Predictions", h1)]
s += tbl("Two correct and two incorrect test sentences per model (most gold entities first, then shortest).", ex_rows(2), [2.1, 1.6, 5.4, 3.9, 4.0], align=["l", "l", "l", "l", "l"], boldcol0=True)

s += [P("Appendix F. Decoding Shift (O-bias) Sweep and Learning Curves", h1)]
rows = [["Delta", "Dev P", "Dev R", "Dev F1", "Test P", "Test R", "Test F1"]]
for x in sw[fin["variant"]]["rows"]: rows.append([x["delta"], p1(x["dev"]["P"]), p1(x["dev"]["R"]), p1(x["dev"]["F1"]), p1(x["test"]["P"]), p1(x["test"]["R"]), p1(x["test"]["F1"])])
hi_i = [i for i, x in enumerate(rows) if i > 0 and x[0] == d_fin]
s += tbl(f"O-bias sweep for V3, the single model chosen by dev. Delta is chosen on dev; the test columns are shown for transparency.", rows, [2.0, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4], hi=tuple(hi_i))
rows = [["Delta", "Dev P", "Dev R", "Dev F1", "Test P", "Test R", "Test F1"]] + [[x["delta"], p1(x["dev"]["P"]), p1(x["dev"]["R"]), p1(x["dev"]["F1"]), p1(x["test"]["P"]), p1(x["test"]["R"]), p1(x["test"]["F1"])] for x in IBE["rows"]]
hi_e = [i for i, x in enumerate(rows) if i > 0 and x[0] == IBE["best_delta"]]
s += tbl("O-bias sweep for the ensemble (delta chosen on dev; test columns for transparency).", rows, [2.0, 2.4, 2.4, 2.4, 2.4, 2.4, 2.4], hi=tuple(hi_e))
s += [fig("f_sweep_to", 6.0, "Figure 6. F1 against delta on dev (solid) and test (dashed): dev prefers a shift of 1.0 to 1.5, test prefers little or none."), fig("f_curves_to", 6.0, "Dev F1 per epoch of the baseline and the final variant.")]
rows = [["Step", "Config", "Epochs run", "Best epoch", "Seconds"]] + [[v["name"], json.dumps({k: val for k, val in v["cfg"].items() if k != "pos_dim"}) + (" + POS" if v["pos"] else ""), v["epochs_run"], v["best_epoch"], v["sec"]] for v in V]
s += tbl("All Bi-LSTM CRF variants (seed 42).", rows, [4.6, 6.0, 2.0, 2.2, 2.2], align=["l", "l", "c", "c", "c"])

s += [P("Appendix G. Reproducibility", h1),
      P("<b>Environment.</b> Python 3.12.3; numpy 2.4.4; scipy 1.17.1; scikit-learn 1.8.0; sklearn-crfsuite 0.5.0; seqeval 1.2.2; nltk 3.10.3; jax and jaxlib 0.11.2; optax 0.2.8; matplotlib 3.10.8. One CPU core, about 4 GB RAM. A Bi-LSTM CRF epoch takes 10 to 25 seconds after a first epoch of about one minute (compilation). Every script uses seed 42."),
      P("<b>Code (archive).</b> The notebook reloads saved predictions and recomputes all tables and figures (set RUN_TRAINING = True to retrain). Modules: wnut_data.py, prep_glove.py, features.py, evalx.py, bilstm_crf.py. Twitter-only runs: trainonly_models.py (CRF, SVM), improve_bilstm.py (Bi-LSTM CRF variants, decoding shift) and ensemble_bilstm.py; then analysis_to.py, make_figures_to.py, build_report3.py. Search, ablation and cross-validation scripts: run_crf2.py, run_svm.py, run_svm2.py, run_cv_final.py."),
      P("<b>Data.</b> WNUT17 from github.com/leondz/emerging_entities_17 (CC BY 4.0); GloVe Twitter 200d from the gensim-data release glove-twitter-200. Predictions and metrics of every run are stored as JSON in the results folder.")]
doc.build(s); print("built", cnt)
