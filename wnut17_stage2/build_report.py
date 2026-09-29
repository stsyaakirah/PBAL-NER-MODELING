import json, os, sys, pickle
sys.path.insert(0, "/home/claude/proj"); os.chdir("/home/claude/proj")
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm, inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.platypus import BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
from PIL import Image as PILImage

R = lambda p: json.load(open("results/" + p))
ex = lambda p: os.path.exists("results/" + p)
S, A, SV, SV2, SF, C2, BL, BS = R("summary.json"), R("analysis.json"), R("svm.json"), R("svm2.json"), R("svm_final.json"), R("crf2.json"), R("bilstm.json"), R("bootstrap.json")
CV = R("cv_final.json") if ex("cv_final.json") else None
SVT = R("svm_trainonly_final.json") if ex("svm_trainonly_final.json") else None
crf, svm, bi = S["crf"], S["svm"], S["bilstm"]
p1 = lambda x: f"{x:.1f}"; p2 = lambda x: f"{x:.2f}"; pc = lambda x: f"{100 * x:.1f}"
M3 = ["CRF", "SVM", "Bi-LSTM CRF"]; key = {"CRF": "crf", "SVM": "svm", "Bi-LSTM CRF": "bilstm"}
T6 = ["person", "location", "corporation", "product", "creative-work", "group"]; SUP = {"person": 429, "location": 150, "corporation": 66, "product": 127, "creative-work": 142, "group": 165}

NAVY, GOLD, BG, LINE, INK, MUT, TINT = [colors.HexColor(c) for c in ("#12406A", "#B8960C", "#F4F5F7", "#D9DEE5", "#1F2937", "#5B6B7C", "#E6EEF6")]
def st(name, **kw):
    base = dict(fontName="Helvetica", fontSize=9.3, leading=13.6, textColor=INK, spaceAfter=5, alignment=TA_LEFT); base.update(kw); return ParagraphStyle(name, **base)
body = st("body"); h1 = st("h1", fontName="Helvetica-Bold", fontSize=14.5, leading=18, textColor=NAVY, spaceBefore=15, spaceAfter=6, keepWithNext=1)
h2 = st("h2", fontName="Helvetica-Bold", fontSize=10.8, leading=14, textColor=NAVY, spaceBefore=9, spaceAfter=3, keepWithNext=1)
cap = st("cap", fontName="Helvetica-Oblique", fontSize=8.2, leading=11, textColor=MUT, spaceAfter=10, spaceBefore=3)
tcap = st("tcap", fontName="Helvetica-Bold", fontSize=8.6, leading=11.5, textColor=NAVY, spaceBefore=8, spaceAfter=3, keepWithNext=1)
note = st("note", fontSize=8, leading=11, textColor=MUT, spaceBefore=3, spaceAfter=8)
bul = st("bul", leftIndent=13, bulletIndent=2, spaceAfter=2.5)
cell = {a: st("c" + a, fontSize=8.3, leading=10.8, spaceAfter=0, alignment={"l": TA_LEFT, "c": TA_CENTER, "r": TA_RIGHT}[a]) for a in "lcr"}
cellb = {a: st("cb" + a, fontName="Helvetica-Bold", fontSize=8.3, leading=10.8, spaceAfter=0, alignment={"l": TA_LEFT, "c": TA_CENTER, "r": TA_RIGHT}[a]) for a in "lcr"}
chdr = {a: st("ch" + a, fontName="Helvetica-Bold", fontSize=8.3, leading=10.8, textColor=colors.white, spaceAfter=0, alignment={"l": TA_LEFT, "c": TA_CENTER, "r": TA_RIGHT}[a]) for a in "lcr"}

cnt = {"t": 0, "f": 0}
def P(t, s=body): return Paragraph(t, s)
def B(items): return [Paragraph(t, bul, bulletText="\u2022") for t in items]
def table(rows, widths, align=None, hi=(), boldcol0=False, hdr=True):
    align = align or ["l"] + ["c"] * (len(rows[0]) - 1); data = []
    for r, row in enumerate(rows):
        data.append([Paragraph(str(v), chdr[align[c]] if (hdr and r == 0) else (cellb[align[c]] if (boldcol0 and c == 0) or r in hi else cell[align[c]])) for c, v in enumerate(row)])
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1 if hdr else 0)
    sty = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("GRID", (0, 0), (-1, -1), 0.4, LINE), ("TOPPADDING", (0, 0), (-1, -1), 3.2), ("BOTTOMPADDING", (0, 0), (-1, -1), 3.2), ("LEFTPADDING", (0, 0), (-1, -1), 4.5), ("RIGHTPADDING", (0, 0), (-1, -1), 4.5)]
    if hdr: sty.append(("BACKGROUND", (0, 0), (-1, 0), NAVY))
    for r in range(1 if hdr else 0, len(rows)):
        if r in hi: sty.append(("BACKGROUND", (0, r), (-1, r), TINT))
        elif (r % 2 == 0): sty.append(("BACKGROUND", (0, r), (-1, r), BG))
    t.setStyle(TableStyle(sty)); return t
def tbl(title, rows, widths, **kw):
    cnt["t"] += 1; return [Paragraph(f"Table {cnt['t']}. {title}", tcap), table(rows, widths, **kw), Spacer(1, 7)]
def fig(name, width_in, caption):
    cnt["f"] += 1; w, h = PILImage.open(f"report_figs/{name}.png").size; W = width_in * inch
    im = Image(f"report_figs/{name}.png", width=W, height=W * h / w); im.hAlign = "CENTER"
    return KeepTogether([im, Paragraph(f"<b>Figure {cnt['f']}.</b> {caption}", cap)])

# ------------------------------------------------------------------ derived numbers
EXTRA = {}
try:
    import prep_glove
    from wnut_data import load
    need = pickle.load(open("glove_vocab.pkl", "rb"))["need_words"]
    for sp in ("train", "dev", "test"):
        toks = [(t, l) for x in load(sp) for t, l in zip(x["tokens"], x["labels"])]
        has = lambda t: any(k in need for k in prep_glove.glove_keys(t))
        EXTRA[sp] = (100 * sum(has(t) for t, _ in toks) / len(toks), 100 * sum(has(t) for t, l in toks if l != "O") / sum(1 for _, l in toks if l != "O"))
except Exception as e:
    print("coverage failed", e)
cov = lambda sp, i: p1(EXTRA[sp][i]) if sp in EXTRA else "n/a"
grid = {(g["fitur"], g["c1"], g["c2"]): g for g in C2["grid"]}
clu_flip = grid[("lengkap", 0.3, 0.03)]["F1"] - grid[("tanpa_cluster", 0.3, 0.03)]["F1"]
ab = C2["ablation"]; d = lambda i: ab[i]["F1"] - ab[i - 1]["F1"]
sa = {a["name"]: a for a in SV["ablation"]}; svf = sa["LinearSVC balanced + GloVe (lengkap)"]; svnw = sa["Tanpa pembobotan kelas"]; svng = sa["Tanpa vektor GloVe"]
dev_norep = SV2["dev"][1]["no_repair"]["F1"]; dev_rep = SV2["dev"][1]["repair"]["F1"]
svt_no = 100 * SVT["no_repair"]["entity"]["f1"] if SVT else None
E = {m: S[key[m]]["errors"] for m in M3}; wrong = lambda e: e["salah_batas"] + e["salah_tipe"] + e["salah_batas_dan_tipe"]

# ------------------------------------------------------------------ document
def footer(c, doc):
    c.saveState(); c.setStrokeColor(LINE); c.setLineWidth(0.5); c.line(2 * cm, 1.65 * cm, A4[0] - 2 * cm, 1.65 * cm)
    c.setFont("Helvetica", 7.8); c.setFillColor(MUT); c.drawString(2 * cm, 1.2 * cm, "NER on WNUT17: CRF, SVM and Bi-LSTM CRF | Kelompok 1 | Advanced NLP, UGM"); c.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page)); c.restoreState()
doc = BaseDocTemplate("/mnt/user-data/outputs/NER_WNUT17_Full_Report.pdf", pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=2.0 * cm, bottomMargin=2.1 * cm,
                      title="Named Entity Recognition on WNUT17 with CRF, SVM and Bi-LSTM CRF", author="Kelompok 1, Pemrosesan Bahasa Alami Lanjut, UGM")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f", leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)], onPage=footer)])
W = 17.0
s = []

# ---- title block
s += [Spacer(1, 4), P("ASSIGNMENT #2, PART 2  |  NER MODELING AND EVALUATION", st("kick", fontName="Helvetica-Bold", fontSize=8.6, textColor=GOLD, spaceAfter=6)),
      P("Named Entity Recognition on Noisy Social Media Text: CRF, SVM and Bi-LSTM CRF on WNUT17", st("title", fontName="Helvetica-Bold", fontSize=20, leading=25, textColor=NAVY, spaceAfter=6)),
      P("Implementation, experiments and results, step by step", st("sub", fontSize=11.5, leading=15, textColor=MUT, spaceAfter=10)),
      P("<b>Kelompok 1</b>, Pemrosesan Bahasa Alami Lanjut, Universitas Gadjah Mada<br/>Amelia Gizzela Sheehan Auni, Aulia Gita Pratiwi, St. Syakirah, Aditya Iko Winarko, Royan Aditya<br/>28 September 2026", st("auth", fontSize=9, leading=13, spaceAfter=12))]
best_ci = BS
abstract = (f"WNUT17 asks whether named entity recognition (NER) systems can find entities they have never seen: 98% of the test surface forms are absent from training. "
            f"We implement three systems on the official splits: a feature-based CRF, a class-weighted linear SVM, and a Bi-LSTM CRF that combines frozen GloVe Twitter vectors, a character CNN and capitalization embeddings. "
            f"Hyperparameters are chosen on the dev set, final models are retrained on train plus dev, and each is scored once on the full gold test file. "
            f"The Bi-LSTM CRF reaches <b>{p1(bi['F1'])}</b> entity F1 (mean of three seeds, SD {p2(bi['std'])}), against <b>{p1(svm['F1'])}</b> for the SVM and <b>{p1(crf['F1'])}</b> for the CRF. "
            f"The Bi-LSTM CRF result is on par with the best shared-task system (41.86), although the comparison is not exact because the released test annotations were corrected. "
            f"Bootstrap intervals of about 3 F1 points show that the Bi-LSTM CRF lead is clear while the SVM lead over the CRF is small. "
            f"Recall on unseen entities is only {p1(S['analysis']['CRF']['unseen']['recall'])} to {p1(S['analysis']['Bi-LSTM CRF']['unseen']['recall'])}%, and corporation, product, creative-work and group stay below 23 F1 for every model.")
box = Table([[P("<b>Abstract</b>", st("ah", fontName="Helvetica-Bold", fontSize=10, textColor=NAVY, spaceAfter=3)), ], [P(abstract, st("ab", fontSize=9.1, leading=13.2, spaceAfter=4))],
             [P("<b>Keywords:</b> named entity recognition, WNUT17, emerging entities, CRF, SVM, Bi-LSTM CRF, GloVe, noisy text", st("kw", fontSize=8.4, textColor=MUT, spaceAfter=0))]], colWidths=[W * cm])
box.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), BG), ("LEFTPADDING", (0, 0), (-1, -1), 11), ("RIGHTPADDING", (0, 0), (-1, -1), 11), ("TOPPADDING", (0, 0), (-1, 0), 9), ("BOTTOMPADDING", (0, -1), (-1, -1), 9)]))
s += [box, Spacer(1, 6)]

# ---- 1 Introduction
s += [P("1. Introduction", h1),
      P("Named entity recognition (NER) finds spans of text that name people, places, organizations and other things, and labels their type. Systems trained on news do well because entities repeat between training and test. Social media breaks this: text is short, spelling and capitalization are unreliable, and new names appear constantly."),
      P("WNUT17 (Derczynski et al., 2017) was built to test exactly this. Entities in dev and test were chosen so that they do not appear in the training tweets. Seven teams took part and the best entity F1 was only 41.86. In Stage 1 we analysed the data and planned the preprocessing. This report covers Stage 2: what we implemented, how each step was done, and what the results show."),
      P("<b>Goals.</b> (1) Implement CRF, SVM and Bi-LSTM CRF taggers under one protocol. (2) Compare them with strict entity-level F1 and the shared-task surface F1. (3) Explain the differences with ablations, uncertainty estimates and error analysis. (4) Record every deviation from the Stage 1 plan.")]

# ---- 2 Data and protocol
s += [P("2. Dataset and Evaluation Protocol", h1), P("2.1 Data and splits", h2),
      P("We use the three official files from github.com/leondz/emerging_entities_17 without re-splitting: <b>wnut17train.conll</b> (Twitter, 2010), <b>emerging.dev.conll</b> (YouTube comments) and <b>emerging.test.annotated</b> (Reddit, StackExchange, Twitter 2017). The test set is the <b>entire</b> gold file. The raw multi-annotator file <i>emerging.test.conll</i>, the unlabeled <i>emerging.test</i> and the <i>source/</i> and <i>submissions/</i> folders are not used. Labels follow IOB2 with six types, 13 labels in total.")]
rows = [["", "Train", "Dev", "Test"], ["Source", "Twitter 2010", "YouTube", "Reddit, StackExchange, Twitter 2017"], ["Sequences", "3,394", "1,009", "1,287"], ["Tokens", "62,730", "15,733", "23,394"], ["Entities (spans)", "1,975", "836", "1,079"],
        ["Mean / max sequence length", "18.5 / 41", "15.6 / 82", "18.2 / 105"], ["Sequences with an entity", "36.2%", "62.2%", "53.5%"], ["Tokens labeled O", "95.0%", "92.1%", "92.6%"]]
for t in T6: rows.append([f"Entities: {t}", {"person": "660", "location": "548", "corporation": "221", "product": "142", "creative-work": "140", "group": "264"}[t], {"person": "470", "location": "74", "corporation": "34", "product": "114", "creative-work": "105", "group": "39"}[t], str(SUP[t])])
s += tbl("Dataset statistics, recomputed from the released files. Counts match the Stage 1 report.", rows, [5.2, 3.2, 3.2, 5.4], boldcol0=False)
s += [P("2.2 Why the task is hard", h2),
      P(f"Measured on entity mentions after lowercasing, {p1(96.3)}% of dev entities and {p1(97.8)}% of test entities never occur as an entity in train (by unique surface form, Stage 1 reports 96.6% and 98.0%). Among entity tokens, 56.0% (dev) and 58.3% (test) are out of vocabulary with respect to train. Once dev is added to training, 90% of test entity mentions are still unseen. Identity features and gazetteers built from training data therefore help very little. A model has to rely on context, word shape and pretrained representations."),
      P("2.3 Protocol", h2)]
s += B(["<b>Tune</b> hyperparameters on dev only, using entity F1. The test file is not touched during tuning.",
        "<b>Retrain</b> the chosen configuration on train plus dev (4,403 sequences) and score it <b>once</b> on the test file. This is common practice on WNUT17 because dev is closer to test in domain. Train-only scores are reported for reference.",
        "<b>Metrics.</b> Entity-level precision, recall and F1 with seqeval in strict IOB2 mode (span and type must match exactly). Surface F1 counts unique (lowercased surface form, type) pairs, as in the shared task. Token accuracy is a reference only.",
        "<b>Uncertainty.</b> 95% bootstrap intervals from 2,000 resamples of the test sentences, five-fold cross-validation grouped by document on train, and three seeds for the neural model."])

# ---- 3 Preprocessing
s += [P("3. Preprocessing, Step by Step", h1), P("3.1 Reading the files and building sequences", h2)]
s += B(["Files are read as UTF-8, two columns (token, label). A blank line ends a document. In the train file, 2,394 lines that hold only a tab end a sentence inside a tweet. Treating both as boundaries gives 3,394 train sequences from 1,000 tweets. Dev and test have one sequence per document.",
        "Tokens are used <b>exactly as released</b>. We do not re-tokenize, so labels stay aligned. Punctuation stays as separate O tokens.",
        "A validation pass finds no I- label without a matching B- or I- before it (0 violations in all splits). All counts in Table 1 are recomputed from the files by the loader."])
s += [P("3.2 Normalization", h2), P("A normalized copy of each token is used for embedding lookup and for lexical features. The original token is always kept, so capitalization and shape features are not lost. The steps run in this order:")]
s += B(["HTML entities are unescaped (for example <i>&amp;amp;</i> becomes <i>&amp;</i>), then Unicode is normalized with NFKC.",
        "Tokens starting with <i>http://</i>, <i>https://</i> or <i>www.</i> become &lt;URL&gt;. Tokens that start with @ (longer than one character) become &lt;USER&gt;.",
        "Every digit becomes 0 (2017 becomes 0000). A character repeated four or more times is cut to three (soooooo becomes sooo). The result is lowercased."])
s += [P("<b>Not done, on purpose:</b> stemming, lemmatization, stop-word removal, spelling correction, and lowercasing of the original token. Words such as <i>of</i> and <i>the</i> can be part of an entity, and shape carries information."),
      P("3.3 Pretrained embedding lookup", h2),
      P(f"GloVe Twitter 200d (glove.twitter.27B, trained on 2 billion tweets) is looked up with an ordered list of keys: the lowercase original token, then the normalized form, then the special keys &lt;url&gt;, &lt;user&gt; and &lt;number&gt; where they apply, and the hashtag body without the # sign. Of the 15,555 distinct lookup keys needed by the three splits, 11,577 exist in GloVe. Token coverage is {cov('train', 0)}% (train), {cov('dev', 0)}% (dev) and {cov('test', 0)}% (test); for entity tokens it is {cov('train', 1)}%, {cov('dev', 1)}% and {cov('test', 1)}%. Tokens without a vector get zeros.")]
s += [P("3.4 Feature dictionaries for the CRF and SVM", h2), P("Each token becomes a dictionary of features (Table 2). Both models use the same dictionaries.")]
rows = [["Group", "Features"], ["Lexical", "lowercase token, normalized form, English stop-word flag (NLTK list)"], ["Context (window +-2)", "lowercase words at -2, -1, +1, +2 (BOS and EOS markers at edges); title-case and upper-case flags at -1 and +1; left and right bigrams"],
        ["Shape", "collapsed shape pattern (Xxxx, dd), is_title, is_upper, has_digit, has_hyphen"], ["Affixes", "prefixes and suffixes of 1 to 4 characters (lowercase), token length (capped at 15)"],
        ["Social media", "is_url, is_mention, is_hashtag, is_emoji"], ["POS tags", "NLTK averaged-perceptron tag (Penn Treebank) of the token and of the tokens at -1 and +1"],
        ["Cluster IDs", "K-Means (k = 500, seed 42) cluster of the GloVe vector of the token and of the tokens at -1 and +1; fitted on the 100,000 most frequent GloVe Twitter words"],
        ["Dense vector (SVM only)", "200-d GloVe vector of the token, scaled by 0.3, appended to the sparse features"]]
s += tbl("Features used by the CRF and SVM.", rows, [3.6, 13.4], align=["l", "l"], boldcol0=True)
s += [P("<b>SVM vectorization.</b> Dictionaries are turned into a sparse matrix with DictVectorizer. Features seen fewer than three times in the training data are dropped, which leaves 28,436 features for the train-only fit. This pruning was needed to make training feasible on one CPU core."),
      P("3.5 Inputs of the Bi-LSTM CRF", h2)]
s += B(["<b>Word:</b> the frozen GloVe vector found by the lookup above; words without a vector share one all-zero vector.",
        "<b>Characters:</b> the first 20 characters of the original token, mapped to ids from a character vocabulary built from the training data (92 characters on train). Unknown characters share one id.",
        "<b>Capitalization class:</b> one of 10 ids (URL, mention, hashtag, contains a digit, no letters, all lowercase, all uppercase, title case, mixed, padding).",
        "<b>Batching:</b> sequences are grouped into length buckets (8, 16, 24, 32, 48, 64, 128). The longest test sequence has 105 tokens, so nothing is truncated."])
s += [P("<b>Labels.</b> 13 IOB2 labels. For the BIOES experiment, labels are converted to 25 tags for training and converted back to BIO before scoring. <b>Transparency note:</b> the set of GloVe vectors loaded includes words from all three splits. No labels are used, the vectors are frozen, and nothing is trained on test text.")]
s += [fig("f_pipeline", 6.6, "Overall pipeline from the raw CoNLL files to evaluation.")]

# ---- 4 Models
s += [P("4. Models", h1), P("4.1 CRF", h2),
      P("A linear-chain CRF (sklearn-crfsuite, L-BFGS, all transitions allowed) with the features of Table 2. It scores whole label sequences, so it learns that I-person must follow B-person or I-person. Regularization strengths c1 (L1) and c2 (L2) were searched on dev."),
      P("4.2 SVM", h2),
      P("A one-vs-rest LinearSVC that labels every token independently, with class weights set to <i>balanced</i> because 92 to 95% of tokens are O. It has no view of neighbouring labels, so its output can contain an I- tag with no B- before it. Under strict scoring such tags are ignored (Section 6.4)."),
      P("4.3 Bi-LSTM CRF", h2),
      P("Each token is represented by its frozen GloVe vector, a character-CNN vector and a capitalization embedding (258 dimensions in total). A one-layer bidirectional LSTM reads the sequence, a linear layer gives label scores, and a CRF layer decodes the best valid label sequence with Viterbi. Training minimizes the CRF negative log-likelihood with Adam."),
      P("The model is implemented in JAX because PyTorch could not be installed in the working environment. The CRF log-partition and Viterbi path were checked against brute-force enumeration on a small case and match exactly.")]
s += [fig("f_architecture", 6.6, "Bi-LSTM CRF architecture.")]
rows = [["", "CRF", "SVM", "Bi-LSTM CRF"], ["Algorithm", "L-BFGS CRF, all transitions", "LinearSVC, one-vs-rest", "Bi-LSTM, 128 units per direction, CRF layer"],
        ["Key setting", f"c1 = {C2['best']['c1']}, c2 = {C2['best']['c2']}, 300 iterations", "C = 0.03, class_weight balanced, 100 iterations", "dropout 0.5, Adam lr 0.001, batch 32, clip 5"],
        ["Input", "feature dictionaries (all groups)", "sparse features (28,436) + GloVe vector", "GloVe 200d frozen, char-CNN (50 filters), capitalization (8d)"],
        ["Decoding", "Viterbi over transitions", "independent per token", "Viterbi over CRF transitions"], ["Final training data", "train + dev", "train + dev", f"train + dev, {bi['epochs']} epochs, seeds 42, 43, 44"]]
s += tbl("Configuration of the three models as used for the final results.", rows, [2.7, 4.6, 4.7, 5.0], align=["l", "l", "l", "l"], boldcol0=True)

# ---- 5 Step by step
s += [P("5. Experiments, Step by Step", h1), P("Table 4 lists what was done, in order, and what came out of each step.")]
rows = [["#", "What was done", "Outcome"],
        ["1", "Set up the environment: Python 3.12 on one CPU core (about 4 GB RAM). Installed scikit-learn, sklearn-crfsuite, seqeval, NLTK, JAX and optax. PyTorch did not fit in the free disk space, so JAX was used.", "Toolchain fixed (versions in Appendix C)."],
        ["2", "Cloned the WNUT17 repository and recomputed every Stage 1 statistic from the raw files.", "All counts reproduced. Unseen entity mentions: 96.3% dev, 97.8% test."],
        ["3", "Downloaded GloVe Twitter 200d (795 MB compressed) and kept the vectors needed for lookup and clustering.", "11,577 of 15,555 lookup keys found."],
        ["4", "Normalized tokens, tagged POS with NLTK, fitted K-Means (k = 500) on GloVe, built feature dictionaries.", "Features cached for reuse."],
        ["5", "CRF search on dev. First grid (3 x 3, 100 iterations): best c1 0.3, c2 0.03, dev F1 23.5, on the edge of the grid. Extended to 18 configurations, 300 iterations, with and without cluster features.", f"Best: full features, c1 0.3, c2 0.003, dev F1 {p1(C2['best']['dev_F1'])}."],
        ["6", "SVM search on dev. A first run on the full feature space did not finish within several minutes, so rare features were dropped and iterations capped at 100 (300 iterations gave the same dev F1 in a check at C = 0.1). Grid C in {0.01, 0.03, 0.1, 0.3}.", f"Best C = 0.03. Class weights, GloVe and dropping BIO repair were tested (Section 6.4)."],
        ["7", "Bi-LSTM CRF: implemented LSTM, CRF loss and Viterbi in JAX; verified against brute force. Trained with early stopping on dev (patience 8, at most 35 epochs), BIO labels.", f"Best epoch {bi['best_ep_bio']}, dev F1 {p1(bi['dev_bio'])}, test F1 {p1(bi['trainonly_bio'])} (train only)."],
        ["8", f"Retrained the Bi-LSTM CRF on train + dev for {bi['epochs']} epochs with seeds 42, 43 and 44. Repeated the search with BIOES labels (train only).", f"Test F1 {', '.join(p1(v) for v in bi['seeds'])}. BIOES: best epoch {bi['bioes_ep']}, test F1 {p1(bi['bioes_test'])}."],
        ["9", "Retrained CRF and SVM on train + dev. Scored all three once on the full test file (1,287 sequences, 1,079 entities). Bootstrapped test sentences 2,000 times.", "Tables 5 and 7."],
        ["10", "Five-fold cross-validation grouped by document on train with the final configurations.", "Table 8." if CV else "Table 8."],
        ["11", "Analysis: per-type scores, error categories, seen versus unseen entities, capitalization, length, confusion matrix, examples.", "Section 6."]]
s += tbl("Chronological log of the work.", rows, [0.8, 11.2, 5.0], align=["c", "l", "l"])

# ---- 6 Results
s += [P("6. Results", h1), P("6.1 Main comparison", h2)]
rows = [["Model", "Precision", "Recall", "F1 entity", "F1 surface", "Token acc."],
        ["CRF", p1(crf["P"]), p1(crf["R"]), p1(crf["F1"]), p1(crf["surf"]), p1(crf["tokacc"])], ["SVM", p1(svm["P"]), p1(svm["R"]), p1(svm["F1"]), p1(svm["surf"]), p1(svm["tokacc"])],
        ["Bi-LSTM CRF (mean of 3 seeds)", p1(bi["P"]), p1(bi["R"]), p1(bi["F1"]), p1(bi["surf"]), p1(bi["tokacc"])],
        ["Reference: predict O everywhere", "-", "-", "-", "-", "92.6"], ["Reference: best shared-task system (UH-RiTUAL)", "-", "-", "41.86", "40.24", "-"]]
s += tbl("Test results in percent (models trained on train + dev, scored once on the full gold test file).", rows, [6.6, 2.0, 1.8, 2.2, 2.4, 2.0], hi=(3,))
s += [P(f"The Bi-LSTM CRF is clearly best at {p1(bi['F1'])} F1 (seeds: {', '.join(p1(v) for v in bi['seeds'])}). The SVM is second ({p1(svm['F1'])}) and the CRF third ({p1(crf['F1'])}). Precision and recall pull in different directions: the CRF is the most cautious (P {p1(crf['P'])}, R {p1(crf['R'])}), the SVM the most generous (P {p1(svm['P'])}, R {p1(svm['R'])}), and the Bi-LSTM CRF is better on both (P {p1(bi['P'])}, R {p1(bi['R'])})."),
      P("Token accuracy hides these differences. Predicting O for every token already gives 92.6%, and the CRF and SVM are almost at that level (92.8 and 92.2). All comparisons therefore use entity F1."),
      fig("f_main", 6.2, "Precision, recall and F1 on the test set.")]
rows = [["Model", "Dev F1 (train only)", "Test F1 (train only)", "Test F1 (train + dev)"],
        ["CRF", p1(crf["dev"]), p1(crf["trainonly"]), p1(crf["F1"])], ["SVM", p1(dev_norep), p1(svt_no) if svt_no is not None else "n/a", p1(svm["F1"])], ["Bi-LSTM CRF (BIO)", p1(bi["dev_bio"]), p1(bi["trainonly_bio"]), f"{p1(bi['F1'])} (mean of 3 seeds)"]]
s += tbl("Scores by training data. Adding dev data to training raises test F1 for every model.", rows, [5.0, 3.9, 4.0, 4.1])
s += [P("The SVM dev score is for the final setting (C = 0.03, no BIO repair). The value of C was picked with BIO repair switched on (dev F1 41.4), and repair was dropped afterwards because it lowered dev F1 (Section 6.4).", note)]

s += [P("6.2 How reliable are the differences?", h2),
      P(f"Resampling the 1,287 test sentences gives 95% intervals of about 3 F1 points for each model (Table 7, seed 42 for the Bi-LSTM CRF). The Bi-LSTM CRF lead over the SVM (+{BS['Bi-LSTM CRF minus SVM'][0]}, interval {BS['Bi-LSTM CRF minus SVM'][1]} to {BS['Bi-LSTM CRF minus SVM'][2]}) is far outside the noise. The SVM lead over the CRF (+{BS['SVM minus CRF'][0]}, interval {BS['SVM minus CRF'][1]} to {BS['SVM minus CRF'][2]}) is positive but small.")]
rows = [["Quantity", "Point estimate", "95% interval"], ["CRF F1", p1(crf["F1"]), f"{BS['CRF'][0]} to {BS['CRF'][1]}"], ["SVM F1", p1(svm["F1"]), f"{BS['SVM'][0]} to {BS['SVM'][1]}"], ["Bi-LSTM CRF F1 (seed 42)", "42.9", f"{BS['Bi-LSTM CRF'][0]} to {BS['Bi-LSTM CRF'][1]}"],
        ["SVM minus CRF", f"+{BS['SVM minus CRF'][0]}", f"{BS['SVM minus CRF'][1]} to {BS['SVM minus CRF'][2]}"], ["Bi-LSTM CRF minus SVM", f"+{BS['Bi-LSTM CRF minus SVM'][0]}", f"{BS['Bi-LSTM CRF minus SVM'][1]} to {BS['Bi-LSTM CRF minus SVM'][2]}"]]
s += tbl("Bootstrap intervals (2,000 resamples of test sentences; paired for the differences).", rows, [7.0, 4.5, 5.5], hi=(4, 5))
if CV:
    rows = [["Model", "Fold F1 values", "Mean", "SD"], ["CRF", ", ".join(p1(v) for v in CV["crf"]), p1(CV["crf_mean"]), p1(CV["crf_std"])], ["SVM", ", ".join(p1(v) for v in CV["svm"]), p1(CV["svm_mean"]), p1(CV["svm_std"])]]
    s += tbl("Five-fold cross-validation grouped by document, train only, final configurations.", rows, [2.6, 9.4, 2.5, 2.5])
    s += [P("Cross-validation scores are higher than test scores because every fold comes from one domain (Twitter 2010) and shares entities across folds. They show that the settings are stable, not what to expect on new domains.", note)]

s += [P("6.3 Results by entity type", h2)]
rows = [["Type", "Test support", "CRF", "SVM", "Bi-LSTM CRF"]] + [[t, str(SUP[t]), p1(crf["types"][t]), p1(svm["types"][t]), p1(bi["types"][t])] for t in T6]
s += tbl("F1 by entity type on the test set (Bi-LSTM CRF is the mean of three seeds).", rows, [4.2, 3.0, 3.2, 3.2, 3.4])
s += [fig("f_types", 6.2, "F1 by entity type."),
      P(f"Person and location are the easiest types for every model. Corporation, product, creative-work and group stay below 23 F1 for all three, and the CRF and SVM are near 10 on most of them. These types have few training examples and overlapping definitions (corporation, group and product), and creative-work titles are long and free-form. The ordering matches what the Stage 1 analysis predicted.")]

s += [P("6.4 Ablations and design decisions", h2), P("<b>CRF features.</b> Features were added cumulatively and scored on dev with the final CRF settings (Figure 5, left).")]
s += B([f"Word identity alone gives {p1(ab[0]['F1'])} F1: memorizing names does not work for unseen entities.",
        f"Shape, affixes and social-media flags add {p1(d(1))}. Context words and bigrams add {p1(d(2))}. POS tags add {p1(d(3))}.",
        f"Cluster IDs add {p1(d(4))} in the final setting, but the sign flips at another setting ({clu_flip:+.1f} at c1 = 0.3, c2 = 0.03). The effect is small and not stable."])
s += [P(f"<b>SVM.</b> On dev (Figure 5, right), removing class weights raises precision to {p1(svnw['P'])} but recall collapses to {p1(svnw['R'])}, and F1 falls from {p1(svf['F1'])} to {p1(svnw['F1'])}. Removing the GloVe vector drops F1 to {p1(svng['F1'])}, a loss of {p1(svf['F1'] - svng['F1'])} points. Both ingredients matter."),
      fig("f_ablation", 6.6, "Left: cumulative CRF feature ablation on dev. Right: SVM ablations on dev; the amber bar is the setting used to select C.")]
rows = [["", "With BIO repair", "Without BIO repair"], ["Dev F1", p1(dev_rep), p1(dev_norep)], ["Test F1", p1(100 * SF["test_repair"]["entity"]["f1"]), p1(svm["F1"])],
        ["Test precision", p1(100 * SF["test_repair"]["entity"]["precision"]), p1(svm["P"])], ["Test recall", p1(100 * SF["test_repair"]["entity"]["recall"]), p1(svm["R"])]]
s += tbl("Effect of the BIO repair step planned in Stage 1 (SVM, C = 0.03).", rows, [5.0, 5.5, 6.5])
s += [P(f"<b>BIO repair.</b> Stage 1 planned to turn every orphan I- tag into a B- tag. On dev and on test this lowered F1. The raw SVM output has {svm['invalid_I']} orphan I- tags on the test set. Strict scoring ignores them, while repair turns them into new entities, most of which are wrong: precision falls from {p1(svm['P'])} to {p1(100 * SF['test_repair']['entity']['precision'])} for a recall gain of under 2 points. The decision to drop repair was taken on dev, before looking at test."),
      P(f"<b>BIO versus BIOES</b> (Bi-LSTM CRF, train only, one seed each). Dev F1 was {p1(bi['dev_bio'])} for BIO and {p1(bi['bioes_dev'])} for BIOES. Test F1 was {p1(bi['trainonly_bio'])} and {p1(bi['bioes_test'])}. The 1.6-point test difference is smaller than the seed-to-seed spread of the final runs (about 2 points from lowest to highest), so we keep BIO and do not claim a BIOES gain."),
      fig("f_curves", 6.2, "Dev F1 per epoch for the Bi-LSTM CRF. Both label schemes rise quickly to about epoch 10 and then move within a band of about 2 points.")]

s += [P("6.5 Error analysis", h2),
      P("Every gold test entity was placed in one category: found exactly, found with a wrong boundary or type, or missed. Predictions that overlap no gold entity are counted separately as spurious.")]
rows = [["Category", "CRF", "SVM", "Bi-LSTM CRF"], ["Correct (exact span and type)"] + [str(E[m]["benar"]) for m in M3], ["Wrong boundary only"] + [str(E[m]["salah_batas"]) for m in M3], ["Wrong type only"] + [str(E[m]["salah_tipe"]) for m in M3],
        ["Wrong boundary and type"] + [str(E[m]["salah_batas_dan_tipe"]) for m in M3], ["Missed entirely"] + [str(E[m]["terlewat"]) for m in M3], ["Spurious predictions"] + [str(E[m]["prediksi_palsu"]) for m in M3]]
s += tbl("Error categories for the 1,079 gold test entities (Bi-LSTM CRF: seed 42).", rows, [7.0, 3.2, 3.2, 3.6])
s += [fig("f_errors", 6.2, "Share of gold entities by outcome; red text shows spurious predictions."),
      P(f"The three models fail in different ways. The CRF misses {E['CRF']['terlewat']} entities ({100 * E['CRF']['terlewat'] / 1079:.0f}%) and adds few spurious ones. The SVM gets the most entities exactly right ({E['SVM']['benar']}), but {wrong(E['SVM'])} more are found with the wrong boundary or type, and it adds {E['SVM']['prediksi_palsu']} spurious predictions. The Bi-LSTM CRF is precise ({E['Bi-LSTM CRF']['prediksi_palsu']} spurious) but misses {E['Bi-LSTM CRF']['terlewat']} ({100 * E['Bi-LSTM CRF']['terlewat'] / 1079:.0f}%). Low recall is the main problem for the CRF and Bi-LSTM CRF, while boundary and type mistakes matter as much as misses for the SVM."),
      P("6.6 Where the models succeed and fail", h2),
      P("Recall was split by properties of the gold entity (Figure 8). Seen means the lowercased surface form occurs as an entity in train or dev.")]
G = S["analysis"]
s += [fig("f_analysis", 6.6, "Recall of gold test entities by novelty, capitalization and length (group sizes in brackets)."),
      P(f"<b>Novelty.</b> Only 109 of 1,079 test entities were seen in train or dev. Recall on them is {p1(G['CRF']['seen']['recall'])} (CRF), {p1(G['SVM']['seen']['recall'])} (SVM) and {p1(G['Bi-LSTM CRF']['seen']['recall'])} (Bi-LSTM CRF). On the 970 unseen entities it is {p1(G['CRF']['unseen']['recall'])}, {p1(G['SVM']['unseen']['recall'])} and {p1(G['Bi-LSTM CRF']['unseen']['recall'])}. Almost all of the gap to a good score comes from unseen entities."),
      P(f"<b>Capitalization.</b> Entities whose tokens all start with a capital letter reach {p1(G['SVM']['kapital']['recall'])} recall (SVM) and {p1(G['Bi-LSTM CRF']['kapital']['recall'])} (Bi-LSTM CRF). Entities with some lowercase token, about a quarter of the test set, stay at {p1(G['CRF']['ada huruf kecil']['recall'])} to {p1(G['SVM']['ada huruf kecil']['recall'])}."),
      P(f"<b>Length.</b> On entities of two or more tokens, the Bi-LSTM CRF recovers {p1(G['Bi-LSTM CRF']['>=2 token']['recall'])} of the entities, against {p1(G['CRF']['>=2 token']['recall'])} (CRF) and {p1(G['SVM']['>=2 token']['recall'])} (SVM). Sequence-level decoding on top of contextual features helps most for multi-word names."),
      fig("f_confusion", 6.6, "Gold type against predicted type. Columns 'wrong boundary' and 'missed' hold entities that did not match a predicted span exactly.")]
ex = [["Sentence (test index)", "Gold", "CRF", "SVM", "Bi-LSTM CRF"],
      ["&amp;gt; Umm , where are all these alleged criticisms ? (27)", "no entity", "Umm = location", "no entity", "no entity"],
      ["Visuals of the avalanche site in Gurez sector . (6)", "Gurez sector = location", "Gurez = location", "Gurez = product", "Gurez sector = location"],
      ["... take back what you said at Glenn Greenwald in the last thread . (50)", "Glenn Greenwald = person", "Glenn Greenwald = location", "Glenn = person; Greenwald = location", "Glenn Greenwald = person"],
      ["( Source : ANI ) Visuals of the avalanche site in Gurez sector . (7)", "ANI = group; Gurez sector = location", "Gurez = location", "ANI = person; Gurez = product", "Gurez = location"],
      ["Living Computer Museum + Labs , Skyview Observatory , Smith Tower and Observation Deck , (54)", "4 locations", "Skyview Observatory = corporation", "8 fragments, mixed types", "3 exact; 'Living Computer Museum' short by '+ Labs'"]]
s += tbl("Examples from the test set (Bi-LSTM CRF: seed 42). Row 1 is the Reddit sentence used in Stage 1.", ex, [5.0, 3.0, 2.9, 3.0, 3.1], align=["l", "l", "l", "l", "l"])

# ---- 7 Discussion
s += [P("7. Discussion", h1),
      P("<b>Why the ranking?</b> The models differ in how much pretrained knowledge reaches them. The SVM and Bi-LSTM CRF see dense GloVe vectors. The CRF sees GloVe only through 500 cluster IDs. In the SVM, dense vectors are worth 13.9 dev F1 points, and the CRF has clearly lower recall on unseen entities (16%, against 28% for the other two). This fits the idea that dense representations matter most when 98% of entities are new. We did not test it directly, for example with a CRF that gets dense vectors or a Bi-LSTM CRF without GloVe, so it remains a hypothesis."),
      P("<b>Precision and recall.</b> Class weighting pushes the SVM to predict entities, which raises recall and adds many spurious and fragmented spans. The CRF and Bi-LSTM CRF are more cautious. The Bi-LSTM CRF combines the highest precision with the highest recall, which suggests that its context and character features are not just trading one for the other."),
      P(f"<b>Domain shift.</b> Dev is YouTube text with 56% person entities, the easiest type. Test has 15% group and 14% location. This is one reason why dev scores are higher than test scores (SVM {p1(dev_norep)} on dev against {p1(svm['F1'])} on test; Bi-LSTM CRF {p1(bi['dev_bio'])} against {p1(bi['F1'])}). The dev score comes from a train-only model and the test score from a train + dev model, so the two are not a clean measure of shift."),
      P("<b>Comparison with the shared task.</b> The Bi-LSTM CRF (41.9) is on par with the best participant (41.86), and the SVM (31.3) and CRF (27.8) sit in the lower and middle part of the reported range (26.30 to 41.86). The comparison is approximate. The released test set holds 1,079 entities after annotation fixes (the paper counts 1,040), our surface F1 lowercases the strings, and our final models use dev for training.")]

# ---- 8 Limitations
s += [P("8. Limitations", h1)]
s += B(["The CRF and SVM were run once. Their runs are deterministic given the data, so uncertainty is described with bootstrap intervals over test sentences, not seeds. The Bi-LSTM CRF has three seeds.",
        "The Bi-LSTM CRF was tuned only through the number of epochs. Learning rate, dropout, hidden size and embedding options were fixed. No experiment removes GloVe from the Bi-LSTM CRF.",
        "The SVM uses only the token's own GloVe vector, prunes rare features, and was capped at 100 iterations. Its C was chosen on dev with BIO repair on, and repair was then dropped.",
        "BIOES was tested with a single train-only run per scheme, so the comparison is inconclusive.",
        "The bootstrap for the Bi-LSTM CRF uses the seed-42 predictions only. Seed variation (SD 0.84) adds to that uncertainty.",
        "GloVe lookups cover words from all splits (frozen vectors, no labels). Surface F1 lowercases strings and may differ slightly from the official script.",
        "The neural model is written in JAX, which most course material does not use. Numbers from a PyTorch port may differ slightly through initialization and dropout."])

# ---- 9 Conclusion
s += [P("9. Conclusion and Future Work", h1),
      P(f"We implemented and compared a CRF, an SVM and a Bi-LSTM CRF on WNUT17 with one protocol and full test-set scoring. The Bi-LSTM CRF with frozen GloVe Twitter vectors, character features and capitalization embeddings reaches {p1(bi['F1'])} entity F1, on par with the best shared-task system. The SVM ({p1(svm['F1'])}) and CRF ({p1(crf['F1'])}) fall well below it. Entities that never appeared in training, entities with lowercase tokens, and the types corporation, product, creative-work and group remain the hard cases for every model."),
      P("<b>Next steps</b>")]
s += B(["Test the explanation directly: a Bi-LSTM CRF without pretrained vectors, and a CRF with dense GloVe features.",
        "Tune the Bi-LSTM CRF (learning rate, dropout, hidden size), train the CRF and SVM with the fuller Stage 1 feature list, and repeat BIOES with several seeds.",
        "Give unknown words a trained vector, for example by replacing rare training words with an unknown token during training (word dropout).",
        "Try contextual encoders such as BERTweet, and study the four weak types with targeted data or weak supervision."])

# ---- References
s += [P("References", h1)]
refs = ["Bird, S., Klein, E. and Loper, E. (2009). Natural Language Processing with Python. O'Reilly.",
        "Bradbury, J. et al. (2018). JAX: composable transformations of Python+NumPy programs. github.com/google/jax.",
        "Cortes, C. and Vapnik, V. (1995). Support-vector networks. Machine Learning, 20(3), 273-297.",
        "Derczynski, L., Nichols, E., van Erp, M. and Limsopatham, N. (2017). Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition. Proc. 3rd Workshop on Noisy User-generated Text, 140-147.",
        "Efron, B. and Tibshirani, R. (1993). An Introduction to the Bootstrap. Chapman and Hall.",
        "Huang, Z., Xu, W. and Yu, K. (2015). Bidirectional LSTM-CRF models for sequence tagging. arXiv:1508.01991.",
        "Kingma, D. and Ba, J. (2015). Adam: a method for stochastic optimization. ICLR.",
        "Lafferty, J., McCallum, A. and Pereira, F. (2001). Conditional random fields: probabilistic models for segmenting and labeling sequence data. ICML, 282-289.",
        "Lample, G., Ballesteros, M., Subramanian, S., Kawakami, K. and Dyer, C. (2016). Neural architectures for named entity recognition. NAACL-HLT, 260-270.",
        "Ma, X. and Hovy, E. (2016). End-to-end sequence labeling via bi-directional LSTM-CNNs-CRF. ACL, 1064-1074.",
        "Nakayama, H. (2018). seqeval: a Python framework for sequence labeling evaluation. github.com/chakki-works/seqeval.",
        "Okazaki, N. (2007). CRFsuite: a fast implementation of conditional random fields. www.chokkan.org/software/crfsuite.",
        "Pedregosa, F. et al. (2011). Scikit-learn: machine learning in Python. JMLR, 12, 2825-2830.",
        "Pennington, J., Socher, R. and Manning, C. (2014). GloVe: global vectors for word representation. EMNLP, 1532-1543.",
        "Ritter, A., Clark, S., Mausam and Etzioni, O. (2011). Named entity recognition in tweets: an experimental study. EMNLP, 1524-1534."]
s += [Paragraph(f"[{i + 1}] {r}", st("ref", fontSize=8.6, leading=11.8, leftIndent=18, firstLineIndent=-18, spaceAfter=3)) for i, r in enumerate(refs)]

# ---- Appendix
s += [PageBreak(), P("Appendix A. Deviations from the Stage 1 Plan", h1)]
rows = [["Item", "Stage 1 plan", "What was done", "Reason or effect"],
        ["SVM BIO repair", "Apply after prediction", "Dropped", "Lower F1 on dev (41.4 vs 44.8) and test (27.4 vs 31.3)"],
        ["SVM label history", "Try previous predicted labels as features", "Not implemented", "Time; SVM predicts tokens independently"],
        ["SVM dense features", "Token and context-window GloVe average, standardized", "Token GloVe vector only, scaled by 0.3", "Simpler; the vector still adds 13.9 dev F1"],
        ["SVM feature space", "All features", "Features seen at least 3 times (28,436)", "Full space too slow on one CPU core"],
        ["CRF/SVM feature list", "Full feature table of the Stage 1 report", "Subset: shape only for token, lowercase words for +-2 context, no is_lower, is_digit, has_apostrophe, is_punct, has_non_ascii", "Not needed for the comparison; may cost a little accuracy"],
        ["Cluster features", "K-Means k = 500 on GloVe", "Fitted on the 100,000 most frequent GloVe words", "Memory and time"],
        ["Bi-LSTM embeddings", "GloVe, fine-tuned", "GloVe, frozen", "Smaller model, no OOV drift"],
        ["Bi-LSTM extras", "Not planned", "Capitalization embedding added", "Adds an explicit shape signal"],
        ["Bi-LSTM tuning", "Learning rate, dropout, hidden size on dev", "Only the epoch count was chosen on dev", "Time; listed as a limitation"],
        ["FastText", "Alternative embedding", "Not run", "Time"], ["BIOES", "Try for Bi-LSTM CRF", "Tried once, train only", "No reliable gain shown"],
        ["Cross-validation", "5-fold on train, by document", "Done as planned with final configurations", "Table 8"], ["Framework", "Not specified", "JAX and optax", "PyTorch could not be installed in the environment"],
        ["Surface F1", "As in the shared task", "Lowercased (surface, type) pairs", "May differ slightly from the official script"]]
s += tbl("Differences between the Stage 1 plan and the implementation.", rows, [2.9, 4.0, 4.9, 5.2], align=["l", "l", "l", "l"], boldcol0=True)

s += [P("Appendix B. Full Tuning Grids (Dev Set)", h1)]
rows = [["Features", "c1", "c2", "Precision", "Recall", "F1 entity", "F1 surface"]]
for g in sorted(C2["grid"], key=lambda z: (z["fitur"] != "lengkap", z["c1"], z["c2"])): rows.append(["full" if g["fitur"] == "lengkap" else "no cluster", g["c1"], g["c2"], p1(g["P"]), p1(g["R"]), p1(g["F1"]), p1(g["surfF1"])])
best_i = [i for i, r in enumerate(rows) if r[0] == "full" and r[1] == C2["best"]["c1"] and r[2] == C2["best"]["c2"]]
s += tbl("CRF search (train only, 300 iterations). The selected row is shaded.", rows, [3.0, 1.8, 1.8, 2.6, 2.4, 2.8, 2.6], hi=tuple(best_i))
svmrows = [["C", "Dev F1 with BIO repair", "Dev F1 without BIO repair"]]
d2 = {x["C"]: x for x in SV2["dev"]}
for gg in SV["grid"] + [{"C": 0.01, "F1": SV2["dev"][0]["repair"]["F1"]}]:
    svmrows.append([gg["C"], p1(gg["F1"]), p1(d2[gg["C"]]["no_repair"]["F1"]) if gg["C"] in d2 else "not run"])
svmrows[1:] = sorted(svmrows[1:], key=lambda r: float(r[0]))
s += tbl("SVM search (train only). C = 0.03 was selected using the middle column.", svmrows, [3.0, 6.0, 6.0], hi=(2,))

s += [P("Appendix C. Reproducibility", h1),
      P("<b>Environment.</b> Python 3.12.3; numpy 2.4.4; scipy 1.17.1; scikit-learn 1.8.0; sklearn-crfsuite 0.5.0 (python-crfsuite 0.9.12); seqeval 1.2.2; nltk 3.10.3; jax and jaxlib 0.11.2; optax 0.2.8; matplotlib 3.10.8. One CPU core, about 4 GB RAM. A Bi-LSTM CRF epoch takes 12 to 20 seconds after a first epoch of about one minute (compilation). All scripts use seed 42, and the extra neural runs use 43 and 44."),
      P("<b>Scripts</b> (in the code archive, run in this order):")]
rows = [["File", "Purpose"], ["verify_stage1.py", "Recompute Stage 1 statistics and novelty numbers"], ["prep_glove.py", "Normalization and extraction of the GloVe vectors that are needed"], ["features.py, wnut_data.py, evalx.py", "Feature extraction, loader with surface F1, strict entity metrics and error categories"],
        ["run_crf.py, run_crf2.py", "CRF grid, extended grid, ablation and final test run"], ["run_svm.py, run_svm2.py, run_svm3.py, run_svm4.py", "SVM grid, BIO repair check, final model, train-only run"],
        ["bilstm_crf.py, run_bilstm.py", "Bi-LSTM CRF in JAX, early stopping, three seeds, BIOES"], ["run_cv_final.py", "Five-fold cross-validation on train with the final settings"], ["analysis.py, make_figures.py, build_report.py", "Post-hoc analysis, figures and bootstrap, this report"]]
s += tbl("Code files.", rows, [6.4, 10.6], align=["l", "l"])
s += [P("<b>Data.</b> WNUT17 from github.com/leondz/emerging_entities_17 (CC BY 4.0). GloVe Twitter 200d from the gensim-data release <i>glove-twitter-200</i>. Predictions for every model and all metrics are stored as JSON in the results folder of the archive.")]

doc.build(s)
print("built", cnt)
