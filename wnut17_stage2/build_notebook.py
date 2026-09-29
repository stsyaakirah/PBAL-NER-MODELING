"""Builds the documented Stage 2 notebook (nbformat). The notebook reloads the saved predictions so it runs in about a minute;
set RUN_TRAINING = True inside it to retrain everything."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda t: C.append(nbf.v4.new_markdown_cell(t.strip("\n")))
code = lambda t: C.append(nbf.v4.new_code_cell(t.strip("\n")))

md("""
# Stage 2: NER on WNUT17 with CRF, SVM and Bi-LSTM CRF
**Kelompok 1, Pemrosesan Bahasa Alami Lanjut, UGM.** This notebook documents and reproduces the Stage 2 pipeline:
data loading and checks, preprocessing, feature engineering, the three models, and the evaluation (precision, recall, F1, confusion matrices, error analysis, examples).

How to use it
* Default mode (`RUN_TRAINING = False`) loads the saved predictions from `results/` and recomputes every metric, table and figure.
* `RUN_TRAINING = True` retrains all models (needs the GloVe Twitter 200d file, see `prep_glove.py`; about 15 minutes for CRF and SVM plus about 20 minutes for three Bi-LSTM CRF seeds on a laptop CPU).
* The Bi-LSTM CRF is written in JAX (`bilstm_crf.py`). Everything else needs only numpy, pandas, scikit-learn, sklearn-crfsuite, seqeval, nltk and matplotlib.
""")
code("""
# ---- Setup: locate the code, the results folder and the official WNUT17 files ----
import os, sys, json, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

HERE = Path.cwd()
CODE_DIR = next(p for p in [HERE, HERE / "kode", HERE.parent / "kode"] if (p / "wnut_data.py").exists())
RESULTS = Path(os.environ["NB_RESULTS"]) if "NB_RESULTS" in os.environ else next(p for p in [HERE / "results", CODE_DIR.parent / "results", CODE_DIR / "results"] if p.exists())
WNUT_DIR = next((p for p in [HERE / "data", CODE_DIR.parent / "data", CODE_DIR / "data", Path("/home/claude/emerging_entities_17")] if (p / "wnut17train.conll").exists()), None)
assert WNUT_DIR is not None, "Put wnut17train.conll, emerging.dev.conll and emerging.test.annotated in a folder called data/"
os.environ["WNUT_DIR"] = str(WNUT_DIR)      # read by wnut_data.py
sys.path.insert(0, str(CODE_DIR)); os.chdir(CODE_DIR)   # module caches (POS tags, clusters) live in the code folder
RUN_TRAINING = False                        # True = retrain every model (slow)
print("code:", CODE_DIR, "| results:", RESULTS, "| data:", WNUT_DIR)
""")

md("""
## 1. Data loading and sanity checks
We use the three **official** files without re-splitting: `wnut17train.conll` (Twitter 2010), `emerging.dev.conll` (YouTube) and `emerging.test.annotated` (Reddit, StackExchange, Twitter 2017). 
`emerging.test.annotated` is the gold test file: the repository's `emerging.test.conll` holds raw multi-annotator labels and is **not** used.
The reader (`wnut_data.py`) ends a sequence at a blank line (document boundary) and at a tab-only line (sentence boundary inside a train tweet).
""")
code("""
from wnut_data import load
from seqeval.metrics.sequence_labeling import get_entities

splits = {s: load(s) for s in ["train", "dev", "test"]}      # each item: {"tokens": [...], "labels": [...], "doc": id}

def bio_violations(seqs):
    \"\"\"Count I-X labels that are not preceded by B-X or I-X (an invalid IOB2 sequence).\"\"\"
    bad = 0
    for x in seqs:
        prev = "O"
        for l in x["labels"]:
            if l.startswith("I-") and prev not in ("B-" + l[2:], "I-" + l[2:]): bad += 1
            prev = l
    return bad

rows = []
for s, seqs in splits.items():
    labels = [l for x in seqs for l in x["labels"]]
    rows.append(dict(split=s, sequences=len(seqs), tokens=len(labels), entities=sum(len(get_entities(x["labels"])) for x in seqs),
                     pct_O=round(100 * labels.count("O") / len(labels), 1), max_len=max(len(x["tokens"]) for x in seqs), bio_violations=bio_violations(seqs)))
stats = pd.DataFrame(rows).set_index("split"); display(stats)
# The counts must match the Stage 1 report exactly
assert stats["sequences"].tolist() == [3394, 1009, 1287] and stats["tokens"].tolist() == [62730, 15733, 23394] and stats["entities"].tolist() == [1975, 836, 1079]
assert stats["bio_violations"].sum() == 0
print("Stage 1 counts reproduced; the test file has", stats.loc["test", "entities"], "gold entities (all of them are scored).")
""")

md("""
## 2. Preprocessing
Tokens are used exactly as released (no re-tokenization, so labels stay aligned). A **normalized copy** of each token is used for embedding lookup and lexical features; the original token is kept for shape and capitalization features.
Steps in order: HTML unescape, NFKC, URL (`http`, `www`, bare `.com` style domains) to `<URL>`, `@user` to `<USER>`, hashtag `#` dropped, every digit to `0`, a character repeated 4+ times capped at 3 (digit runs are **not** capped), lowercase.
No stemming, lemmatization or stop-word removal.
""")
code("""
from prep_glove import normalize, glove_keys

examples = ["2017", "12,000", "soooooo", "!!!!!!!", "&amp;", "@paulwalk", "http://t.co/x", "amazon.com", "#Ravens", "THANK"]
display(pd.DataFrame({"token": examples, "normalized": [normalize(t) for t in examples], "GloVe lookup keys": [glove_keys(t) for t in examples]}))

# Regression tests for the three bugs fixed after the first Stage 2 run
assert normalize("2017") == "0000"          # was "000": digits became 0, then the repeat cap collapsed them
assert normalize("amazon.com") == "<URL>"   # the Stage 1 plan maps "*.com" to <URL>
assert normalize("#Ravens") == "ravens"     # the Stage 1 plan drops the '#' and keeps the word
assert normalize("soooooo") == "sooo" and normalize("&amp;") == "&" and normalize("@paulwalk") == "<USER>"
""")

md("""
## 3. Feature engineering for CRF and SVM
Each token becomes a dictionary of features: lexical (lowercase, normalized, stop-word flag), context words at -2..+2 with bigrams and BOS/EOS markers, prefixes and suffixes of 1 to 4 characters, shape, social-media flags, NLTK POS tags (token and neighbours), and K-Means cluster IDs of GloVe vectors (k = 500).
The cluster group needs the GloVe file, so this demo builds the other groups. The SVM additionally appends the token's 200-d GloVe vector (scaled by 0.3).
""")
code("""
import nltk
from features import sent_feats
try: nltk.pos_tag(["test"])
except LookupError: nltk.download("averaged_perceptron_tagger_eng", quiet=True)

sent = splits["test"][6]                                   # "Visuals of the avalanche site in Gurez sector ."
pos = [t for _, t in nltk.pos_tag(sent["tokens"])]
feats = sent_feats(sent["tokens"], pos, ["word", "shape", "affix", "social", "context", "pos"])
i = sent["tokens"].index("Gurez")
print(" ".join(sent["tokens"]), "\\ntoken:", sent["tokens"][i], "| gold label:", sent["labels"][i], "| number of features:", len(feats[i]))
display(pd.Series({k: v for k, v in feats[i].items() if k in ["w.lower", "w.norm", "w.shape", "w.title", "pre3", "suf3", "pos", "pos-1", "w-1", "w+1", "bi-1"]}).to_frame("value"))
""")

md("""
## 4. Models and training
| Model | Implementation | Final settings (chosen on dev) |
|---|---|---|
| CRF | `sklearn-crfsuite`, L-BFGS, all transitions | full features, c1 = 0.3, c2 = 0.003, 300 iterations |
| SVM | `sklearn` LinearSVC, one-vs-rest | C = 0.03, `class_weight="balanced"`, rare features dropped, GloVe vector appended, no BIO repair |
| Bi-LSTM CRF | JAX + optax (`bilstm_crf.py`) | frozen GloVe Twitter 200d + char-CNN (50 filters) + capitalization embedding, 1 Bi-LSTM layer of 128 units per direction, CRF layer, dropout 0.5, Adam 1e-3, batch 32 |

Protocol: tune on dev, retrain on train + dev, score once on the test file. Hyperparameter grids and ablations are in the report (Appendix) and in `results/*.json`.
""")
code("""
if RUN_TRAINING:
    import subprocess
    for script in ["prep_glove.py", "fix_crf.py", "fix_svm.py", "fix_bilstm.py"]:      # each script writes its metrics and predictions to results/
        print("running", script); subprocess.run([sys.executable, script], check=True)
else:
    print("RUN_TRAINING = False: using the saved predictions in", RESULTS)
""")
code("""
# The core of the Bi-LSTM CRF, printed from the source file so the notebook shows exactly what was run.
import inspect
try:
    import bilstm_crf
    for fn in [bilstm_crf.emissions, bilstm_crf.crf_nll, bilstm_crf.viterbi]: print(inspect.getsource(fn))
except ImportError as e:
    print("JAX/optax not installed here (", e, "). Open bilstm_crf.py to read the model code.")
""")

md("""
## 5. Evaluation on the test set
Strict entity-level precision, recall and F1 (`seqeval`, IOB2: span and type must match exactly), the shared-task surface F1, and token accuracy. 
The Bi-LSTM CRF predictions loaded here are seed 42; the mean over three seeds is shown below the table.
""")
code("""
from evalx import evaluate, error_breakdown, type_confusion
PRED_FILES = {"CRF": "pred_fix_crf_test.json", "SVM": "pred_fix_svm_test.json", "Bi-LSTM CRF": "pred_fix_bilstm_test.json"}
preds = {m: json.load(open(RESULTS / f)) for m, f in PRED_FILES.items()}
test = splits["test"]; MODELS = list(preds)
assert all(len(p) == len(test) and all(len(a) == len(b["tokens"]) for a, b in zip(p, test)) for p in preds.values())   # aligned with the gold file
res = {m: evaluate(test, p) for m, p in preds.items()}

main = pd.DataFrame({m: {"Precision": 100 * r["entity"]["precision"], "Recall": 100 * r["entity"]["recall"], "F1 entity": 100 * r["entity"]["f1"],
                         "F1 surface": 100 * r["surface"]["f1"], "Token accuracy": 100 * r["token_acc"]} for m, r in res.items()}).T.round(1)
display(main)
fb = RESULTS / "fix_bilstm.json"
if fb.exists():
    runs = json.load(open(fb))["B"]["runs"]; f1s = [100 * r["test"]["entity"]["f1"] for r in runs]
    print("Bi-LSTM CRF, seeds", [r["seed"] for r in runs], "F1:", [round(v, 2) for v in f1s], "| mean %.2f, SD %.2f" % (np.mean(f1s), np.std(f1s)))
""")
code("""
# Bootstrap: resample the test sentences 1000 times to see how much an F1 difference of a few points means.
def strict_spans(labels):
    \"\"\"Entities under strict IOB2: an entity starts at B-X and continues with I-X (an orphan I-X is ignored).\"\"\"
    out, i = set(), 0
    while i < len(labels):
        if labels[i].startswith("B-"):
            t, j = labels[i][2:], i + 1
            while j < len(labels) and labels[j] == "I-" + t: j += 1
            out.add((i, j - 1, t)); i = j
        else: i += 1
    return out
gold = [x["labels"] for x in test]
stat = {m: tuple(np.array(v) for v in ([len(strict_spans(g) & strict_spans(p)) for g, p in zip(gold, pr)], [len(strict_spans(p)) for p in pr], [len(strict_spans(g)) for g in gold])) for m, pr in preds.items()}
rng = np.random.RandomState(0); idx = rng.randint(0, len(gold), (1000, len(gold)))
boot = {m: np.array([2 * s[0][i].sum() / (s[1][i].sum() + s[2][i].sum()) for i in idx]) * 100 for m, s in stat.items()}
for m, v in boot.items(): print(f"{m:12s} F1 95% interval: {np.percentile(v, 2.5):.1f} to {np.percentile(v, 97.5):.1f}")
d = boot["SVM"] - boot["CRF"]; print("SVM minus CRF: %.1f (95%% interval %.1f to %.1f)" % (d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)))
""")
code("""
# F1 by entity type and error categories (every gold entity is exactly one of: correct, wrong boundary/type, missed)
T6 = ["person", "location", "corporation", "product", "creative-work", "group"]
display(pd.DataFrame({m: {t: round(100 * r["per_type"][t]["f1-score"], 1) for t in T6} for m, r in res.items()}))
errs = pd.DataFrame({m: error_breakdown(test, p) for m, p in preds.items()}).rename(index={"benar": "correct", "salah_batas": "wrong boundary", "salah_tipe": "wrong type", "salah_batas_dan_tipe": "wrong boundary and type", "terlewat": "missed", "prediksi_palsu": "spurious predictions"})
display(errs)
""")

md("""
### Confusion matrix for each model
Rows are the gold type; columns are the predicted type when the span is exactly right, plus **wrong boundary** (the prediction only overlaps the gold span) and **missed** (no overlap). 
The last row, **SPURIOUS**, counts predicted spans that overlap no gold entity, by predicted type.
""")
code("""
fig, axes = plt.subplots(1, 3, figsize=(17, 5.2), sharey=True)
for ax, m in zip(axes, MODELS):
    M, rows_, cols_ = type_confusion(test, preds[m])
    ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1)); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.55 * M.max() else "#1F2937")
    ax.set_xticks(range(len(cols_))); ax.set_xticklabels(cols_, rotation=55, ha="right", fontsize=8); ax.set_title(m)
    ax.axhline(5.5, color="#5B6B7C", lw=1)
axes[0].set_yticks(range(len(rows_))); axes[0].set_yticklabels(rows_); axes[0].set_ylabel("gold type"); plt.tight_layout(); plt.show()
""")

md("""
### Where the models succeed and fail
Recall of gold entities by novelty ("seen" means the lowercased surface form is an entity in train or dev), capitalization and length.
""")
code("""
seen = {" ".join(x["tokens"][b:e + 1]).lower() for s in ("train", "dev") for x in splits[s] for _, b, e in get_entities(x["labels"])}
groups = {"seen in train+dev": [], "unseen": [], "all tokens capitalized": [], "some lowercase token": [], "1 token": [], "2+ tokens": []}
for i, x in enumerate(test):
    for (t, b, e) in get_entities(x["labels"]):
        toks = x["tokens"][b:e + 1]; surf = " ".join(toks).lower(); cap = all(w[:1].isupper() or not w[:1].isalpha() for w in toks)
        for g in (("seen in train+dev" if surf in seen else "unseen"), ("all tokens capitalized" if cap else "some lowercase token"), ("1 token" if e == b else "2+ tokens")):
            groups[g].append((i, (t, b, e)))
rec = {m: {g: round(100 * np.mean([ent in set(get_entities(preds[m][i])) for i, ent in items]), 1) for g, items in groups.items()} for m in MODELS}
display(pd.concat([pd.DataFrame(rec), pd.Series({g: len(v) for g, v in groups.items()}, name="n gold")], axis=1))
""")

md("### Examples of correct and incorrect predictions (test set)")
code("""
def spans(x, labels): return [(t, " ".join(x["tokens"][b:e + 1])) for t, b, e in get_entities(labels)]
def exact(i, m): return set(get_entities(preds[m][i])) == set(get_entities(test[i]["labels"]))
short = [i for i, x in enumerate(test) if 0 < len(get_entities(x["labels"])) and len(x["tokens"]) <= 18]
for m in MODELS:
    ok = [i for i in short if exact(i, m)][:2]; bad = [i for i in short if not exact(i, m)][:2]
    for tag, ids in (("CORRECT", ok), ("INCORRECT", bad)):
        for i in ids:
            print(f"[{m} | {tag}] {' '.join(test[i]['tokens'])}\\n    gold: {spans(test[i], test[i]['labels'])}\\n    pred: {spans(test[i], preds[m][i])}")
""")

md("""
## 6. Summary
* The Bi-LSTM CRF with frozen GloVe Twitter vectors, a character CNN and capitalization embeddings is clearly best; the SVM and CRF are far behind, and their gap to each other is small.
* Entities that never appeared in training, entities with lowercase tokens, and the types corporation, product, creative-work and group are the hard cases for every model.
* Deviations from the Stage 1 plan and the three preprocessing bugs fixed after the first run are listed in the report (Appendix A and B).
""")
nb["cells"] = C; nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "Stage2_NER_WNUT17_Kelompok1.ipynb"); print("notebook written:", len(C), "cells")
