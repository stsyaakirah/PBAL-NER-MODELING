"""Section B: CRF and SVM on the pooled 80/10/10 split (data_partB), seed 42.

Hyperparameters are REUSED from Section A (trainonly_models.py), not re-searched:
  CRF  sklearn-crfsuite lbfgs, c1=0.3, c2=0.003, max_iterations=300, all_possible_transitions
  SVM  LinearSVC C=0.03, class_weight balanced, features seen <3 dropped, GloVe*0.3 appended, no BIO repair
Only the data split changes. New test split is scored once; nothing is tuned on it.
"""
import os, json, numpy as np, scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
import sklearn_crfsuite
from wnut_data import read_conll
from features import sent_feats, get_pos, glove_vec_table, bio_repair, GROUPS
from evalx import evaluate, error_breakdown

os.makedirs("results_partB", exist_ok=True)
D = {sp: read_conll(f"data_partB/{sp}.conll") for sp in ("train", "dev", "test")}
P = {sp: get_pos(f"pB_{sp}", D[sp]) for sp in D}          # fresh POS caches, distinct from Section A
LAB = {sp: [x["labels"] for x in D[sp]] for sp in D}
FLAT = lambda q: [l for s in q for l in s]
brief = lambda r: {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2),
                   "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}

# ---------------- CRF ----------------
Fc = {sp: [sent_feats(x["tokens"], p, GROUPS) for x, p in zip(D[sp], P[sp])] for sp in D}
crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.3, c2=0.003, max_iterations=300, all_possible_transitions=True)
crf.fit(Fc["train"], LAB["train"])
pl = lambda X: [list(map(str, q)) for q in crf.predict(X)]
crf_dev, crf_test = pl(Fc["dev"]), pl(Fc["test"])
Rc = {"dev": evaluate(D["dev"], crf_dev), "test": evaluate(D["test"], crf_test),
      "test_errors": error_breakdown(D["test"], crf_test)}
json.dump(crf_test, open("results_partB/pred_partB_crf_test.json", "w"))
json.dump(crf_dev, open("results_partB/pred_partB_crf_dev.json", "w"))
json.dump(Rc, open("results_partB/partB_crf.json", "w"), indent=1, default=float)
print("CRF  dev", brief(Rc["dev"]), "| test", brief(Rc["test"]), flush=True)

# ---------------- SVM ----------------
DICTS = {sp: [f for x, p in zip(D[sp], P[sp]) for f in sent_feats(x["tokens"], p, GROUPS)] for sp in D}
GLV = {sp: np.vstack(glove_vec_table(D[sp])) for sp in D}
LENS = {sp: [len(x["tokens"]) for x in D[sp]] for sp in D}
key = lambda k, v: f"{k}={v}" if not isinstance(v, (float, bool)) else k
def unflat(flat, lens):
    out, i = [], 0
    for n in lens: out.append(list(flat[i:i + n])); i += n
    return out
tr_d = DICTS["train"]
cnt = Counter(key(k, v) for d in tr_d for k, v in d.items()); keep = {k for k, c in cnt.items() if c >= 3}
filt = lambda ds: [{k: v for k, v in d.items() if key(k, v) in keep} for d in ds]
dv = DictVectorizer().fit(filt(tr_d))
mk = lambda ds, g: sp.hstack([dv.transform(filt(ds)), sp.csr_matrix(g * 0.3)]).tocsr()
svm = LinearSVC(C=0.03, class_weight="balanced", tol=1e-3, max_iter=100, random_state=42)
svm.fit(mk(tr_d, GLV["train"]), FLAT(LAB["train"]))
svm_dev = unflat(svm.predict(mk(DICTS["dev"], GLV["dev"])), LENS["dev"])
svm_test = unflat(svm.predict(mk(DICTS["test"], GLV["test"])), LENS["test"])
Rs = {"dev": evaluate(D["dev"], svm_dev), "test": evaluate(D["test"], svm_test),
      "test_errors": error_breakdown(D["test"], svm_test)}
json.dump(svm_test, open("results_partB/pred_partB_svm_test.json", "w"))
json.dump(svm_dev, open("results_partB/pred_partB_svm_dev.json", "w"))
json.dump(Rs, open("results_partB/partB_svm.json", "w"), indent=1, default=float)
print("SVM  dev", brief(Rs["dev"]), "| test", brief(Rs["test"]), flush=True)
print("CRF/SVM Section B done.", flush=True)
