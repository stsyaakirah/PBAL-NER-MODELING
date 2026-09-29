"""Twitter-only protocol: CRF and SVM trained on train only (final settings chosen on dev); predictions saved for the analysis."""
"""Rerun of the final SVM after the normalization fix (C = 0.03 chosen on dev before the fix; no BIO repair)."""
"""Cek batas grid SVM (C=0.01) dan keputusan BIO repair berdasarkan dev (bukan test)."""
import json, time, numpy as np, scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from wnut_data import load
from features import *
from evalx import evaluate, error_breakdown
D = {s: load(s) for s in ["train", "dev", "test"]}; P = {s: get_pos(s, D[s]) for s in D}
LAB = {s: [x["labels"] for x in D[s]] for s in D}; FLAT = lambda q: [l for s in q for l in s]
DICTS = {s: [f for x, p in zip(D[s], P[s]) for f in sent_feats(x["tokens"], p, GROUPS)] for s in D}
GLV = {s: np.vstack(glove_vec_table(D[s])) for s in D}; LENS = {s: [len(x["tokens"]) for x in D[s]] for s in D}
key = lambda k, v: f"{k}={v}" if not isinstance(v, (float, bool)) else k
def unflat(flat, lens):
    out, i = [], 0
    for n in lens: out.append(list(flat[i:i + n])); i += n
    return out
def fp_unused(fit_splits, eval_split, C):
    tr_d = [d for s in fit_splits for d in DICTS[s]]; cnt = Counter(key(k, v) for d in tr_d for k, v in d.items()); keep = {k for k, c in cnt.items() if c >= 3}
    filt = lambda ds: [{k: v for k, v in d.items() if key(k, v) in keep} for d in ds]; dv = DictVectorizer().fit(filt(tr_d))
    mk = lambda ds, g: sp.hstack([dv.transform(filt(ds)), sp.csr_matrix(g * 0.3)]).tocsr()
    clf = LinearSVC(C=C, class_weight="balanced", tol=1e-3, max_iter=100, random_state=42).fit(mk(tr_d, np.vstack([GLV[s] for s in fit_splits])), [l for s in fit_splits for l in FLAT(LAB[s])])
    raw = unflat(clf.predict(mk(DICTS[eval_split], GLV[eval_split])), LENS[eval_split]); return raw, [bio_repair(s) for s in raw]
brief = lambda r: {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2), "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}

def fit_predict_multi(fit_splits, eval_splits, C=0.03):
    """Fit one LinearSVC (balanced classes, rare features dropped) and predict several splits. Returns {split: (raw, repaired)}."""
    tr_d = [d for s in fit_splits for d in DICTS[s]]; cnt = Counter(key(k, v) for d in tr_d for k, v in d.items()); keep = {k for k, c in cnt.items() if c >= 3}
    filt = lambda ds: [{k: v for k, v in d.items() if key(k, v) in keep} for d in ds]; dv = DictVectorizer().fit(filt(tr_d))
    mk = lambda ds, g: sp.hstack([dv.transform(filt(ds)), sp.csr_matrix(g * 0.3)]).tocsr()
    clf = LinearSVC(C=C, class_weight="balanced", tol=1e-3, max_iter=100, random_state=42).fit(mk(tr_d, np.vstack([GLV[s] for s in fit_splits])), [l for s in fit_splits for l in FLAT(LAB[s])])
    out = {}
    for es in eval_splits:
        raw = unflat(clf.predict(mk(DICTS[es], GLV[es])), LENS[es]); out[es] = (raw, [bio_repair(x) for x in raw])
    return out
import sklearn_crfsuite
# ---- CRF, trained on train only (Twitter), final configuration chosen on dev
Fc = {s: [sent_feats(x["tokens"], p, GROUPS) for x, p in zip(D[s], P[s])] for s in ("train", "dev", "test")}
m = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.3, c2=0.003, max_iterations=300, all_possible_transitions=True).fit(Fc["train"], LAB["train"])
plc = lambda X: [list(map(str, q)) for q in m.predict(X)]
pdv, pts = plc(Fc["dev"]), plc(Fc["test"])
Rc = {"dev": evaluate(D["dev"], pdv), "test": evaluate(D["test"], pts), "test_errors": error_breakdown(D["test"], pts)}
json.dump(pts, open("results/pred_to_crf_test.json", "w")); json.dump(Rc, open("results/to_crf.json", "w"), indent=1, default=float); print("CRF done", round(100 * Rc["test"]["entity"]["f1"], 2), flush=True)
# ---- SVM, trained on train only, no BIO repair
o = fit_predict_multi(["train"], ["dev", "test"]); (rd, pd_), (rt, pt_) = o["dev"], o["test"]
Rs = {"dev": evaluate(D["dev"], rd), "dev_repair": evaluate(D["dev"], pd_), "test": evaluate(D["test"], rt), "test_repair": evaluate(D["test"], pt_), "test_errors": error_breakdown(D["test"], rt), "invalid_I": sum(1 for s in rt for a, b in zip(s, bio_repair(s)) if a != b)}
json.dump(rt, open("results/pred_to_svm_test.json", "w")); json.dump(Rs, open("results/to_svm.json", "w"), indent=1, default=float); print("SVM done", round(100 * Rs["test"]["entity"]["f1"], 2), flush=True)
