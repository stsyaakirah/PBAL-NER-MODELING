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
def fp(fit_splits, eval_split, C):
    tr_d = [d for s in fit_splits for d in DICTS[s]]; cnt = Counter(key(k, v) for d in tr_d for k, v in d.items()); keep = {k for k, c in cnt.items() if c >= 3}
    filt = lambda ds: [{k: v for k, v in d.items() if key(k, v) in keep} for d in ds]; dv = DictVectorizer().fit(filt(tr_d))
    mk = lambda ds, g: sp.hstack([dv.transform(filt(ds)), sp.csr_matrix(g * 0.3)]).tocsr()
    clf = LinearSVC(C=C, class_weight="balanced", tol=1e-3, max_iter=100, random_state=42).fit(mk(tr_d, np.vstack([GLV[s] for s in fit_splits])), [l for s in fit_splits for l in FLAT(LAB[s])])
    raw = unflat(clf.predict(mk(DICTS[eval_split], GLV[eval_split])), LENS[eval_split]); return raw, [bio_repair(s) for s in raw]
brief = lambda r: {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2), "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}
R = {"dev": []}
for C in (0.01, 0.03):
    raw, fix = fp(["train"], "dev", C)
    R["dev"].append({"C": C, "repair": brief(evaluate(D["dev"], fix)), "no_repair": brief(evaluate(D["dev"], raw))}); print("DEV", R["dev"][-1], flush=True)
json.dump(R, open("results/svm2.json", "w"), indent=1, default=float)
best = max(R["dev"], key=lambda z: z["repair"]["F1"]); R["choice_by_dev"] = {"C": best["C"], "repair_better_on_dev": best["repair"]["F1"] >= best["no_repair"]["F1"]}
print("CHOICE", R["choice_by_dev"], flush=True)
if best["C"] != 0.03:
    raw, fix = fp(["train", "dev"], "test", best["C"])
    R["test_C001"] = {"repair": evaluate(D["test"], fix), "no_repair": evaluate(D["test"], raw)}; json.dump(fix, open("results/pred_svm_c001_test.json", "w")); print("TEST C0.01", brief(R["test_C001"]["repair"]), brief(R["test_C001"]["no_repair"]), flush=True)
json.dump(R, open("results/svm2.json", "w"), indent=1, default=float); print("DONE", flush=True)
