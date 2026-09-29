"""SVM (LinearSVC) untuk NER: fitur jarang (min count 3) + vektor GloVe Twitter, lalu BIO repair."""
import json, time, numpy as np, scipy.sparse as sp
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from wnut_data import load
from features import *
from evalx import evaluate, error_breakdown

D = {s: load(s) for s in ["train", "dev", "test"]}
P = {s: get_pos(s, D[s]) for s in D}
LAB = {s: [x["labels"] for x in D[s]] for s in D}
FLAT = lambda seqs: [l for s in seqs for l in s]
DICTS = {s: [f for x, p in zip(D[s], P[s]) for f in sent_feats(x["tokens"], p, GROUPS)] for s in D}
GLV = {s: np.vstack(glove_vec_table(D[s])) for s in D}
LENS = {s: [len(x["tokens"]) for x in D[s]] for s in D}
SCALE, MAXIT, MINCNT = 0.3, 100, 3
key = lambda k, v: f"{k}={v}" if not isinstance(v, (float, bool)) else k
def unflat(flat, lens):
    out, i = [], 0
    for n in lens: out.append(list(flat[i:i + n])); i += n
    return out
def brief(r): return {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2),
                      "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}

def fit_predict(tr_d, tr_g, tr_y, ev_d, ev_g, C=0.1, cw="balanced", use_glove=True):
    cnt = Counter(key(k, v) for d in tr_d for k, v in d.items()); keep = {k for k, c in cnt.items() if c >= MINCNT}
    filt = lambda ds: [{k: v for k, v in d.items() if key(k, v) in keep} for d in ds]
    dv = DictVectorizer().fit(filt(tr_d))
    mk = lambda ds, g: sp.hstack([dv.transform(filt(ds)), sp.csr_matrix(g * SCALE)]).tocsr() if use_glove else dv.transform(filt(ds))
    clf = LinearSVC(C=C, class_weight=cw, tol=1e-3, max_iter=MAXIT, random_state=42).fit(mk(tr_d, tr_g), tr_y)
    return clf.predict(mk(ev_d, ev_g))
def run(fit_splits, eval_split, **kw):
    raw = fit_predict([d for s in fit_splits for d in DICTS[s]], np.vstack([GLV[s] for s in fit_splits]),
                      [l for s in fit_splits for l in FLAT(LAB[s])], DICTS[eval_split], GLV[eval_split], **kw)
    pr = unflat(raw, LENS[eval_split]); return pr, [bio_repair(s) for s in pr]

R = {"settings": {"scale_glove": SCALE, "max_iter": MAXIT, "min_count": MINCNT, "tol": 1e-3}}
def save(): json.dump(R, open("results/svm.json", "w"), indent=1, default=float)
# 1) tuning C (train -> dev)
R["grid"] = []; best = None
for C in [0.03, 0.1, 0.3]:
    t0 = time.time(); _, pr = run(["train"], "dev", C=C); r = evaluate(D["dev"], pr); b = brief(r); b["C"] = C
    R["grid"].append(b); print("GRID", b, round(time.time() - t0), "s", flush=True)
    if best is None or r["entity"]["f1"] > best[0]: best = (r["entity"]["f1"], C)
C = best[1]; R["best"] = {"C": C, "dev_F1": round(best[0] * 100, 2)}; save()
# 2) final: train+dev -> test, dengan dan tanpa BIO repair
raw, fix = run(["train", "dev"], "test", C=C)
R["test_norepair"] = evaluate(D["test"], raw); R["test"] = evaluate(D["test"], fix); R["test_errors"] = error_breakdown(D["test"], fix)
R["repair_changed_tokens"] = sum(1 for s, f in zip(raw, fix) for a, b in zip(s, f) if a != b)
json.dump(fix, open("results/pred_svm_test.json", "w")); save()
print("TEST", brief(R["test"]), "tanpa repair", brief(R["test_norepair"]), "token diubah", R["repair_changed_tokens"], flush=True)
_, pt = run(["train"], "test", C=C); R["test_trainonly"] = evaluate(D["test"], pt); json.dump(pt, open("results/pred_svm_trainonly_test.json", "w")); save()
print("TEST train-only", brief(R["test_trainonly"]), flush=True)
# 3) ablasi (train -> dev)
R["ablation"] = [dict(brief(evaluate(D["dev"], run(["train"], "dev", C=C, **kw)[1])), name=name)
                 for name, kw in [("Tanpa pembobotan kelas", {"cw": None}), ("Tanpa vektor GloVe", {"use_glove": False})]]
R["ablation"].insert(0, dict(next(g for g in R["grid"] if g["C"] == C), name="LinearSVC balanced + GloVe (lengkap)")); save(); print("ABL", R["ablation"], flush=True)
# 4) CV 5-fold per dokumen (train+dev)
allseq = D["train"] + D["dev"]; lens = LENS["train"] + LENS["dev"]
docs = [x["doc"] for x in D["train"]] + [x["doc"] + 100000 for x in D["dev"]]
d_all, g_all, y_all = DICTS["train"] + DICTS["dev"], np.vstack([GLV["train"], GLV["dev"]]), FLAT(LAB["train"]) + FLAT(LAB["dev"])
starts = np.cumsum([0] + lens); cv = []
for k, (a, b) in enumerate(GroupKFold(5).split(allseq, groups=docs)):
    idx = lambda sq: np.concatenate([np.arange(starts[i], starts[i + 1]) for i in sq]); ia, ib = idx(a), idx(b)
    raw = fit_predict([d_all[i] for i in ia], g_all[ia], [y_all[i] for i in ia], [d_all[i] for i in ib], g_all[ib], C=C)
    pr = [bio_repair(s) for s in unflat(raw, [lens[i] for i in b])]
    cv.append(evaluate([allseq[i] for i in b], pr)["entity"]["f1"] * 100); print("CV fold", k, round(cv[-1], 2), flush=True)
R["cv"] = {"folds": [round(v, 2) for v in cv], "mean": round(float(np.mean(cv)), 2), "std": round(float(np.std(cv)), 2)}
save(); print("DONE", flush=True)
