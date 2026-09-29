"""5-fold CV per dokumen pada TRAIN saja (sesuai rencana Stage 1) dengan konfigurasi akhir: CRF (lengkap, c1 0,3, c2 0,003, 300 iter) dan SVM (C 0,03, tanpa BIO repair)."""
import json, numpy as np, scipy.sparse as sp, sklearn_crfsuite
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import GroupKFold
from sklearn.svm import LinearSVC
from wnut_data import load
from features import *
from evalx import evaluate
tr = load("train"); pos = get_pos("train", tr); docs = [x["doc"] for x in tr]
Xc = [sent_feats(x["tokens"], p, GROUPS) for x, p in zip(tr, pos)]; y = [x["labels"] for x in tr]
G = np.vstack(glove_vec_table(tr)); lens = [len(x["tokens"]) for x in tr]; starts = np.cumsum([0] + lens)
flat = [f for s in Xc for f in s]; yflat = [l for s in y for l in s]
key = lambda k, v: f"{k}={v}" if not isinstance(v, (float, bool)) else k
import re, os
done = {"CRF": {}, "SVM": {}}
if os.path.exists("results/cv_final_partial.log"):
    for line in open("results/cv_final_partial.log"):
        m = re.match(r"(CRF|SVM) fold (\d) ([\d.]+)", line)
        if m: done[m.group(1)][int(m.group(2))] = float(m.group(3))
R = {"crf": [], "svm": []}
for k, (a, b) in enumerate(GroupKFold(5).split(tr, groups=docs)):
    if k in done["CRF"] and k in done["SVM"]:
        R["crf"].append(done["CRF"][k]); R["svm"].append(done["SVM"][k]); print("reuse fold", k, flush=True); continue
    m = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.3, c2=0.003, max_iterations=300, all_possible_transitions=True).fit([Xc[i] for i in a], [y[i] for i in a])
    R["crf"].append(round(evaluate([tr[i] for i in b], [list(map(str, q)) for q in m.predict([Xc[i] for i in b])])["entity"]["f1"] * 100, 2)); print("CRF fold", k, R["crf"][-1], flush=True)
    idx = lambda sq: np.concatenate([np.arange(starts[i], starts[i + 1]) for i in sq]); ia, ib = idx(a), idx(b)
    cnt = Counter(key(kk, v) for i in ia for kk, v in flat[i].items()); keep = {c for c, n in cnt.items() if n >= 3}
    filt = lambda ii: [{kk: v for kk, v in flat[i].items() if key(kk, v) in keep} for i in ii]
    dv = DictVectorizer().fit(filt(ia)); mk = lambda ii: sp.hstack([dv.transform(filt(ii)), sp.csr_matrix(G[ii] * 0.3)]).tocsr()
    clf = LinearSVC(C=0.03, class_weight="balanced", tol=1e-3, max_iter=100, random_state=42).fit(mk(ia), [yflat[i] for i in ia])
    pr = clf.predict(mk(ib)); out, p0 = [], 0
    for i in b: out.append(list(pr[p0:p0 + lens[i]])); p0 += lens[i]
    R["svm"].append(round(evaluate([tr[i] for i in b], out)["entity"]["f1"] * 100, 2)); print("SVM fold", k, R["svm"][-1], flush=True)
for m in R.copy(): R[m + "_mean"] = round(float(np.mean(R[m])), 2); R[m + "_std"] = round(float(np.std(R[m])), 2)
json.dump(R, open("results/cv_final.json", "w"), indent=1); print("DONE", R, flush=True)
