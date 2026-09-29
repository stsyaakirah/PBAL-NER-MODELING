import json, time, itertools, numpy as np, sklearn_crfsuite
from sklearn.model_selection import GroupKFold
from wnut_data import load
from features import *
from evalx import evaluate, error_breakdown

D = {s: load(s) for s in ["train", "dev", "test"]}
P = {s: get_pos(s, D[s]) for s in D}
LAB = {s: [x["labels"] for x in D[s]] for s in D}
def feats(s, groups): return [sent_feats(x["tokens"], p, groups) for x, p in zip(D[s], P[s])]
def fit(X, y, c1=0.1, c2=0.1):
    m = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=100, all_possible_transitions=True)
    m.fit(X, y); return m
R = {}
def brief(r): return {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2),
                      "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}

# 1-2) ablasi + tuning: dibaca ulang dari log jika RESUME_LOG diberikan
import os
if os.environ.get("RESUME_LOG"):
    R["ablation"], R["grid"] = [], []
    for line in open(os.environ["RESUME_LOG"]):
        if line.startswith("ABL "): R["ablation"].append(eval(line[4:].rsplit("}", 1)[0] + "}", {"np": np}))
        elif line.startswith("GRID "): R["grid"].append(eval(line[5:], {"np": np}))
    g = max(R["grid"], key=lambda z: z["F1"]); best = (g["F1"] / 100, g["c1"], g["c2"])
    R["best"] = {"c1": g["c1"], "c2": g["c2"], "dev_F1": g["F1"]}
    Xtr, Xdv = feats("train", GROUPS), feats("dev", GROUPS); print("RESUME", R["best"], flush=True)
else:
    # 1) ablasi fitur kumulatif (train -> dev)
    steps = [("Kata (lower, norm, stopword)", ["word"]),
             ("+ Bentuk, afiks, sinyal media sosial", ["word", "shape", "affix", "social"]),
             ("+ Konteks +-2 dan bigram", ["word", "shape", "affix", "social", "context"]),
             ("+ POS tag", ["word", "shape", "affix", "social", "context", "pos"]),
             ("+ Cluster embedding (lengkap)", GROUPS)]
    R["ablation"] = []
    for name, g in steps:
        t0 = time.time(); m = fit(feats("train", g), LAB["train"])
        r = evaluate(D["dev"], m.predict(feats("dev", g))); b = brief(r); b["name"] = name
        R["ablation"].append(b); print("ABL", b, round(time.time() - t0), "s", flush=True)
    
    # 2) tuning c1,c2 pada dev (fitur lengkap)
    Xtr, Xdv = feats("train", GROUPS), feats("dev", GROUPS)
    R["grid"] = []; best = None
    for c1, c2 in itertools.product([0.03, 0.1, 0.3], [0.03, 0.1, 0.3]):
        m = fit(Xtr, LAB["train"], c1, c2); r = evaluate(D["dev"], m.predict(Xdv)); b = brief(r); b.update(c1=c1, c2=c2)
        R["grid"].append(b); print("GRID", b, flush=True)
        if best is None or r["entity"]["f1"] > best[0]: best = (r["entity"]["f1"], c1, c2)
    R["best"] = {"c1": best[1], "c2": best[2], "dev_F1": round(best[0] * 100, 2)}; print("BEST", R["best"], flush=True)

# 3) final: train saja dan train+dev -> test
c1, c2 = best[1], best[2]; Xte = feats("test", GROUPS)
m = fit(Xtr, LAB["train"], c1, c2); pt = [list(map(str, q)) for q in m.predict(Xte)]
R["test_trainonly"] = evaluate(D["test"], pt); json.dump(pt, open("results/pred_crf_trainonly_test.json", "w"))
m = fit(Xtr + Xdv, LAB["train"] + LAB["dev"], c1, c2); pt = [list(map(str, q)) for q in m.predict(Xte)]
R["test"] = evaluate(D["test"], pt); R["test_errors"] = error_breakdown(D["test"], pt)
json.dump(pt, open("results/pred_crf_test.json", "w"))
# bobot transisi & fitur teratas (untuk analisis)
R["top_features"] = [(f"{a}->{b}", round(w, 3)) for (a, b), w in sorted(m.transition_features_.items(), key=lambda kv: -kv[1])[:6]]
print("TEST", brief(R["test"]), flush=True)

# 4) CV 5-fold per dokumen pada train+dev
allseq = D["train"] + D["dev"]; Xall = Xtr + Xdv; yall = LAB["train"] + LAB["dev"]
groups = [("tr", x["doc"]) if i < len(D["train"]) else ("dv", x["doc"]) for i, x in enumerate(allseq)]
gid = {g: k for k, g in enumerate(sorted(set(groups)))}; groups = [gid[g] for g in groups]
cv = []
for k, (a, b) in enumerate(GroupKFold(5).split(Xall, yall, groups)):
    m = fit([Xall[i] for i in a], [yall[i] for i in a], c1, c2)
    r = evaluate([allseq[i] for i in b], m.predict([Xall[i] for i in b])); cv.append(r["entity"]["f1"] * 100)
    print("CV fold", k, round(cv[-1], 2), flush=True)
R["cv"] = {"folds": [round(v, 2) for v in cv], "mean": round(float(np.mean(cv)), 2), "std": round(float(np.std(cv)), 2)}
json.dump(R, open("results/crf.json", "w"), indent=1, default=float); print("DONE", flush=True)
