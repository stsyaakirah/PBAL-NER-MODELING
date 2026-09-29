"""Pencarian CRF diperluas (dev): fitur {lengkap, tanpa cluster} x c1 x c2, max_iter 300; lalu train+dev -> test."""
import json, itertools, numpy as np, sklearn_crfsuite
from wnut_data import load
from features import *
from evalx import evaluate, error_breakdown
SETS = {"lengkap": GROUPS, "tanpa_cluster": [g for g in GROUPS if g != "cluster"]}
D = {s: load(s) for s in ["train", "dev", "test"]}; P = {s: get_pos(s, D[s]) for s in D}
LAB = {s: [x["labels"] for x in D[s]] for s in D}
F = {(n, s): [sent_feats(x["tokens"], p, g) for x, p in zip(D[s], P[s])] for n, g in SETS.items() for s in D}
fit = lambda X, y, c1, c2: sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=300, all_possible_transitions=True).fit(X, y)
pl = lambda m, X: [list(map(str, q)) for q in m.predict(X)]
brief = lambda r: {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2), "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}
R = {"grid": []}; best = None
for n in SETS:
    for c1, c2 in itertools.product([0.1, 0.3, 1.0], [0.003, 0.01, 0.03]):
        m = fit(F[(n, "train")], LAB["train"], c1, c2); r = evaluate(D["dev"], pl(m, F[(n, "dev")]))
        b = brief(r); b.update(fitur=n, c1=c1, c2=c2); R["grid"].append(b); print("GRID", b, flush=True)
        if best is None or r["entity"]["f1"] > best[0]: best = (r["entity"]["f1"], n, c1, c2)
R["best"] = {"fitur": best[1], "c1": best[2], "c2": best[3], "dev_F1": round(best[0] * 100, 2)}; print("BEST", R["best"], flush=True)
n, c1, c2 = best[1:]
m = fit(F[(n, "train")], LAB["train"], c1, c2); R["test_trainonly"] = evaluate(D["test"], pl(m, F[(n, "test")]))
m = fit(F[(n, "train")] + F[(n, "dev")], LAB["train"] + LAB["dev"], c1, c2); pt = pl(m, F[(n, "test")])
R["test"] = evaluate(D["test"], pt); R["test_errors"] = error_breakdown(D["test"], pt); json.dump(pt, open("results/pred_crf2_test.json", "w"))
# hasil ablasi fitur pada konfigurasi terbaik (train -> dev), untuk slide ablasi yang konsisten
R["ablation"] = []
for name, g in [("Kata", ["word"]), ("+ Bentuk, afiks, media sosial", ["word", "shape", "affix", "social"]), ("+ Konteks +-2, bigram", ["word", "shape", "affix", "social", "context"]),
                ("+ POS tag", SETS["tanpa_cluster"]), ("+ Cluster embedding", GROUPS)]:
    Xa = [sent_feats(x["tokens"], p, g) for x, p in zip(D["train"], P["train"])]; Xd = [sent_feats(x["tokens"], p, g) for x, p in zip(D["dev"], P["dev"])]
    b = brief(evaluate(D["dev"], pl(fit(Xa, LAB["train"], c1, c2), Xd))); b["name"] = name; R["ablation"].append(b); print("ABL", b, flush=True)
json.dump(R, open("results/crf2.json", "w"), indent=1, default=float); print("TEST", brief(R["test"]), flush=True); print("DONE", flush=True)
