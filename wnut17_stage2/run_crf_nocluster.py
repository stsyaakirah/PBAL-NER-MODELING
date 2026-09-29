"""Varian CRF tanpa cluster embedding: tuning c1,c2 di dev, lalu train+dev -> test (pemilihan konfigurasi berdasarkan dev)."""
import json, itertools, numpy as np, sklearn_crfsuite
from wnut_data import load
from features import *
from evalx import evaluate, error_breakdown
G = ["word", "shape", "affix", "social", "context", "pos"]
D = {s: load(s) for s in ["train", "dev", "test"]}; P = {s: get_pos(s, D[s]) for s in D}
LAB = {s: [x["labels"] for x in D[s]] for s in D}
F = {s: [sent_feats(x["tokens"], p, G) for x, p in zip(D[s], P[s])] for s in D}
fit = lambda X, y, c1, c2: sklearn_crfsuite.CRF(algorithm="lbfgs", c1=c1, c2=c2, max_iterations=100, all_possible_transitions=True).fit(X, y)
brief = lambda r: {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2), "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}
R = {"grid": []}; best = None
for c1, c2 in itertools.product([0.03, 0.1, 0.3], [0.03, 0.1, 0.3]):
    m = fit(F["train"], LAB["train"], c1, c2); r = evaluate(D["dev"], [list(map(str, q)) for q in m.predict(F["dev"])])
    b = brief(r); b.update(c1=c1, c2=c2); R["grid"].append(b); print("GRID", b, flush=True)
    if best is None or r["entity"]["f1"] > best[0]: best = (r["entity"]["f1"], c1, c2)
R["best"] = {"c1": best[1], "c2": best[2], "dev_F1": round(best[0] * 100, 2)}
m = fit(F["train"] + F["dev"], LAB["train"] + LAB["dev"], best[1], best[2]); pt = [list(map(str, q)) for q in m.predict(F["test"])]
R["test"] = evaluate(D["test"], pt); R["test_errors"] = error_breakdown(D["test"], pt); json.dump(pt, open("results/pred_crf_nocluster_test.json", "w"))
json.dump(R, open("results/crf_nocluster.json", "w"), indent=1, default=float); print("TEST", brief(R["test"]), R["best"], flush=True)
