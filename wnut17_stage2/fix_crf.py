"""Rerun of the final CRF after the normalization bug fix. Hyperparameters are the ones chosen on dev BEFORE the fix
(full features, c1 = 0.3, c2 = 0.003, 300 iterations); nothing is re-tuned."""
import json, sklearn_crfsuite
from wnut_data import load
from features import *
from evalx import evaluate, error_breakdown
D = {s: load(s) for s in ["train", "dev", "test"]}; P = {s: get_pos(s, D[s]) for s in D}
LAB = {s: [x["labels"] for x in D[s]] for s in D}
F = {s: [sent_feats(x["tokens"], p, GROUPS) for x, p in zip(D[s], P[s])] for s in D}   # per-token feature dictionaries
fit = lambda X, y: sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.3, c2=0.003, max_iterations=300, all_possible_transitions=True).fit(X, y)
pl = lambda m, X: [list(map(str, q)) for q in m.predict(X)]
R = {}
m = fit(F["train"], LAB["train"])                                   # train only: dev score and reference test score
R["dev_trainonly"] = evaluate(D["dev"], pl(m, F["dev"])); R["test_trainonly"] = evaluate(D["test"], pl(m, F["test"]))
m = fit(F["train"] + F["dev"], LAB["train"] + LAB["dev"])          # final model: train + dev, test scored once
pt = pl(m, F["test"]); R["test"] = evaluate(D["test"], pt); R["test_errors"] = error_breakdown(D["test"], pt)
json.dump(pt, open("results/pred_fix_crf_test.json", "w")); json.dump(R, open("results/fix_crf.json", "w"), indent=1, default=float)
print("CRF done", round(R["test"]["entity"]["f1"] * 100, 2), flush=True)
