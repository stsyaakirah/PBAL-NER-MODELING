"""ANALYSIS ONLY (not the official protocol): how much does labeled data from the target domain help?
The test documents are split into two random halves; the final CRF configuration is trained on train + dev + one half and scored on the
other half (and vice versa), then compared with the official train + dev model on the same sentences."""
import json, numpy as np, sklearn_crfsuite
from wnut_data import load
from features import *
from evalx import evaluate
from seqeval.metrics.sequence_labeling import get_entities
D = {s: load(s) for s in ("train", "dev", "test")}; P = {s: get_pos(s, D[s]) for s in D}
F = {s: [sent_feats(x["tokens"], p, GROUPS) for x, p in zip(D[s], P[s])] for s in D}
fit = lambda X, y: sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.3, c2=0.003, max_iterations=300, all_possible_transitions=True).fit(X, y)
pl = lambda m, X: [list(map(str, q)) for q in m.predict(X)]
te = D["test"]; n = len(te); rng = np.random.RandomState(42); perm = rng.permutation(n); halves = [sorted(perm[: n // 2].tolist()), sorted(perm[n // 2:].tolist())]
base = json.load(open("results/pred_fix_crf_test.json"))                 # official train + dev model, scored on the same sentences
surf = lambda x: [" ".join(x["tokens"][b:e + 1]).lower() for _, b, e in get_entities(x["labels"])]
seen_td = {s for k in ("train", "dev") for x in D[k] for s in surf(x)}
cross = [None] * n; rates = []
for a, b in ((0, 1), (1, 0)):
    Xtr = F["train"] + F["dev"] + [F["test"][i] for i in halves[a]]; ytr = [x["labels"] for x in D["train"] + D["dev"]] + [te[i]["labels"] for i in halves[a]]
    pr = pl(fit(Xtr, ytr), [F["test"][i] for i in halves[b]])
    for i, p in zip(halves[b], pr): cross[i] = p
    seen_a = seen_td | {s for i in halves[a] for s in surf(te[i])}; ents_b = [s for i in halves[b] for s in surf(te[i])]
    rates.append((np.mean([s in seen_td for s in ents_b]), np.mean([s in seen_a for s in ents_b]))); print("fold done", flush=True)
out = {"F1 official (train + dev)": round(100 * evaluate(te, base)["entity"]["f1"], 1), "F1 with the other half of the test added to training": round(100 * evaluate(te, cross)["entity"]["f1"], 1),
       "entity mentions seen in training, official (%)": round(100 * float(np.mean([r[0] for r in rates])), 1), "entity mentions seen in training, with other half (%)": round(100 * float(np.mean([r[1] for r in rates])), 1), "sequences added per fold": len(halves[0])}
json.dump(out, open("results/target_domain_adaptation_crf.json", "w"), indent=1); print(out)
