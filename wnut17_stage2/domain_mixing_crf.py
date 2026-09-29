"""Training-domain mixing experiment with the final CRF configuration (no re-tuning):
dev only, train + dev (official protocol) and a pooled random split of all three files (shows why mixing test data into training is not valid)."""
import json, numpy as np, sklearn_crfsuite
from wnut_data import load
from features import *
from evalx import evaluate
from seqeval.metrics.sequence_labeling import get_entities
D = {s: load(s) for s in ("train", "dev", "test")}; P = {s: get_pos(s, D[s]) for s in D}
F = {s: [sent_feats(x["tokens"], p, GROUPS) for x, p in zip(D[s], P[s])] for s in D}
fit = lambda X, y: sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.3, c2=0.003, max_iterations=300, all_possible_transitions=True).fit(X, y)
pl = lambda m, X: [list(map(str, q)) for q in m.predict(X)]
f1 = lambda seqs, pr: round(100 * evaluate(seqs, pr)["entity"]["f1"], 1)
out = {}
m = fit(F["dev"], [x["labels"] for x in D["dev"]]); out["dev only -> test"] = f1(D["test"], pl(m, F["test"]))
items = [(s, x["doc"], F[s][i], x) for s in D for i, x in enumerate(D[s])]                 # pool train + dev + test
docs = sorted({(s, d) for s, d, _, _ in items}); rng = np.random.RandomState(42); rng.shuffle(docs); held = set(docs[: int(0.15 * len(docs))])   # 15% of documents held out
tr = [it for it in items if (it[0], it[1]) not in held]; te = [it for it in items if (it[0], it[1]) in held]
m = fit([it[2] for it in tr], [it[3]["labels"] for it in tr]); pr = pl(m, [it[2] for it in te]); te_seqs = [it[3] for it in te]
out["pooled random split (85/15, by document)"] = f1(te_seqs, pr)
seen_tr = {" ".join(x["tokens"][b:e + 1]).lower() for _, _, _, x in tr for _, b, e in get_entities(x["labels"])}
ents = [" ".join(x["tokens"][b:e + 1]).lower() for x in te_seqs for _, b, e in get_entities(x["labels"])]
out["pooled: held-out entity mentions already seen in training (%)"] = round(100 * np.mean([e in seen_tr for e in ents]), 1); out["pooled: n held-out sequences / entities"] = (len(te), len(ents))
R = json.load(open("results/fix_crf.json")); out["train only -> test (official)"] = round(100 * R["test_trainonly"]["entity"]["f1"], 1); out["train + dev -> test (official)"] = round(100 * R["test"]["entity"]["f1"], 1)
json.dump(out, open("results/domain_mixing_crf.json", "w"), indent=1, default=float); print(out)
