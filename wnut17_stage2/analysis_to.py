"""Post-hoc analysis for the Twitter-only protocol: recall of gold test entities by novelty (relative to the TRAIN file only), capitalization and length."""
import json, numpy as np
from collections import defaultdict
from seqeval.metrics.sequence_labeling import get_entities
from wnut_data import load
D = {s: load(s) for s in ("train", "dev", "test")}; test = D["test"]
PRED = {"CRF": "pred_to_crf_test.json", "SVM": "pred_to_svm_test.json", "Bi-LSTM CRF (baseline)": "pred_to_bilstm_base_nobias_test.json", "Bi-LSTM CRF (ensemble)": "pred_to_bilstm_ens_nobias_test.json"}
P = {m: json.load(open("results/" + f)) for m, f in PRED.items()}
seen = {" ".join(x["tokens"][b:e + 1]).lower() for x in D["train"] for _, b, e in get_entities(x["labels"])}       # Twitter-only training: 'seen' means seen in train
def group_recall(pred):
    G = defaultdict(lambda: [0, 0])
    for x, p in zip(test, pred):
        pe = set(get_entities(p))
        for (t, b, e) in get_entities(x["labels"]):
            toks = x["tokens"][b:e + 1]; surf = " ".join(toks).lower()
            for g in (("seen" if surf in seen else "unseen"), ("1 token" if e == b else ">=2 token"), ("kapital" if all(w[:1].isupper() or not w[:1].isalpha() for w in toks) else "ada huruf kecil")):
                G[g][1] += 1; G[g][0] += int((t, b, e) in pe)
    return {g: {"tp": a, "n": n, "recall": round(100 * a / n, 1) if n else None} for g, (a, n) in G.items()}
out = {"group_recall": {m: group_recall(p) for m, p in P.items()}}; json.dump(out, open("results/analysis_to.json", "w"), indent=1)
for m in P: print(m, {g: (v["recall"], v["n"]) for g, v in out["group_recall"][m].items()})
