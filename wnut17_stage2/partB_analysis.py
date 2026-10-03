"""Section B analysis: bootstrap CI, per-type F1, confusion, error breakdown, and the
key leakage decomposition (recall on entities that LEAKED into the new train vs entities
still genuinely unseen). Writes results_partB/partB_analysis.json. Runs on whatever model
prediction files exist in results_partB/ (CRF, SVM, and Bi-LSTM once trained).
"""
import json, os, numpy as np
from collections import defaultdict
from seqeval.metrics.sequence_labeling import get_entities
from seqeval.metrics import f1_score
from wnut_data import read_conll
from evalx import evaluate, error_breakdown, type_confusion

D = {sp: read_conll(f"data_partB/{sp}.conll") for sp in ("train", "dev", "test")}
test = D["test"]
train_surf = {" ".join(x["tokens"][b:e + 1]).lower() for x in D["train"] for _, b, e in get_entities(x["labels"])}

MODELS = [("CRF", "pred_partB_crf_test.json"), ("SVM", "pred_partB_svm_test.json"),
          ("Bi-LSTM CRF", "pred_partB_bilstm_test.json")]
PRED = {}
for name, f in MODELS:
    p = os.path.join("results_partB", f)
    if os.path.exists(p): PRED[name] = json.load(open(p))

def boot_f1(gold, pred, n=2000, seed=42):
    """2000-resample bootstrap CI on strict entity F1 (resample sequences with replacement)."""
    rng = np.random.RandomState(seed); N = len(gold); vals = []
    idx = np.arange(N)
    for _ in range(n):
        s = rng.choice(idx, N, replace=True)
        vals.append(f1_score([gold[i] for i in s], [pred[i] for i in s]) * 100)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return round(float(np.mean(vals)), 2), round(float(lo), 2), round(float(hi), 2)

def seen_unseen_recall(pred):
    """Recall on test entities that leaked into new train (surface seen) vs genuinely unseen."""
    G = defaultdict(lambda: [0, 0])
    for x, p in zip(test, pred):
        pe = set(get_entities(p))
        for (t, b, e) in get_entities(x["labels"]):
            surf = " ".join(x["tokens"][b:e + 1]).lower()
            k = "leaked (seen in new train)" if surf in train_surf else "genuinely unseen"
            G[k][1] += 1; G[k][0] += int((t, b, e) in pe)
    return {k: {"tp": a, "n": nn, "recall": round(100 * a / nn, 1) if nn else None} for k, (a, nn) in G.items()}

gold = [x["labels"] for x in test]
out = {}
for name, pred in PRED.items():
    ev = evaluate(test, pred)
    m, lo, hi = boot_f1(gold, pred)
    M, rows, cols = type_confusion(test, pred)
    out[name] = {
        "entity": {k: round(ev["entity"][k] * 100, 2) for k in ("precision", "recall", "f1")},
        "surface_f1": round(ev["surface"]["f1"] * 100, 2),
        "token_acc": round(ev["token_acc"] * 100, 2),
        "per_type": {t: {"f1": round(v["f1-score"] * 100, 2), "support": v["support"]} for t, v in ev["per_type"].items()},
        "bootstrap_test_f1": {"mean": m, "lo": lo, "hi": hi},
        "error_breakdown": error_breakdown(test, pred),
        "confusion": {"matrix": M.tolist(), "rows": rows, "cols": cols},
        "recall_leaked_vs_unseen": seen_unseen_recall(pred),
    }
    print(name, "F1", out[name]["entity"]["f1"], "CI", (lo, hi), "| leaked/unseen recall:",
          {k: v["recall"] for k, v in out[name]["recall_leaked_vs_unseen"].items()}, flush=True)

json.dump(out, open("results_partB/partB_analysis.json", "w"), indent=1, default=float)
print("partB_analysis.json written for models:", list(PRED.keys()), flush=True)
