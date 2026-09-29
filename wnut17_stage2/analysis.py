"""Analisis lanjutan dari prediksi test: kebaruan entitas, kapitalisasi, panjang entitas, kebingungan tipe, contoh."""
import json, numpy as np
from collections import Counter, defaultdict
from seqeval.metrics.sequence_labeling import get_entities
from wnut_data import load

D = {s: load(s) for s in ["train", "dev", "test"]}
TYPES = ["person", "location", "corporation", "product", "creative-work", "group"]
PRED = {"CRF": "results/pred_fix_crf_test.json", "SVM": "results/pred_fix_svm_test.json", "Bi-LSTM CRF": "results/pred_fix_bilstm_test.json"}
P = {m: json.load(open(f)) for m, f in PRED.items()}
seen = {" ".join(x["tokens"][b:e + 1]).lower() for s in ("train", "dev") for x in D[s] for _, b, e in get_entities(x["labels"])}
seen_t = {t.lower() for s in ("train", "dev") for x in D[s] for t in x["tokens"]}

def group_recall(pred):
    G = defaultdict(lambda: [0, 0])  # grup -> [tp, total]
    for x, p in zip(D["test"], pred):
        pe = set(get_entities(p))
        for (t, b, e) in get_entities(x["labels"]):
            toks = x["tokens"][b:e + 1]; surf = " ".join(toks).lower()
            gs = ["seen" if surf in seen else "unseen", "1 token" if e == b else ">=2 token",
                  "kapital" if all(w[:1].isupper() or not w[:1].isalpha() for w in toks) else "ada huruf kecil",
                  "OOV" if all(w.lower() not in seen_t for w in toks) else "ada token dikenal"]
            for g in gs:
                G[g][1] += 1; G[g][0] += int((t, b, e) in pe)
    return {g: {"tp": a, "n": n, "recall": round(100 * a / n, 1) if n else None} for g, (a, n) in G.items()}

def confusion(pred):
    """Untuk setiap entitas gold: tipe prediksi jika batas tepat sama; 'batas' jika hanya tumpang tindih; 'terlewat' jika tidak ada."""
    M = {t: Counter() for t in TYPES}
    for x, p in zip(D["test"], pred):
        pe = get_entities(p)
        for (t, b, e) in get_entities(x["labels"]):
            ex = [pt for (pt, pb, pe_) in pe if (pb, pe_) == (b, e)]
            if ex: M[t][ex[0]] += 1
            elif any(not (pe_ < b or pb > e) for (_, pb, pe_) in pe): M[t]["batas"] += 1
            else: M[t]["terlewat"] += 1
    return {t: dict(c) for t, c in M.items()}

S = {"group_recall": {m: group_recall(P[m]) for m in P}, "confusion": {m: confusion(P[m]) for m in P}}
# contoh prediksi: kalimat Reddit Stage 1 + kandidat menarik
ex = {}
i = next(i for i, x in enumerate(D["test"]) if "Umm" in x["tokens"] and "criticisms" in x["tokens"])
ex["reddit"] = {"idx": i, "tokens": D["test"][i]["tokens"], **{m: P[m][i] for m in P}, "gold": D["test"][i]["labels"]}
def spans(x, p): return [(t, " ".join(x["tokens"][b:e + 1])) for t, b, e in get_entities(p)]
cands = []
for i, x in enumerate(D["test"]):
    g = set(get_entities(x["labels"]));
    if not g or len(x["tokens"]) > 22: continue
    ok = {m: set(get_entities(P[m][i])) == g for m in P}
    cands.append((i, ok, spans(x, x["labels"]), {m: spans(x, P[m][i]) for m in P}, " ".join(x["tokens"])))
S["cand_all_correct_bilstm_only"] = [(c[0], c[4], c[2], c[3]) for c in cands if c[1]["Bi-LSTM CRF"] and not c[1]["CRF"] and not c[1]["SVM"]][:6]
S["cand_all_wrong"] = [(c[0], c[4], c[2], c[3]) for c in cands if not any(c[1].values())][:12]
S["examples"] = ex
json.dump(S, open("results/analysis.json", "w"), indent=1, default=str)
for m in P: print(m, {g: v["recall"] for g, v in S["group_recall"][m].items()})
print(json.dumps(S["examples"]["reddit"]))
