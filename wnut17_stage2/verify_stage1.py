from collections import Counter
import json
from wnut_data import load, entities
D = {s: load(s) for s in ["train", "dev", "test"]}
out = {}
for s, seqs in D.items():
    n_tok = sum(len(x["tokens"]) for x in seqs)
    labs = Counter(l for x in seqs for l in x["labels"])
    ents = [e for x in seqs for e in entities(x["labels"])]
    n_ent_seq = sum(1 for x in seqs if any(l != "O" for l in x["labels"]))
    lens = [len(x["tokens"]) for x in seqs]
    ent_len = [e - b + 1 for _, b, e in ents]
    # pelanggaran BIO: I- tanpa B-/I- sebelumnya dengan tipe sama
    viol = 0
    for x in seqs:
        prev = "O"
        for l in x["labels"]:
            if l.startswith("I-") and prev not in ("B-" + l[2:], "I-" + l[2:]):
                viol += 1
            prev = l
    out[s] = dict(docs=len(set(x["doc"] for x in seqs)), seqs=len(seqs), tokens=n_tok, entities=len(ents),
                  avg_len=round(n_tok / len(seqs), 1), pct_seq_with_ent=round(100 * n_ent_seq / len(seqs), 1),
                  pct_O=round(100 * labs["O"] / n_tok, 1), max_len=max(lens), avg_ent_len=round(sum(ent_len) / len(ent_len), 2),
                  max_ent_len=max(ent_len), bio_violations=viol,
                  types=dict(Counter(t for t, _, _ in ents)), labels=dict(labs))
print(json.dumps(out, indent=1))
tot = {k: sum(out[s][k] for s in out) for k in ["seqs", "tokens", "entities"]}
print("TOTAL", tot)
# contoh Reddit
for i, x in enumerate(D["test"]):
    if "Umm" in x["tokens"] and "criticisms" in x["tokens"]:
        print("CONTOH idx", i, list(zip(x["tokens"], x["labels"])))
# kebaruan
tr = {" ".join(x["tokens"][b:e + 1]).lower() for x in D["train"] for _, b, e in entities(x["labels"])}
tr_tok = {t.lower() for x in D["train"] for t in x["tokens"]}
for s in ["dev", "test"]:
    surf = [" ".join(x["tokens"][b:e + 1]).lower() for x in D[s] for _, b, e in entities(x["labels"])]
    new = sum(1 for z in surf if z not in tr)
    et = [t.lower() for x in D[s] for t, l in zip(x["tokens"], x["labels"]) if l != "O"]
    at = [t.lower() for x in D[s] for t in x["tokens"]]
    print(s, "surface baru %.1f%%" % (100 * new / len(surf)),
          "OOV token entitas %.1f%%" % (100 * sum(t not in tr_tok for t in et) / len(et)),
          "OOV semua token %.1f%%" % (100 * sum(t not in tr_tok for t in at) / len(at)),
          "surf sekali %.1f%%" % (100 * sum(1 for z, c in Counter(surf).items() if c == 1) / len(set(surf))))
