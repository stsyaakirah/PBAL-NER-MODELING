"""Section B: pooled 80/10/10 random split (document-level), seed 42.

Pools the three official WNUT17 files, splits by DOCUMENT (source_file, doc_id) so a
tweet's sequences never cross splits, writes data_partB/{train,dev,test}.conll, and runs
conservation, BIO and entity-leakage checks. Section A (official split) is untouched.

Mixing is authorized for Section B ONLY. The result is not comparable to the shared task.
"""
import os, json
import numpy as np
from collections import defaultdict, Counter
from seqeval.metrics.sequence_labeling import get_entities
from wnut_data import read_conll, DATA, FILES

OFFICIAL = {  # split name -> file, read straight from the official data dir
    "train": os.path.join(DATA, FILES["train"]),
    "dev":   os.path.join(DATA, FILES["dev"]),
    "test":  os.path.join(DATA, FILES["test"]),
}
OUT = "data_partB"
os.makedirs(OUT, exist_ok=True)

def n_entities(seqs):
    return sum(len(get_entities(x["labels"])) for x in seqs)

def bio_violations(seqs):
    """Count IOB2 violations: an I-X not immediately preceded by B-X or I-X."""
    v = 0
    for x in seqs:
        prev = "O"
        for l in x["labels"]:
            if l.startswith("I-") and prev not in ("B-" + l[2:], "I-" + l[2:]):
                v += 1
            prev = l
    return v

# ---- load pooled, tag every sequence with its global document key ----
pooled = []
for src, path in OFFICIAL.items():
    seqs = read_conll(path)
    for x in seqs:
        x["src"] = src
        x["key"] = (src, x["doc"])       # global document key; never collides across files
    pooled += seqs

orig_seqs = len(pooled)
orig_tokens = sum(len(x["tokens"]) for x in pooled)
orig_ents = n_entities(pooled)
print("pooled originals:", orig_seqs, "seqs", orig_tokens, "tokens", orig_ents, "entities", flush=True)
assert (orig_seqs, orig_tokens, orig_ents) == (5690, 101857, 3890), \
    f"pooled totals {(orig_seqs, orig_tokens, orig_ents)} != expected (5690,101857,3890)"

# ---- group sequences by document, shuffle documents, cut 80/10/10 ----
by_doc = defaultdict(list)
for x in pooled:
    by_doc[x["key"]].append(x)
keys = list(by_doc.keys())
rng = np.random.RandomState(42)
rng.shuffle(keys)
n = len(keys)
n_tr = int(0.8 * n)
n_dv = int(0.1 * n)
split_keys = {"train": keys[:n_tr], "dev": keys[n_tr:n_tr + n_dv], "test": keys[n_tr + n_dv:]}
print("documents:", n, "-> train/dev/test docs:", len(split_keys["train"]), len(split_keys["dev"]), len(split_keys["test"]), flush=True)

split_seqs = {}
for sp, ks in split_keys.items():
    seqs = [x for k in ks for x in by_doc[k]]
    split_seqs[sp] = seqs

# ---- write new CoNLL files (blank line between sequences) ----
def write_conll(seqs, path):
    with open(path, "w", encoding="utf-8") as f:
        for x in seqs:
            for t, l in zip(x["tokens"], x["labels"]):
                f.write(f"{t}\t{l}\n")
            f.write("\n")
for sp in ("train", "dev", "test"):
    write_conll(split_seqs[sp], os.path.join(OUT, f"{sp}.conll"))

# ---- read back and verify conservation + BIO on the WRITTEN files ----
reloaded = {sp: read_conll(os.path.join(OUT, f"{sp}.conll")) for sp in ("train", "dev", "test")}
tot_seqs = sum(len(reloaded[s]) for s in reloaded)
tot_tokens = sum(len(x["tokens"]) for s in reloaded for x in reloaded[s])
tot_ents = sum(n_entities(reloaded[s]) for s in reloaded)
print("written totals:", tot_seqs, tot_tokens, tot_ents, flush=True)
assert (tot_seqs, tot_tokens, tot_ents) == (5690, 101857, 3890), "conservation FAILED after write"
bios = {sp: bio_violations(reloaded[sp]) for sp in reloaded}
print("BIO violations:", bios, flush=True)
assert sum(bios.values()) == 0, "BIO violations present"

# ---- achieved percentages ----
def stats(seqs):
    return {"seqs": len(seqs), "tokens": sum(len(x["tokens"]) for x in seqs), "entities": n_entities(seqs)}
S = {sp: stats(reloaded[sp]) for sp in reloaded}
pct = {sp: {k: round(100 * S[sp][k] / {"seqs": tot_seqs, "tokens": tot_tokens, "entities": tot_ents}[k], 2) for k in S[sp]} for sp in S}

# ---- entity leakage: gold mentions in dev/test whose surface also appears as an entity in NEW train ----
def surfaces(seqs):
    out = Counter()
    for x in seqs:
        for t, b, e in get_entities(x["labels"]):
            out[" ".join(x["tokens"][b:e + 1]).lower()] += 1
    return out
train_surf = set(surfaces(reloaded["train"]))
def leakage(seqs):
    total = 0; seen = 0
    for x in seqs:
        for t, b, e in get_entities(x["labels"]):
            surf = " ".join(x["tokens"][b:e + 1]).lower()
            total += 1; seen += int(surf in train_surf)
    return {"mentions": total, "in_train": seen, "leak_pct": round(100 * seen / total, 1) if total else None}
leak = {sp: leakage(reloaded[sp]) for sp in ("dev", "test")}

# entities that appear in more than one split (surface level)
surf_by_split = {sp: set(surfaces(reloaded[sp])) for sp in reloaded}
multi = [s for s in (surf_by_split["train"] | surf_by_split["dev"] | surf_by_split["test"])
         if sum(s in surf_by_split[sp] for sp in reloaded) >= 2]
# highlight frequently repeated ones
allcnt = surfaces(pooled)
freq_multi = sorted([(s, allcnt[s]) for s in multi], key=lambda z: -z[1])[:20]

summary = {
    "seed": 42, "unit": "document (source_file, doc_id)",
    "documents": {"total": n, "train": len(split_keys["train"]), "dev": len(split_keys["dev"]), "test": len(split_keys["test"])},
    "counts": S, "achieved_pct": pct,
    "conservation_ok": True, "bio_violations": bios,
    "leakage": leak,
    "entities_in_multiple_splits": len(multi),
    "top_repeated_multi_split": freq_multi,
}
json.dump(summary, open(os.path.join(OUT, "split_stats.json"), "w"), indent=1)
print("\n=== LEAKAGE ===")
print("new dev  leak %:", leak["dev"]["leak_pct"], "(", leak["dev"]["in_train"], "/", leak["dev"]["mentions"], ")")
print("new test leak %:", leak["test"]["leak_pct"], "(", leak["test"]["in_train"], "/", leak["test"]["mentions"], ")")
print("entities in >=2 splits:", len(multi), "| top:", freq_multi[:8])
print("\nsplit_stats.json written. Section B split ready.")
