"""Opsi 1, multi-seed variant (user-authorized, Twitter-only protocol).

Trains the BASELINE Bi-LSTM CRF architecture with several independent seeds and ensembles
them. Seed diversity is the classic variance-reduction lever; unlike the structural variants
(V6-V8), every member here is a full-strength baseline. Seed 42 stays the primary single-model
record; this multi-seed ensemble is reported as a labeled experiment.

Members: seed 42 baseline (reuse ckpt_V0.pkl) + seeds 43,44,45,46 (trained here).
Delta (O-bias) tuned on DEV only; test scored for transparency, never used to choose.
"""
import json, os, time, pickle, numpy as np
import jax, jax.numpy as jnp
from wnut_data import load
from features import bio_repair
from bilstm_crf import (Encoder, train, predict, emissions, viterbi, make_batch,
                        bucket_of, DEFAULT)
from evalx import evaluate, error_breakdown

SEEDS = [42, 43, 44, 45, 46]
D = {s: load(s) for s in ("train", "dev", "test")}
ALL = D["train"] + D["dev"] + D["test"]
enc = Encoder(D["train"], ALL)                       # one baseline encoder shared by all seeds
E = jnp.asarray(enc.E)
pct = lambda x: round(100 * x, 2)
brief = lambda ev: {"P": pct(ev["entity"]["precision"]), "R": pct(ev["entity"]["recall"]),
                    "F1": pct(ev["entity"]["f1"]), "surf": pct(ev["surface"]["f1"])}

def seed_ckpt(s): return f"results/ckpt_seed{s}.pkl"

params_by_seed = {}
for s in SEEDS:
    if s == 42 and os.path.exists("results/ckpt_V0.pkl"):
        blob = pickle.load(open("results/ckpt_V0.pkl", "rb"))
        params_by_seed[s] = jax.tree_util.tree_map(jnp.asarray, blob["params"])
        print("seed 42: reused ckpt_V0", flush=True); continue
    if os.path.exists(seed_ckpt(s)):
        blob = pickle.load(open(seed_ckpt(s), "rb"))
        params_by_seed[s] = jax.tree_util.tree_map(jnp.asarray, blob["params"])
        print(f"seed {s}: resumed", flush=True); continue
    t0 = time.time()
    p, be, hist, dec = train(enc, D["train"], D["dev"], 35, seed=s, cfg=dict(DEFAULT), patience=8,
                             log=lambda r: print(f"seed{s}", r["epoch"], r.get("dev_f1"), flush=True))
    rt = evaluate(D["test"], predict(p, enc, dec, D["test"]))
    rec = {"seed": s, "best_epoch": be, "test": brief(rt), "sec": round(time.time() - t0)}
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, p), "rec": rec}, open(seed_ckpt(s), "wb"))
    params_by_seed[s] = p
    print(f"DONE seed {s} test {rec['test']}", flush=True)

# ---- ensemble: average emission scores + CRF params, then Viterbi ----
mods = [params_by_seed[s] for s in SEEDS]
emfns = [jax.jit(lambda W, C, CP, M, pp=p: emissions(pp, E, W, C, CP, M, None, False, 0.0)) for p in mods]
st = jnp.mean(jnp.stack([p["start"] for p in mods]), axis=0)
en = jnp.mean(jnp.stack([p["end"] for p in mods]), axis=0)
tr = jnp.mean(jnp.stack([p["trans"] for p in mods]), axis=0)
vit = jax.jit(viterbi)

def averaged_batches(seqs, bs=64):
    encs = [enc.encode(x) for x in seqs]; byb = {}
    for i, e in enumerate(encs): byb.setdefault(bucket_of(e["n"]), []).append(i)
    out = []
    for L, idxs in byb.items():
        for s0 in range(0, len(idxs), bs):
            ch = idxs[s0:s0 + bs]; W, C, CP, Y, M, RW = make_batch(encs, ch, L, bs)
            ems = [np.asarray(f(W, C, CP, M)) for f in emfns]
            out.append((ch, L, np.mean(ems, axis=0), M, [encs[i]["n"] for i in ch]))
    return out

def decode(batches, n_seqs, delta):
    preds = [None] * n_seqs
    for ch, L, em, M, ns in batches:
        e = jnp.asarray(em).at[..., 0].add(-delta); pr = np.asarray(vit(e, M, st, en, tr))
        for r, (i, n) in enumerate(zip(ch, ns)):
            labs = [enc.tags[t] for t in pr[r, :min(n, L)]]
            preds[i] = bio_repair(labs + ["O"] * (n - len(labs)))
    return preds

Bd, Bt = averaged_batches(D["dev"]), averaged_batches(D["test"])
rows = []
for delta in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    dv = evaluate(D["dev"], decode(Bd, len(D["dev"]), delta))
    ts = evaluate(D["test"], decode(Bt, len(D["test"]), delta))
    rows.append({"delta": delta, "dev": brief(dv), "test": brief(ts)})
    print("multiseed delta", delta, "dev", rows[-1]["dev"]["F1"], "test", rows[-1]["test"]["F1"], "surf", rows[-1]["test"]["surf"], flush=True)
best_delta = max(rows, key=lambda r: (r["dev"]["F1"], -r["delta"]))["delta"]
p0 = decode(Bt, len(D["test"]), 0.0); p1 = decode(Bt, len(D["test"]), best_delta)
json.dump(p0, open("results/pred_to_bilstm_multiseed_nobias_test.json", "w"))
json.dump(p1, open("results/pred_to_bilstm_multiseed_test.json", "w"))
R = {"seeds": SEEDS, "rows": rows, "best_delta": best_delta,
     "test_delta0": evaluate(D["test"], p0), "test": evaluate(D["test"], p1),
     "test_errors": error_breakdown(D["test"], p1), "dev": evaluate(D["dev"], decode(Bd, len(D["dev"]), best_delta)),
     "members_test": {s: pickle.load(open(seed_ckpt(s), "rb"))["rec"]["test"] if os.path.exists(seed_ckpt(s)) else None for s in SEEDS if s != 42}}
json.dump(R, open("results/ensemble_multiseed.json", "w"), indent=1, default=float)
print("MULTISEED done | test@0 entF1", pct(R["test_delta0"]["entity"]["f1"]), "surfF1", pct(R["test_delta0"]["surface"]["f1"]),
      "| best delta", best_delta, "entF1", pct(R["test"]["entity"]["f1"]), "surfF1", pct(R["test"]["surface"]["f1"]), flush=True)
