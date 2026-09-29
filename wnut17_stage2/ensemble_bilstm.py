"""Ensemble of the six seed-42 Bi-LSTM CRF variants (V0 to V5, all trained on train only).
Rule fixed BEFORE looking at its scores: average the emission scores and the CRF start, end and transition scores of the six models, then Viterbi-decode.
Candidates are compared on DEV (at delta = 0); the O-bias delta is then tuned on dev for the winner. Test is scored once per delta for transparency."""
import json, pickle, numpy as np, jax, jax.numpy as jnp
from wnut_data import load
from features import get_pos, bio_repair
from bilstm_crf import Encoder, emissions, viterbi, make_batch, bucket_of, DEFAULT
from evalx import evaluate, error_breakdown
D = {s: load(s) for s in ("train", "dev", "test")}; ALL = D["train"] + D["dev"] + D["test"]
for s in D:
    for x, p in zip(D[s], get_pos(s, D[s])): x["pos"] = p
IB = json.load(open("results/improve_bilstm.json")); pct = lambda x: round(100 * x, 2)
brief = lambda ev: {"P": pct(ev["entity"]["precision"]), "R": pct(ev["entity"]["recall"]), "F1": pct(ev["entity"]["f1"]), "surf": pct(ev["surface"]["f1"])}
models = []
for v in IB["variants"]:
    blob = pickle.load(open(f"results/ckpt_{v['name'][:2]}.pkl", "rb")); p = jax.tree_util.tree_map(jnp.asarray, blob["params"]); cfg = {**DEFAULT, **v["cfg"]}
    enc = Encoder(D["train"], ALL, pos=v["pos"]); E = p["E_s"] * cfg.get("emb_scale", 0.1) if "E_s" in p else jnp.asarray(enc.E)
    em = jax.jit(lambda W, C, CP, M, p=p, E=E: emissions(p, E, W, C, CP, M, None, False, 0.0))          # emission scores of one model
    models.append({"name": v["name"], "enc": enc, "em": em, "start": p["start"], "end": p["end"], "trans": p["trans"]})
vit = jax.jit(viterbi)

def averaged_emissions(seqs, bs=64):
    """Average emission scores of all models per batch. Returns the batches so that several deltas can be decoded without recomputing."""
    encs = [[m["enc"].encode(x) for x in seqs] for m in models]; byb = {}
    for i, e in enumerate(encs[0]): byb.setdefault(bucket_of(e["n"]), []).append(i)
    out = []
    for L, idxs in byb.items():
        for s in range(0, len(idxs), bs):
            ch = idxs[s:s + bs]; ems = []
            for m, el in zip(models, encs):
                W, C, CP, Y, M, RW = make_batch(el, ch, L, bs); ems.append(np.asarray(m["em"](W, C, CP, M)))
            out.append((ch, L, np.mean(ems, axis=0), M, [encs[0][i]["n"] for i in ch]))
    return out
st, en, tr = (jnp.mean(jnp.stack([m[k] for m in models]), axis=0) for k in ("start", "end", "trans"))
def decode(batches, n_seqs, delta):
    preds = [None] * n_seqs
    for ch, L, em, M, ns in batches:
        e = jnp.asarray(em).at[..., 0].add(-delta); pr = np.asarray(vit(e, M, st, en, tr))
        for r, (i, n) in enumerate(zip(ch, ns)): labs = [models[0]["enc"].tags[t] for t in pr[r, :min(n, L)]]; preds[i] = bio_repair(labs + ["O"] * (n - len(labs)))
    return preds
Bd, Bt = averaged_emissions(D["dev"]), averaged_emissions(D["test"]); rows = []
for delta in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    dv = evaluate(D["dev"], decode(Bd, len(D["dev"]), delta)); ts = evaluate(D["test"], decode(Bt, len(D["test"]), delta)); rows.append({"delta": delta, "dev": brief(dv), "test": brief(ts)}); print("ensemble delta", delta, rows[-1]["dev"]["F1"], rows[-1]["test"]["F1"], flush=True)
best_delta = max(rows, key=lambda r: (r["dev"]["F1"], -r["delta"]))["delta"]
p0, p1 = decode(Bt, len(D["test"]), 0.0), decode(Bt, len(D["test"]), best_delta)
json.dump(p0, open("results/pred_to_bilstm_ens_nobias_test.json", "w")); json.dump(p1, open("results/pred_to_bilstm_ens_test.json", "w"))
best_single_dev = max(v["dev_F1"] for v in IB["variants"])
R = {"members": [m["name"] for m in models], "rows": rows, "best_delta": best_delta, "dev_delta0": rows[0]["dev"]["F1"], "best_single_dev_delta0": best_single_dev, "wins_dev_at_delta0": rows[0]["dev"]["F1"] > best_single_dev,
     "test_delta0": evaluate(D["test"], p0), "test": evaluate(D["test"], p1), "test_errors": error_breakdown(D["test"], p1), "dev": evaluate(D["dev"], decode(Bd, len(D["dev"]), best_delta))}
json.dump(R, open("results/ensemble_bilstm.json", "w"), indent=1, default=float); print("ENS done | dev@0:", rows[0]["dev"]["F1"], "best single dev:", best_single_dev, "| best delta", best_delta, "| test@0", rows[0]["test"]["F1"], flush=True)
