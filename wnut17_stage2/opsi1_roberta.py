"""Opsi 1 (contextual): Bi-LSTM CRF + frozen RoBERTa features (Twitter-only, seed 42).

Same baseline architecture (frozen GloVe + char-CNN + capitalization + Bi-LSTM 128/dir + CRF)
with one added frozen input per token: the RoBERTa-base last-hidden-state vector (768d),
extracted by extract_roberta.py. Nothing in RoBERTa is fine-tuned. Early stopping on dev,
O-bias delta tuned on dev; test scored for transparency. Labeled as an enhancement to the
Bi-LSTM CRF, not a 4th model. Reuses the canonical LSTM/CRF/Viterbi from bilstm_crf.py.
"""
import os, json, time, pickle, math, numpy as np
import jax, jax.numpy as jnp
import optax
from wnut_data import load
from features import bio_repair
from evalx import evaluate, error_breakdown
from bilstm_crf import (Encoder, lstm_dir, crf_nll, viterbi, glorot, cap_id, N_CAP,
                        MAXW, BUCKETS, bucket_of, BIO_TAGS)

SEED = 42; CTX = 768
D = {s: load(s) for s in ("train", "dev", "test")}
ALL = D["train"] + D["dev"] + D["test"]
RB = {s: pickle.load(open(f"results/roberta_{s}.pkl", "rb")) for s in D}
enc = Encoder(D["train"], ALL)
Eglove = jnp.asarray(enc.E)
cfg = dict(hidden=128, char_dim=30, char_filters=50, cap_dim=8, dropout=0.5, lr=1e-3, bs=32, ctx_dim=CTX)
pct = lambda x: round(100 * x, 2)
brief = lambda ev: {"P": pct(ev["entity"]["precision"]), "R": pct(ev["entity"]["recall"]),
                    "F1": pct(ev["entity"]["f1"]), "surf": pct(ev["surface"]["f1"])}

def init_params(key, n_chars, n_tags, cfg):
    ks = jax.random.split(key, 12); H = cfg["hidden"]; Ce = cfg["char_dim"]; F = cfg["char_filters"]
    din = 200 + F + cfg["cap_dim"] + cfg["ctx_dim"]
    def lstm(k1, k2):
        b = jnp.zeros(4 * H).at[H:2 * H].set(1.0)
        return {"Wx": glorot(k1, (din, 4 * H)), "Wh": glorot(k2, (H, 4 * H)), "b": b}
    return {"char_emb": jax.random.normal(ks[0], (n_chars, Ce)) * 0.1,
            "conv_w": glorot(ks[1], (3, Ce, F)), "conv_b": jnp.zeros(F),
            "cap_emb": jax.random.normal(ks[2], (N_CAP, cfg["cap_dim"])) * 0.1,
            "fw": lstm(ks[3], ks[4]), "bw": lstm(ks[5], ks[6]),
            "out_w": glorot(ks[7], (2 * H, n_tags)), "out_b": jnp.zeros(n_tags),
            "trans": jnp.zeros((n_tags, n_tags)), "start": jnp.zeros(n_tags), "end": jnp.zeros(n_tags)}

def emissions(p, E, W, C, CP, R, M, key, train, drop):
    we = E[W]
    ce = p["char_emb"][C]; Lw = ce.shape[2]; cpad = jnp.pad(ce, ((0, 0), (0, 0), (1, 1), (0, 0)))
    conv = sum(jnp.einsum("btlc,cf->btlf", cpad[:, :, k:k + Lw], p["conv_w"][k]) for k in range(3)) + p["conv_b"]
    conv = jnp.where((C > 0)[..., None], jax.nn.relu(conv), -1e9)
    cf = jnp.max(conv, axis=2); cf = jnp.where(cf < -1e8, 0.0, cf)
    x = jnp.concatenate([we, cf, p["cap_emb"][CP], R], -1)   # R = frozen RoBERTa (B,T,768)
    if train:
        k1, k2 = jax.random.split(key)
        x = x * jax.random.bernoulli(k1, 1 - drop, x.shape) / (1 - drop)
    h = jnp.concatenate([lstm_dir(p["fw"], x, M, False), lstm_dir(p["bw"], x, M, True)], -1)
    if train: h = h * jax.random.bernoulli(k2, 1 - drop, h.shape) / (1 - drop)
    return h @ p["out_w"] + p["out_b"]

def make_batch(encs, R_list, idxs, L, bs):
    B = bs; W = np.zeros((B, L), np.int32); C = np.zeros((B, L, MAXW), np.int32)
    CP = np.zeros((B, L), np.int32); Y = np.zeros((B, L), np.int32)
    R = np.zeros((B, L, CTX), np.float32); M = np.zeros((B, L), bool); RW = np.zeros(B, np.float32)
    for r, i in enumerate(idxs):
        e = encs[i]; n = min(e["n"], L)
        W[r, :n] = e["w"][:n]; C[r, :n] = e["c"][:n]; CP[r, :n] = e["cap"][:n]; Y[r, :n] = e["y"][:n]
        R[r, :n] = R_list[i][:n]; M[r, :n] = True; RW[r] = 1
    for r in range(len(idxs), B): M[r, 0] = True
    return W, C, CP, R, Y, M, RW

opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(cfg["lr"]))
def loss_fn(p, W, C, CP, R, Y, M, RW, key):
    em = emissions(p, Eglove, W, C, CP, R, M, key, True, cfg["dropout"])
    nll = crf_nll(em, Y, M, p["start"], p["end"], p["trans"])
    return (nll * RW).sum() / jnp.maximum(RW.sum(), 1.0)

@jax.jit
def train_step(p, ost, W, C, CP, R, Y, M, RW, key):
    l, g = jax.value_and_grad(loss_fn)(p, W, C, CP, R, Y, M, RW, key)
    up, ost = opt.update(g, ost, p); return optax.apply_updates(p, up), ost, l

@jax.jit
def decode_step(p, W, C, CP, R, M, delta):
    em = emissions(p, Eglove, W, C, CP, R, M, None, False, 0.0).at[..., 0].add(-delta)
    return viterbi(em, M, p["start"], p["end"], p["trans"])

def predict(p, seqs, R_list, delta=0.0, bs=64):
    encs = [enc.encode(x) for x in seqs]; out = [None] * len(seqs); byb = {}
    for i, e in enumerate(encs): byb.setdefault(bucket_of(e["n"]), []).append(i)
    for L, idxs in byb.items():
        for s in range(0, len(idxs), bs):
            ch = idxs[s:s + bs]; W, C, CP, R, Y, M, RW = make_batch(encs, R_list, ch, L, bs)
            pr = np.asarray(decode_step(p, W, C, CP, jnp.asarray(R), M, jnp.float32(delta)))
            for r, i in enumerate(ch):
                n = min(encs[i]["n"], L); labs = [enc.tags[t] for t in pr[r, :n]]
                labs += ["O"] * (encs[i]["n"] - n); out[i] = bio_repair(labs)
    return out

# ---- train ----
encs_tr = [enc.encode(x) for x in D["train"]]; rng = np.random.RandomState(SEED); key = jax.random.PRNGKey(SEED)
params = init_params(key, len(enc.c2i) + 2, len(enc.tags), cfg); ost = opt.init(params)
byb = {}
for i, e in enumerate(encs_tr): byb.setdefault(bucket_of(e["n"]), []).append(i)
best, best_params, best_ep, hist = -1, params, 0, []
t0 = time.time()
for ep in range(1, 36):
    batches = []
    for L, idxs in byb.items():
        idxs = list(rng.permutation(idxs)); batches += [(L, idxs[s:s + cfg["bs"]]) for s in range(0, len(idxs), cfg["bs"])]
    rng.shuffle(batches); tl = 0.0
    for L, ch in batches:
        W, C, CP, R, Y, M, RW = make_batch(encs_tr, RB["train"], ch, L, cfg["bs"]); key, sk = jax.random.split(key)
        params, ost, l = train_step(params, ost, W, C, CP, jnp.asarray(R), Y, M, RW, sk); tl += float(l)
    dev_pred = predict(params, D["dev"], RB["dev"]); r = evaluate(D["dev"], dev_pred)
    f1 = round(r["entity"]["f1"] * 100, 2); hist.append(f1)
    print("roberta ep", ep, "dev_f1", f1, "loss", round(tl / len(batches), 3), flush=True)
    if f1 > best: best, best_params, best_ep = f1, params, ep
    if ep - best_ep >= 8: break
print("best epoch", best_ep, "dev", best, "| train sec", round(time.time() - t0), flush=True)

# ---- O-bias sweep on dev, then test ----
rows = []
for delta in (0.0, 0.5, 1.0, 1.5, 2.0):
    dv = evaluate(D["dev"], predict(best_params, D["dev"], RB["dev"], delta))
    ts = evaluate(D["test"], predict(best_params, D["test"], RB["test"], delta))
    rows.append({"delta": delta, "dev": brief(dv), "test": brief(ts)})
    print("sweep delta", delta, "dev", rows[-1]["dev"]["F1"], "test", rows[-1]["test"]["F1"], "surf", rows[-1]["test"]["surf"], flush=True)
best_delta = max(rows, key=lambda r: (r["dev"]["F1"], -r["delta"]))["delta"]
p0 = predict(best_params, D["test"], RB["test"], 0.0)
p1 = predict(best_params, D["test"], RB["test"], best_delta)
json.dump(p0, open("results/pred_to_bilstm_roberta_nobias_test.json", "w"))
json.dump(p1, open("results/pred_to_bilstm_roberta_test.json", "w"))
R = {"seed": SEED, "best_epoch": best_ep, "dev_curve": hist, "best_delta": best_delta,
     "test_delta0": evaluate(D["test"], p0), "test": evaluate(D["test"], p1),
     "test_errors": error_breakdown(D["test"], p1), "dev": evaluate(D["dev"], predict(best_params, D["dev"], RB["dev"], best_delta))}
json.dump(R, open("results/bilstm_roberta.json", "w"), indent=1, default=float)
print("ROBERTA done | test@0 entF1", pct(R["test_delta0"]["entity"]["f1"]), "surfF1", pct(R["test_delta0"]["surface"]["f1"]),
      "| delta", best_delta, "entF1", pct(R["test"]["entity"]["f1"]), "surfF1", pct(R["test"]["surface"]["f1"]), flush=True)
