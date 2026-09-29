"""Bi-LSTM CRF (GloVe Twitter beku + char-CNN + fitur kapitalisasi) dalam JAX."""
import re, math, time, pickle, numpy as np
import jax, jax.numpy as jnp
from jax import lax
import optax
from prep_glove import glove_keys
from wnut_data import load
from features import bio_repair
from evalx import evaluate

BIO_TAGS = ["O"] + [f"{p}-{t}" for t in ["person", "location", "corporation", "product", "creative-work", "group"] for p in "BI"]
BIOES_TAGS = ["O"] + [f"{p}-{t}" for t in ["person", "location", "corporation", "product", "creative-work", "group"] for p in "BIES"]
MAXW = 20
BUCKETS = [8, 16, 24, 32, 48, 64, 128]

def bio2bioes(labs):
    """Convert a BIO label sequence to BIOES (25 tags)."""
    out = []
    for i, l in enumerate(labs):
        if l == "O": out.append(l); continue
        nxt = labs[i + 1] if i + 1 < len(labs) else "O"
        cont = nxt == "I-" + l[2:]
        if l[0] == "B": out.append(l if cont else "S-" + l[2:])
        else: out.append(l if cont else "E-" + l[2:])
    return out

def bioes2bio(labs):
    """Convert BIOES predictions back to BIO so that all models are scored the same way."""
    out = []
    for l in labs:
        if l == "O": out.append(l)
        elif l[0] in "BS": out.append("B-" + l[2:])
        else: out.append("I-" + l[2:])
    return bio_repair(out)

def cap_id(w):
    """Capitalization class of a token (URL, mention, hashtag, digit, no letters, lower, upper, title, mixed)."""
    if re.match(r"^(https?://|www\.)", w, re.I): return 7
    if w.startswith("@") and len(w) > 1: return 8
    if w.startswith("#") and len(w) > 1: return 9
    if any(c.isdigit() for c in w): return 5
    if not any(c.isalpha() for c in w): return 6
    if w.islower(): return 1
    if w.isupper(): return 2
    if w.istitle(): return 3
    return 4
N_CAP = 10

class Encoder:
    def __init__(self, train_seqs, all_seqs, bioes=False, pos=False):
        self.tags = BIOES_TAGS if bioes else BIO_TAGS; self.bioes = bioes
        self.tag2id = {t: i for i, t in enumerate(self.tags)}
        self.use_pos = pos
        if pos:   # optional POS-tag input (NLTK tags; every sequence must carry x['pos']); id 0 = tag not seen in training
            self.pos2i = {t: i + 1 for i, t in enumerate(sorted({t for x in train_seqs for t in x['pos']}))}; self.n_pos = len(self.pos2i) + 1
        chars = {c for x in train_seqs for t in x["tokens"] for c in t}
        self.c2i = {c: i + 2 for i, c in enumerate(sorted(chars))}
        nv = np.load("glove_sub.npz")["need_vecs"]; v = pickle.load(open("glove_vocab.pkl", "rb"))["need_words"]
        self.w2i = {}; rows = [np.zeros(200, np.float32), np.zeros(200, np.float32)]
        for x in all_seqs:
            for t in x["tokens"]:
                for k in glove_keys(t):
                    if k in v:
                        if k not in self.w2i: self.w2i[k] = len(rows); rows.append(nv[v[k]])
                        break
        self.E = np.stack(rows)
    def wid(self, t):
        for k in glove_keys(t):
            if k in self.w2i: return self.w2i[k]
        return 1
    def encode(self, x):
        n = len(x["tokens"])
        w = np.array([self.wid(t) for t in x["tokens"]], np.int32)
        c = np.zeros((n, MAXW), np.int32)
        for i, t in enumerate(x["tokens"]):
            for j, ch in enumerate(t[:MAXW]): c[i, j] = self.c2i.get(ch, 1)
        cp = np.array([cap_id(t) for t in x["tokens"]], np.int32)
        if self.use_pos: cp = cp + N_CAP * np.array([self.pos2i.get(t, 0) for t in x["pos"]], np.int32)   # one integer carries capitalization class and POS id
        labs = bio2bioes(x["labels"]) if self.bioes else x["labels"]
        y = np.array([self.tag2id[l] for l in labs], np.int32)
        return {"w": w, "c": c, "cap": cp, "y": y, "n": n}

def bucket_of(n):
    """Smallest length bucket that fits a sequence (batches use fixed shapes so that JAX compiles once per bucket)."""
    for b in BUCKETS:
        if n <= b: return b
    return BUCKETS[-1]

def make_batch(encs, idxs, L, bs):
    """Pad a list of encoded sequences to length L and batch size bs; padding rows get weight 0."""
    B = bs; W = np.zeros((B, L), np.int32); C = np.zeros((B, L, MAXW), np.int32)
    CP = np.zeros((B, L), np.int32); Y = np.zeros((B, L), np.int32); M = np.zeros((B, L), bool); RW = np.zeros(B, np.float32)
    for r, i in enumerate(idxs):
        e = encs[i]; n = min(e["n"], L)
        W[r, :n] = e["w"][:n]; C[r, :n] = e["c"][:n]; CP[r, :n] = e["cap"][:n]; Y[r, :n] = e["y"][:n]; M[r, :n] = True; RW[r] = 1
    for r in range(len(idxs), B):  # baris pengisi (bobot 0) agar bentuk tetap
        M[r, 0] = True
    return W, C, CP, Y, M, RW

# ---------------- parameter ----------------
def glorot(key, shape):
    """Glorot (Xavier) uniform initialization."""
    lim = math.sqrt(6.0 / (shape[0] + shape[-1])); return jax.random.uniform(key, shape, minval=-lim, maxval=lim)

def init_params(key, n_chars, n_tags, cfg):
    """Create all parameters: char embeddings, char-CNN, capitalization embedding, forward and backward LSTM, output layer, CRF transition scores."""
    ks = jax.random.split(key, 12); H = cfg["hidden"]; Ce = cfg["char_dim"]; F = cfg["char_filters"]
    din = 200 + F + cfg["cap_dim"] + cfg.get("pos_dim", 0)
    def lstm(k1, k2):
        b = jnp.zeros(4 * H).at[H:2 * H].set(1.0)
        return {"Wx": glorot(k1, (din, 4 * H)), "Wh": glorot(k2, (H, 4 * H)), "b": b}
    P = {"char_emb": jax.random.normal(ks[0], (n_chars, Ce)) * 0.1, "conv_w": glorot(ks[1], (3, Ce, F)) , "conv_b": jnp.zeros(F),
            "cap_emb": jax.random.normal(ks[2], (N_CAP, cfg["cap_dim"])) * 0.1,
            "fw": lstm(ks[3], ks[4]), "bw": lstm(ks[5], ks[6]),
            "out_w": glorot(ks[7], (2 * H, n_tags)), "out_b": jnp.zeros(n_tags),
            "trans": jnp.zeros((n_tags, n_tags)), "start": jnp.zeros(n_tags), "end": jnp.zeros(n_tags)}
    if cfg.get("pos_dim", 0): P["pos_emb"] = jax.random.normal(ks[8], (cfg["n_pos"], cfg["pos_dim"])) * 0.1
    return P

def lstm_dir(p, xs, mask, reverse):
    """xs (B,T,D) mask (B,T) -> (B,T,H)"""
    if reverse: xs, mask = xs[:, ::-1], mask[:, ::-1]
    H = p["Wh"].shape[0]; B = xs.shape[0]
    xp = jnp.einsum("btd,dh->tbh", xs, p["Wx"]) + p["b"]
    def step(carry, inp):
        h, c = carry; xg, m = inp
        g = xg + h @ p["Wh"]
        i, f, gg, o = jnp.split(g, 4, axis=-1)
        c2 = jax.nn.sigmoid(f) * c + jax.nn.sigmoid(i) * jnp.tanh(gg)
        h2 = jax.nn.sigmoid(o) * jnp.tanh(c2)
        m = m[:, None]; h = jnp.where(m, h2, h); c = jnp.where(m, c2, c)
        return (h, c), h
    z = jnp.zeros((B, H))
    _, hs = lax.scan(step, (z, z), (xp, mask.T))
    hs = jnp.transpose(hs, (1, 0, 2))
    return hs[:, ::-1] if reverse else hs

def emissions(params, E, W, C, CP, M, key, train, drop, wdrop=0.0):
    """Label scores per token: [GloVe | char-CNN | capitalization] -> dropout -> Bi-LSTM -> dropout -> linear layer."""
    we = E[W]                                              # (B,T,200)
    if train and wdrop > 0: we = we * jax.random.bernoulli(jax.random.fold_in(key, 11), 1 - wdrop, (we.shape[0], we.shape[1], 1))   # word dropout: zero the GloVe vector, like a word without a vector
    ce = params["char_emb"][C]                             # (B,T,Lw,Ce)
    Lw = ce.shape[2]; cpad = jnp.pad(ce, ((0, 0), (0, 0), (1, 1), (0, 0)))
    conv = sum(jnp.einsum("btlc,cf->btlf", cpad[:, :, k:k + Lw], params["conv_w"][k]) for k in range(3)) + params["conv_b"]
    conv = jnp.where((C > 0)[..., None], jax.nn.relu(conv), -1e9)
    cf = jnp.max(conv, axis=2); cf = jnp.where(cf < -1e8, 0.0, cf)
    if "pos_emb" in params: x = jnp.concatenate([we, cf, params["cap_emb"][CP % N_CAP], params["pos_emb"][CP // N_CAP]], -1)
    else: x = jnp.concatenate([we, cf, params["cap_emb"][CP]], -1)
    if train:
        k1, k2 = jax.random.split(key)
        x = x * jax.random.bernoulli(k1, 1 - drop, x.shape) / (1 - drop)
    h = jnp.concatenate([lstm_dir(params["fw"], x, M, False), lstm_dir(params["bw"], x, M, True)], -1)
    if train: h = h * jax.random.bernoulli(k2, 1 - drop, h.shape) / (1 - drop)
    return h @ params["out_w"] + params["out_b"]

def crf_nll(em, tags, mask, start, end, trans):
    """Negative log-likelihood of the gold label sequence under the linear-chain CRF (forward algorithm)."""
    B, T, K = em.shape
    eg = jnp.take_along_axis(em, tags[..., None], -1)[..., 0]
    score = start[tags[:, 0]] + (eg * mask).sum(1) + (trans[tags[:, :-1], tags[:, 1:]] * mask[:, 1:]).sum(1)
    last = tags[jnp.arange(B), mask.sum(1) - 1]; score = score + end[last]
    def step(alpha, inp):
        e_t, m_t = inp
        nxt = jax.nn.logsumexp(alpha[:, :, None] + trans[None], axis=1) + e_t
        return jnp.where(m_t[:, None], nxt, alpha), None
    alpha, _ = lax.scan(step, start[None] + em[:, 0], (em[:, 1:].transpose(1, 0, 2), mask[:, 1:].T))
    return jax.nn.logsumexp(alpha + end[None], axis=1) - score

def viterbi(em, mask, start, end, trans):
    """Best label sequence under the CRF (Viterbi decoding with back-pointers)."""
    B, T, K = em.shape
    def step(alpha, inp):
        e_t, m_t = inp
        sc = alpha[:, :, None] + trans[None]
        bp = jnp.argmax(sc, axis=1); nxt = jnp.max(sc, axis=1) + e_t
        return jnp.where(m_t[:, None], nxt, alpha), jnp.where(m_t[:, None], bp, jnp.arange(K)[None])
    alpha, bps = lax.scan(step, start[None] + em[:, 0], (em[:, 1:].transpose(1, 0, 2), mask[:, 1:].T))
    lastt = jnp.argmax(alpha + end[None], axis=1)
    def back(tag, bp):
        return jnp.take_along_axis(bp, tag[:, None], 1)[:, 0], tag
    t0, ys = lax.scan(back, lastt, bps, reverse=True)
    return jnp.concatenate([t0[None], ys], 0).T

def make_fns(cfg, n_tags):
    """Build the jitted training step (Adam, gradient clipping 5) and the jitted decoder."""
    drop = cfg["dropout"]; wd = cfg.get("word_dropout", 0.0); opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(cfg["lr"]))
    Ef = lambda params, E: params["E_s"] * cfg["emb_scale"] if "E_s" in params else E   # fine-tuned embeddings (small effective learning rate) or the frozen GloVe table
    def loss_fn(params, E, W, C, CP, Y, M, RW, key):
        em = emissions(params, Ef(params, E), W, C, CP, M, key, True, drop, wd)
        nll = crf_nll(em, Y, M, params["start"], params["end"], params["trans"])
        return (nll * RW).sum() / jnp.maximum(RW.sum(), 1.0)
    @jax.jit
    def train_step(params, ost, E, W, C, CP, Y, M, RW, key):
        l, g = jax.value_and_grad(loss_fn)(params, E, W, C, CP, Y, M, RW, key)
        up, ost = opt.update(g, ost, params); return optax.apply_updates(params, up), ost, l
    @jax.jit
    def decode(params, E, W, C, CP, M, delta=0.0):
        em = emissions(params, Ef(params, E), W, C, CP, M, None, False, 0.0)
        em = em.at[..., 0].add(-delta)   # delta > 0 lowers the score of the O tag (trades precision for recall); tuned on dev
        return viterbi(em, M, params["start"], params["end"], params["trans"])
    return opt, train_step, decode

def predict(params, enc, decode, seqs, bs=64, delta=0.0):
    """Decode a list of sequences; predictions are converted back to BIO (and repaired if BIOES)."""
    encs = [enc.encode(x) for x in seqs]; out = [None] * len(seqs)
    byb = {}
    for i, e in enumerate(encs): byb.setdefault(bucket_of(e["n"]), []).append(i)
    E = jnp.asarray(enc.E)
    for L, idxs in byb.items():
        for s in range(0, len(idxs), bs):
            ch = idxs[s:s + bs]; W, C, CP, Y, M, RW = make_batch(encs, ch, L, bs)
            pr = np.asarray(decode(params, E, W, C, CP, M, jnp.float32(delta)))
            for r, i in enumerate(ch):
                n = min(encs[i]["n"], L); labs = [enc.tags[t] for t in pr[r, :n]]
                labs += ["O"] * (encs[i]["n"] - n)  # kalimat >128 token: sisa diberi O (tidak terjadi di data ini)
                out[i] = bioes2bio(labs) if enc.bioes else bio_repair(labs)
    return out

DEFAULT = dict(hidden=128, char_dim=30, char_filters=50, cap_dim=8, dropout=0.5, lr=1e-3, bs=32)

def train(enc, train_seqs, dev_seqs, epochs, seed=42, cfg=None, log=print, patience=None):
    """Train with Adam; when dev_seqs is given, keep the epoch with the best dev entity F1 (early stopping with 'patience'). Returns the best parameters, best epoch and history."""
    cfg = {**DEFAULT, **(cfg or {})}; rng = np.random.RandomState(seed); key = jax.random.PRNGKey(seed)
    params = init_params(key, len(enc.c2i) + 2, len(enc.tags), cfg)
    if cfg.get("finetune"): cfg.setdefault("emb_scale", 0.1); params["E_s"] = jnp.asarray(enc.E) / cfg["emb_scale"]   # trainable copy of GloVe
    opt, train_step, decode = make_fns(cfg, len(enc.tags)); ost = opt.init(params)
    encs = [enc.encode(x) for x in train_seqs]; E = jnp.asarray(enc.E); bs = cfg["bs"]
    byb = {}
    for i, e in enumerate(encs): byb.setdefault(bucket_of(e["n"]), []).append(i)
    hist, best, best_params, best_ep = [], -1, params, 0
    for ep in range(1, epochs + 1):
        t0 = time.time(); batches = []
        for L, idxs in byb.items():
            idxs = list(rng.permutation(idxs))
            batches += [(L, idxs[s:s + bs]) for s in range(0, len(idxs), bs)]
        rng.shuffle(batches); tl = 0.0
        for L, ch in batches:
            W, C, CP, Y, M, RW = make_batch(encs, ch, L, bs); key, sk = jax.random.split(key)
            params, ost, l = train_step(params, ost, E, W, C, CP, Y, M, RW, sk); tl += float(l)
        rec = {"epoch": ep, "loss": tl / len(batches), "sec": round(time.time() - t0, 1)}
        if dev_seqs is not None:
            r = evaluate(dev_seqs, predict(params, enc, decode, dev_seqs)); rec["dev_f1"] = round(r["entity"]["f1"] * 100, 2)
            rec["dev_P"] = round(r["entity"]["precision"] * 100, 2); rec["dev_R"] = round(r["entity"]["recall"] * 100, 2)
            if rec["dev_f1"] > best: best, best_params, best_ep = rec["dev_f1"], params, ep
        hist.append(rec); log(rec)
        if patience and dev_seqs is not None and ep - best_ep >= patience: break
    if dev_seqs is None: best_params, best_ep = params, epochs
    return best_params, best_ep, hist, decode
