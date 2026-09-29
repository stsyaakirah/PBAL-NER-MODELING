"""Ekstraksi fitur CRF/SVM sesuai rencana Stage 1 (jendela +-2)."""
import os, re, pickle, unicodedata
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.cluster import MiniBatchKMeans
from prep_glove import normalize, glove_keys
from wnut_data import load

STOP = set(stopwords.words("english"))
EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF\u2190-\u21FF\u2B00-\u2BFF]")

def word_shape(w):
    """Shape pattern of a token: X for upper case, x for lower case, d for digits (Xxxx, dd)."""
    s = re.sub(r"[A-Z]", "X", w)
    s = re.sub(r"[a-z]", "x", s)
    s = re.sub(r"\d", "d", s)
    return s

def short_shape(w):
    """Shape pattern with runs of the same symbol collapsed (Xxxx becomes Xx)."""
    return re.sub(r"(.)\1+", r"\1", word_shape(w))

# ---------- cache POS + cluster ----------
def get_pos(split, seqs):
    """NLTK averaged-perceptron POS tags for a split, cached on disk because tagging is slow."""
    p = f"cache_pos_{split}.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    out = [[t for _, t in nltk.pos_tag(x["tokens"])] for x in seqs]
    pickle.dump(out, open(p, "wb")); return out

def get_clusters():
    """K-Means (k = 500) cluster ID for every GloVe word needed; fitted on the 100,000 most frequent GloVe Twitter words and cached."""
    p = "cache_clusters.pkl"
    if os.path.exists(p):
        return pickle.load(open(p, "rb"))
    z = np.load("glove_sub.npz"); v = pickle.load(open("glove_vocab.pkl", "rb"))
    km = MiniBatchKMeans(n_clusters=500, random_state=42, batch_size=4096, n_init=3, max_iter=100)
    km.fit(z["top_vecs"])
    ids = km.predict(z["need_vecs"])
    d = {w: int(ids[i]) for w, i in v["need_words"].items()}
    pickle.dump(d, open(p, "wb")); return d

CLUS = None
def cluster_of(tok):
    """Cluster ID of a token (first GloVe lookup key that exists), or -1 when the token has no GloVe vector."""
    global CLUS
    if CLUS is None: CLUS = get_clusters()
    for k in glove_keys(tok):
        if k in CLUS: return CLUS[k]
    return -1

GROUPS = ["word", "shape", "affix", "social", "context", "pos", "cluster"]

def tok_feats(tokens, i, pos, groups):
    """Feature dictionary of token i; 'groups' selects which feature groups are included (used for the ablation)."""
    w = tokens[i]; lw = w.lower(); f = {"bias": 1.0}
    if "word" in groups:
        f["w.lower"] = lw; f["w.norm"] = normalize(w); f["w.stop"] = lw in STOP
    if "shape" in groups:
        f["w.shape"] = short_shape(w); f["w.title"] = w.istitle(); f["w.upper"] = w.isupper()
        f["w.digit"] = any(c.isdigit() for c in w); f["w.hyphen"] = "-" in w
    if "affix" in groups:
        for k in range(1, 5):
            f[f"pre{k}"] = lw[:k]; f[f"suf{k}"] = lw[-k:]
        f["len"] = min(len(w), 15)
    if "social" in groups:
        f["is_url"] = normalize(w) == "<URL>"; f["is_mention"] = w.startswith("@") and len(w) > 1
        f["is_hashtag"] = w.startswith("#") and len(w) > 1; f["is_emoji"] = bool(EMOJI.search(w))
    if "pos" in groups:
        f["pos"] = pos[i]
        f["pos-1"] = pos[i - 1] if i > 0 else "BOS"; f["pos+1"] = pos[i + 1] if i < len(tokens) - 1 else "EOS"
    if "cluster" in groups:
        f["clu"] = cluster_of(w)
        f["clu-1"] = cluster_of(tokens[i - 1]) if i > 0 else -2
        f["clu+1"] = cluster_of(tokens[i + 1]) if i < len(tokens) - 1 else -2
    if "context" in groups:
        for d in (-2, -1, 1, 2):
            j = i + d
            if 0 <= j < len(tokens):
                nw = tokens[j]
                f[f"w{d:+d}"] = nw.lower()
                if abs(d) == 1:
                    f[f"w{d:+d}.title"] = nw.istitle(); f[f"w{d:+d}.upper"] = nw.isupper()
            else:
                f[f"w{d:+d}"] = "BOS" if j < 0 else "EOS"
        f["bi-1"] = (tokens[i - 1].lower() if i > 0 else "BOS") + "|" + lw
        f["bi+1"] = lw + "|" + (tokens[i + 1].lower() if i < len(tokens) - 1 else "EOS")
        if i == 0: f["BOS"] = True
        if i == len(tokens) - 1: f["EOS"] = True
    return f

def sent_feats(tokens, pos, groups):
    """Feature dictionaries for every token of a sentence (the input of the CRF and the SVM)."""
    return [tok_feats(tokens, i, pos, groups) for i in range(len(tokens))]

def glove_vec_table(seqs):
    """Vektor GloVe per token (200d), nol jika tidak ada."""
    nv = np.load("glove_sub.npz")["need_vecs"]; v = pickle.load(open("glove_vocab.pkl", "rb"))["need_words"]
    mats = []
    for x in seqs:
        m = np.zeros((len(x["tokens"]), 200), dtype=np.float32)
        for i, t in enumerate(x["tokens"]):
            for k in glove_keys(t):
                if k in v: m[i] = nv[v[k]]; break
        mats.append(m)
    return mats

def bio_repair(labels):
    """Turn every orphan I-X (no B-X or I-X before it) into B-X. Tested for the SVM and finally NOT used (it lowered F1)."""
    out, prev = [], "O"
    for l in labels:
        if l.startswith("I-") and prev not in ("B-" + l[2:], "I-" + l[2:]):
            l = "B-" + l[2:]
        out.append(l); prev = l
    return out
