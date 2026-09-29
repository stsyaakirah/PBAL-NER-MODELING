"""Model NER Bi-LSTM CRF (PyTorch) untuk WNUT17.

Arsitektur: embedding kata (bentuk ternormalisasi) + representasi karakter (char-BiLSTM)
-> Bi-LSTM kata -> linear -> lapis CRF (torchcrf). Embedding kata dilatih dari awal
secara default; opsi --glove memuat vektor GloVe/word2vec (format text) bila tersedia.

CPU-friendly: ukuran kecil, early stopping pada F1 entity dev. Seed 42.

Jalankan:  python train_bilstm_crf.py --epochs 20
           python train_bilstm_crf.py --glove /path/glove.txt --epochs 30
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from ner import evaluate as ev
from ner.data import BIO_LABELS, ENTITY_TYPES, load_splits, normalize

SEED = 42
OUT = Path(__file__).resolve().parent / "outputs"
PAD, UNK = "<PAD>", "<UNK>"


def set_seed(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


# ----------------------------- Vokabulari -----------------------------
def build_vocab(train):
    wc, cc = Counter(), Counter()
    for sent in train:
        for tok, _ in sent:
            wc[normalize(tok)] += 1
            cc.update(tok)
    words = [PAD, UNK] + sorted(wc)
    chars = [PAD, UNK] + sorted(cc)
    w2i = {w: i for i, w in enumerate(words)}
    c2i = {c: i for i, c in enumerate(chars)}
    l2i = {l: i for i, l in enumerate(BIO_LABELS)}
    return w2i, c2i, l2i


def load_glove(path, w2i, dim):
    """Isi matriks embedding dari file text (kata v1 v2 ...). Kata tak ada -> acak."""
    emb = np.random.normal(0, 0.1, (len(w2i), dim)).astype("float32")
    emb[0] = 0  # PAD
    found = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            w = parts[0]
            if w in w2i:
                emb[w2i[w]] = np.asarray(parts[1:], dtype="float32")
                found += 1
    print(f"  GloVe: {found:,}/{len(w2i):,} kata cocok")
    return emb


# ----------------------------- Encoding -----------------------------
def encode(sents, w2i, c2i, l2i, max_wlen=20):
    out = []
    for sent in sents:
        toks = [t for t, _ in sent]
        wids = [w2i.get(normalize(t), w2i[UNK]) for t in toks]
        cids = [[c2i.get(c, c2i[UNK]) for c in t[:max_wlen]] or [c2i[PAD]] for t in toks]
        lids = [l2i[l] for _, l in sent]
        out.append((toks, wids, cids, lids))
    return out


def make_batches(data, batch_size, shuffle=False):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    for k in range(0, len(idx), batch_size):
        yield [data[j] for j in idx[k:k + batch_size]]


def pad_batch(batch, device):
    B = len(batch)
    slen = [len(x[1]) for x in batch]
    S = max(slen)
    wlen = max(len(c) for x in batch for c in x[2])
    w = torch.zeros(B, S, dtype=torch.long)
    c = torch.zeros(B, S, wlen, dtype=torch.long)
    y = torch.zeros(B, S, dtype=torch.long)
    mask = torch.zeros(B, S, dtype=torch.bool)
    for i, (_, wids, cids, lids) in enumerate(batch):
        n = len(wids)
        w[i, :n] = torch.tensor(wids)
        y[i, :n] = torch.tensor(lids)
        mask[i, :n] = 1
        for j, ch in enumerate(cids):
            c[i, j, :len(ch)] = torch.tensor(ch)
    return (w.to(device), c.to(device), y.to(device), mask.to(device),
            torch.tensor(slen).to(device))


# ----------------------------- Model -----------------------------
class BiLSTMCRF(nn.Module):
    def __init__(self, n_words, n_chars, n_labels, word_dim=100, char_dim=30,
                 char_hidden=25, hidden=200, dropout=0.5, pretrained=None):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_dim, padding_idx=0)
        if pretrained is not None:
            self.word_emb.weight.data.copy_(torch.from_numpy(pretrained))
        self.char_emb = nn.Embedding(n_chars, char_dim, padding_idx=0)
        self.char_lstm = nn.LSTM(char_dim, char_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.word_lstm = nn.LSTM(word_dim + 2 * char_hidden, hidden // 2,
                                 batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden, n_labels)
        self.crf = CRF(n_labels, batch_first=True)

    def _char_repr(self, c):
        B, S, W = c.shape
        ce = self.char_emb(c.view(B * S, W))
        _, (h, _) = self.char_lstm(ce)
        h = torch.cat([h[0], h[1]], dim=-1)  # (B*S, 2*char_hidden)
        return h.view(B, S, -1)

    def _emit(self, w, c):
        we = self.word_emb(w)
        ch = self._char_repr(c)
        x = self.dropout(torch.cat([we, ch], dim=-1))
        out, _ = self.word_lstm(x)
        return self.fc(self.dropout(out))

    def loss(self, w, c, y, mask):
        return -self.crf(self._emit(w, c), y, mask=mask, reduction="mean")

    def decode(self, w, c, mask):
        return self.crf.decode(self._emit(w, c), mask=mask)


# ----------------------------- Latih / evaluasi -----------------------------
def evaluate(model, data, batch_size, device, i2l):
    model.eval()
    toks_all, yt, yp = [], [], []
    with torch.no_grad():
        for batch in make_batches(data, batch_size):
            w, c, y, mask, _ = pad_batch(batch, device)
            paths = model.decode(w, c, mask)
            for (toks, _, _, lids), path in zip(batch, paths):
                toks_all.append(toks)
                yt.append([i2l[i] for i in lids])
                yp.append([i2l[i] for i in path])
    ent = ev.entity_scores(yt, yp)
    surf = ev.surface_scores(toks_all, yt, yp)
    return ent, surf, (toks_all, yt, yp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--word-dim", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=200)
    ap.add_argument("--glove", type=str, default=None, help="path file embedding text (opsional)")
    args = ap.parse_args()

    set_seed()
    OUT.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BiLSTM-CRF] device={device}")

    data = load_splits()
    w2i, c2i, l2i = build_vocab(data["train"])
    i2l = {i: l for l, i in l2i.items()}
    enc = {s: encode(data[s], w2i, c2i, l2i) for s in ("train", "dev", "test")}
    print(f"  vocab kata={len(w2i):,} char={len(c2i)} label={len(l2i)}")

    pretrained = None
    if args.glove:
        pretrained = load_glove(args.glove, w2i, args.word_dim)

    model = BiLSTMCRF(len(w2i), len(c2i), len(l2i), word_dim=args.word_dim,
                      hidden=args.hidden, pretrained=pretrained).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for batch in make_batches(enc["train"], args.batch_size, shuffle=True):
            w, c, y, mask, _ = pad_batch(batch, device)
            opt.zero_grad()
            loss = model.loss(w, c, y, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item()
        dev_ent, dev_surf, _ = evaluate(model, enc["dev"], args.batch_size, device, i2l)
        print(f"  epoch {ep:2d} | loss {total:7.1f} | dev entity F1 {dev_ent['f1']:.4f} "
              f"surface F1 {dev_surf['f1']:.4f}")
        if dev_ent["f1"] > best_f1:
            best_f1 = dev_ent["f1"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    print(f"\n[BiLSTM-CRF] model terbaik: dev entity F1 {best_f1:.4f}")

    results = {}
    for split in ("dev", "test"):
        ent, surf, (toks, yt, yp) = evaluate(model, enc[split], args.batch_size, device, i2l)
        results[split] = {"entity": ent, "surface": surf}
        print(f"\n===== {split.upper()} =====")
        print(f"F1 entity : P={ent['precision']:.4f} R={ent['recall']:.4f} F1={ent['f1']:.4f}")
        print(f"F1 surface: P={surf['precision']:.4f} R={surf['recall']:.4f} F1={surf['f1']:.4f}")
        print(ev.entity_report(yt, yp))
        if split == "test":
            print("Confusion matrix per tipe (span):")
            print(ev.format_confusion(ev.confusion_by_type(toks, yt, yp, ENTITY_TYPES), ENTITY_TYPES))

    (OUT / "bilstm_crf_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[BiLSTM-CRF] tersimpan: {OUT / 'bilstm_crf_results.json'}")


if __name__ == "__main__":
    main()
