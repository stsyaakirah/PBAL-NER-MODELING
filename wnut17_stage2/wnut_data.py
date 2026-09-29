"""Loader dan utilitas evaluasi untuk WNUT17 (CoNLL dua kolom, skema BIO)."""
import os
from seqeval.metrics.sequence_labeling import get_entities

DATA = os.environ.get("WNUT_DIR") or (os.path.join(os.path.dirname(os.path.abspath(__file__)), "data") if os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")) else "/home/claude/emerging_entities_17")  # folder with the three official WNUT17 files
FILES = {
    "train": "wnut17train.conll",
    "dev": "emerging.dev.conll",
    "test": "emerging.test.annotated",
}

def read_conll(path, split_on_tab_only=False):
    """Return list of dict(tokens, labels, doc). Train: baris tab-only = batas kalimat."""
    seqs, toks, labs, doc = [], [], [], 0
    def flush():
        nonlocal toks, labs
        if toks:
            seqs.append({"tokens": toks, "labels": labs, "doc": doc})
        toks, labs = [], []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line == "":            # batas dokumen
                flush(); doc += 1
            elif line.strip() == "":  # tab-only: batas kalimat dalam dokumen
                flush()
            else:
                parts = line.split("\t")
                toks.append(parts[0]); labs.append(parts[-1])
    flush()
    return seqs

def load(split):
    """Load one official split ('train', 'dev' or 'test') as a list of {tokens, labels, doc} sequences."""
    return read_conll(os.path.join(DATA, FILES[split]))

def entities(labels):
    """Entity spans (type, start, end) of a BIO label sequence (seqeval)."""
    return get_entities(labels)  # (type, start, end)

def surface_set(seqs_tokens, seqs_labels):
    """Pasangan unik (surface form huruf kecil, tipe) untuk F1 surface."""
    out = set()
    for toks, labs in zip(seqs_tokens, seqs_labels):
        for t, s, e in get_entities(labs):
            out.add((" ".join(toks[s:e + 1]).lower(), t))
    return out

def surface_prf(gold_tokens, gold_labels, pred_tokens, pred_labels):
    """Shared-task surface F1: precision, recall and F1 over unique (lowercased surface form, type) pairs."""
    g = surface_set(gold_tokens, gold_labels)
    p = surface_set(pred_tokens, pred_labels)
    tp = len(g & p)
    pr = tp / len(p) if p else 0.0
    rc = tp / len(g) if g else 0.0
    f = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
    return pr, rc, f
