import numpy as np
from collections import Counter
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score, accuracy_score
from seqeval.scheme import IOB2
from seqeval.metrics.sequence_labeling import get_entities
from wnut_data import surface_prf

TYPES = ["person", "location", "corporation", "product", "creative-work", "group"]

def evaluate(seqs, pred):
    """All metrics for one model: strict entity P/R/F1 (seqeval, IOB2), per type, surface F1, token accuracy."""
    gold = [x["labels"] for x in seqs]; toks = [x["tokens"] for x in seqs]
    rep = classification_report(gold, pred, mode="strict", scheme=IOB2, output_dict=True, zero_division=0)
    res = {"entity": {k: rep["micro avg"][k] for k in ("precision", "recall", "f1-score")},
           "per_type": {t: {k: rep[t][k] for k in ("precision", "recall", "f1-score", "support")} for t in TYPES if t in rep}}
    res["entity"] = {"precision": res["entity"]["precision"], "recall": res["entity"]["recall"], "f1": res["entity"]["f1-score"]}
    sp, sr, sf = surface_prf(toks, gold, toks, pred)
    res["surface"] = {"precision": sp, "recall": sr, "f1": sf}
    g = [l for s in gold for l in s]; p = [l for s in pred for l in s]
    res["token_acc"] = float(np.mean([a == b for a, b in zip(g, p)]))
    ent_idx = [i for i, l in enumerate(g) if l != "O"]
    tp = sum(1 for i in range(len(g)) if g[i] != "O" and g[i] == p[i])
    npred = sum(1 for l in p if l != "O"); ngold = len(ent_idx)
    tpp = tp / npred if npred else 0; tr = tp / ngold if ngold else 0
    res["token_f1_nonO"] = {"precision": tpp, "recall": tr, "f1": 2 * tpp * tr / (tpp + tr) if tpp + tr else 0}
    return res

def error_breakdown(seqs, pred):
    """Klasifikasi tiap entitas gold: exact / salah tipe / salah batas / terlewat; plus FP."""
    c = Counter(); fp_spurious = 0
    for x, p in zip(seqs, pred):
        G = set(get_entities(x["labels"])); P = set(get_entities(p))
        for (t, b, e) in G:
            if (t, b, e) in P: c["benar"] += 1; continue
            overl = [(pt, pb, pe) for (pt, pb, pe) in P if not (pe < b or pb > e)]
            if not overl: c["terlewat"] += 1
            elif any(pt == t for (pt, _, _) in overl): c["salah_batas"] += 1
            elif any((pb, pe) == (b, e) for (_, pb, pe) in overl): c["salah_tipe"] += 1
            else: c["salah_batas_dan_tipe"] += 1
        for (t, b, e) in P:
            if (t, b, e) in G: continue
            if not any(not (ge < b or gb > e) for (_, gb, ge) in G): fp_spurious += 1
    c["prediksi_palsu"] = fp_spurious
    return dict(c)


def type_confusion(seqs, pred):
    """Entity-level confusion matrix (Stage 2 requirement: one per model).
    Rows: gold entity type, plus SPURIOUS (predicted spans that overlap no gold span).
    Columns: predicted type when the span boundaries match exactly, 'wrong boundary' when a predicted span only overlaps
    the gold span, and 'missed' when nothing overlaps. Returns (matrix, row_labels, col_labels)."""
    rows = TYPES + ["SPURIOUS"]; cols = TYPES + ["wrong boundary", "missed"]
    M = np.zeros((len(rows), len(cols)), dtype=int)
    for x, p in zip(seqs, pred):
        G, Pd = get_entities(x["labels"]), get_entities(p)
        for (t, b, e) in G:
            exact = [pt for (pt, pb, pe) in Pd if (pb, pe) == (b, e)]
            if exact: M[rows.index(t), cols.index(exact[0])] += 1
            elif any(not (pe < b or pb > e) for (_, pb, pe) in Pd): M[rows.index(t), cols.index("wrong boundary")] += 1
            else: M[rows.index(t), cols.index("missed")] += 1
        for (pt, pb, pe) in Pd:
            if not any(not (pe < gb or pb > ge) for (_, gb, ge) in G): M[rows.index("SPURIOUS"), cols.index(pt)] += 1
    return M, rows, cols
