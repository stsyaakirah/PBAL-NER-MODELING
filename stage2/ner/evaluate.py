"""Evaluasi NER: F1 entity (seqeval, strict) + F1 surface (khas WNUT17) + utilitas.

Dua metrik resmi WNUT17:
  - F1 entity : exact match batas dan tipe (seqeval).
  - F1 surface: pasangan unik (surface form lowercase, tipe); entitas berulang
    dihitung sekali. Metrik ini tidak ada scorer resminya di rilis data, jadi
    diimplementasikan di sini dan divalidasi terhadap submission peserta.
"""
from __future__ import annotations

from collections import Counter

from seqeval.metrics import classification_report as seq_report
from seqeval.metrics import f1_score, precision_score, recall_score
from seqeval.scheme import IOB2


def spans_from_labels(labels: list[str]) -> list[tuple[int, int, str]]:
    """Ekstrak span (start, end_exclusive, tipe) dari satu urutan label BIO."""
    spans, start, typ = [], None, None
    for i, lab in enumerate(labels + ["O"]):
        if lab.startswith("B-"):
            if typ is not None:
                spans.append((start, i, typ))
            start, typ = i, lab[2:]
        elif lab.startswith("I-") and typ == lab[2:]:
            continue
        else:  # O atau I- yang tidak cocok -> tutup span berjalan
            if typ is not None:
                spans.append((start, i, typ))
            start, typ = None, None
    return spans


def bio_repair(labels: list[str]) -> list[str]:
    """Perbaiki urutan BIO tak valid dari classifier per-token (mis. SVM).

    I-X yang tidak didahului B-X/I-X diubah menjadi B-X.
    """
    fixed, prev = [], "O"
    for lab in labels:
        if lab.startswith("I-"):
            typ = lab[2:]
            if not (prev == f"B-{typ}" or prev == f"I-{typ}"):
                lab = f"B-{typ}"
        fixed.append(lab)
        prev = lab
    return fixed


def entity_scores(y_true: list[list[str]], y_pred: list[list[str]]) -> dict:
    """F1/precision/recall level entity (micro) memakai seqeval strict IOB2."""
    return {
        "precision": precision_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0),
        "recall": recall_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0),
        "f1": f1_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0),
    }


def entity_report(y_true: list[list[str]], y_pred: list[list[str]]) -> str:
    return seq_report(y_true, y_pred, mode="strict", scheme=IOB2, digits=4, zero_division=0)


def _surface_set(seqs_tokens, seqs_labels) -> Counter:
    """Kumpulan pasangan unik (surface_form_lower, tipe) sebagai Counter (untuk contoh)."""
    pairs = Counter()
    for toks, labs in zip(seqs_tokens, seqs_labels):
        for start, end, typ in spans_from_labels(labs):
            surface = " ".join(toks[start:end]).lower()
            pairs[(surface, typ)] += 1
    return pairs


def surface_scores(tokens, y_true, y_pred) -> dict:
    """F1 surface: precision/recall/F1 atas himpunan pasangan unik (surface, tipe)."""
    gold = set(_surface_set(tokens, y_true))
    pred = set(_surface_set(tokens, y_pred))
    tp = len(gold & pred)
    p = tp / len(pred) if pred else 0.0
    r = tp / len(gold) if gold else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "n_pred": len(pred), "n_gold": len(gold)}


def confusion_by_type(tokens, y_true, y_pred, types) -> dict:
    """Confusion antar-tipe pada level span (match batas persis, tipe bisa beda).

    Kembalikan matriks dict[gold_type][pred_type] plus baris/kolom 'O' untuk
    span yang tak terdeteksi (miss) atau salah-deteksi (spurious).
    """
    labels = types + ["O"]
    mat = {g: {p: 0 for p in labels} for g in labels}
    for toks, gt, pr in zip(tokens, y_true, y_pred):
        gold = {(s, e): t for s, e, t in spans_from_labels(gt)}
        pred = {(s, e): t for s, e, t in spans_from_labels(pr)}
        for span, gtyp in gold.items():
            ptyp = pred.get(span, "O")  # O = tidak terdeteksi dengan batas sama
            mat[gtyp][ptyp] += 1
        for span, ptyp in pred.items():
            if span not in gold:
                mat["O"][ptyp] += 1  # span palsu (spurious)
    return mat


def format_confusion(mat, types) -> str:
    labels = types + ["O"]
    short = {t: t[:6] for t in labels}
    head = "gold\\pred".ljust(14) + "".join(short[p].rjust(8) for p in labels)
    lines = [head]
    for g in labels:
        lines.append(short[g].ljust(14) + "".join(str(mat[g][p]).rjust(8) for p in labels))
    lines.append("(baris O = span palsu; kolom O = entitas gold yang tidak terdeteksi)")
    return "\n".join(lines)


def error_examples(tokens, y_true, y_pred, n=8):
    """Kembalikan (benar, salah): kalimat dengan prediksi span cocok / tidak."""
    correct, wrong = [], []
    for toks, gt, pr in zip(tokens, y_true, y_pred):
        g = set(spans_from_labels(gt))
        p = set(spans_from_labels(pr))
        if not g and not p:
            continue
        rec = (toks, sorted(g), sorted(p))
        (correct if g == p else wrong).append(rec)
    return correct[:n], wrong[:n]


def render_spans(toks, spans):
    return ", ".join(f"'{' '.join(toks[s:e])}'->{t}" for s, e, t in spans) or "(none)"
