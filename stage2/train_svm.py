"""Model NER SVM (scikit-learn LinearSVC, one-vs-rest) untuk WNUT17.

Klasifikasi per token: fitur yang sama dengan CRF (ner/features.py) diubah menjadi
vektor sparse one-hot lewat DictVectorizer. Karena label O > 92% token, dipakai
class_weight='balanced'. Prediksi diperbaiki dengan BIO repair (I-X tanpa B-X -> B-X)
sebelum dievaluasi, agar urutan BIO valid.

Jalankan:  python train_svm.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC

from ner import evaluate as ev
from ner.data import ENTITY_TYPES, load_splits
from ner.features import POS_AVAILABLE, dataset_features

SEED = 42
OUT = Path(__file__).resolve().parent / "outputs"


def flatten(X, y):
    fx = [f for sent in X for f in sent]
    fy = [l for sent in y for l in sent]
    return fx, fy


def regroup(flat_pred, sent_lengths):
    """Kembalikan prediksi datar menjadi list-per-kalimat, lalu BIO repair."""
    out, k = [], 0
    for n in sent_lengths:
        out.append(ev.bio_repair(list(flat_pred[k:k + n])))
        k += n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--C", type=float, default=0.1, help="parameter regularisasi SVM")
    ap.add_argument("--no-pos", action="store_true")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    use_pos = POS_AVAILABLE and not args.no_pos

    print(f"[SVM] memuat data ... (POS aktif: {use_pos})")
    data = load_splits()
    X, y, toks = {}, {}, {}
    for split in ("train", "dev", "test"):
        X[split], y[split], toks[split] = dataset_features(data[split], use_pos=use_pos)

    print("[SVM] vektorisasi fitur (DictVectorizer) ...")
    vec = DictVectorizer(sparse=True)
    fxtr, fytr = flatten(X["train"], y["train"])
    Xtr = vec.fit_transform(fxtr)
    print(f"  dimensi fitur: {Xtr.shape[1]:,}")

    print(f"[SVM] melatih LinearSVC (C={args.C}, class_weight=balanced) ...")
    clf = LinearSVC(C=args.C, class_weight="balanced", max_iter=5000, random_state=SEED)
    clf.fit(Xtr, fytr)

    results = {}
    for split in ("dev", "test"):
        fx, _ = flatten(X[split], y[split])
        pred_flat = clf.predict(vec.transform(fx))
        pred = regroup(pred_flat, [len(s) for s in y[split]])
        ent = ev.entity_scores(y[split], pred)
        surf = ev.surface_scores(toks[split], y[split], pred)
        results[split] = {"entity": ent, "surface": surf}
        print(f"\n===== {split.upper()} =====")
        print(f"F1 entity : P={ent['precision']:.4f} R={ent['recall']:.4f} F1={ent['f1']:.4f}")
        print(f"F1 surface: P={surf['precision']:.4f} R={surf['recall']:.4f} F1={surf['f1']:.4f}")
        print(ev.entity_report(y[split], pred))
        if split == "test":
            print("Confusion matrix per tipe (span):")
            print(ev.format_confusion(ev.confusion_by_type(toks[split], y[split], pred, ENTITY_TYPES), ENTITY_TYPES))

    (OUT / "svm_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[SVM] tersimpan: {OUT / 'svm_results.json'}")


if __name__ == "__main__":
    main()
