"""Model NER CRF (sklearn-crfsuite) untuk WNUT17.

Fitur dari ner/features.py (jendela konteks +/-2, affix, shape, POS).
Latih di train, laporkan di dev dan test dengan F1 entity + F1 surface,
confusion matrix per tipe, dan contoh prediksi benar/salah.

Jalankan:  python train_crf.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn_crfsuite import CRF

from ner import evaluate as ev
from ner.data import ENTITY_TYPES, load_splits
from ner.features import POS_AVAILABLE, dataset_features

SEED = 42
OUT = Path(__file__).resolve().parent / "outputs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c1", type=float, default=0.0, help="regularisasi L1 (0.0 terbaik di dev)")
    ap.add_argument("--c2", type=float, default=0.1, help="regularisasi L2")
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--no-pos", action="store_true", help="matikan fitur POS")
    args = ap.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    use_pos = POS_AVAILABLE and not args.no_pos

    print(f"[CRF] memuat data ... (POS aktif: {use_pos})")
    data = load_splits()
    X, y, toks = {}, {}, {}
    for split in ("train", "dev", "test"):
        X[split], y[split], toks[split] = dataset_features(data[split], use_pos=use_pos)
        print(f"  {split}: {len(X[split])} kalimat")

    print(f"[CRF] melatih (c1={args.c1}, c2={args.c2}, max_iter={args.max_iter}) ...")
    crf = CRF(algorithm="lbfgs", c1=args.c1, c2=args.c2,
              max_iterations=args.max_iter, all_possible_transitions=True)
    crf.fit(X["train"], y["train"])

    results = {}
    for split in ("dev", "test"):
        pred = crf.predict(X[split])
        ent = ev.entity_scores(y[split], pred)
        surf = ev.surface_scores(toks[split], y[split], pred)
        results[split] = {"entity": ent, "surface": surf}
        print(f"\n===== {split.upper()} =====")
        print(f"F1 entity : P={ent['precision']:.4f} R={ent['recall']:.4f} F1={ent['f1']:.4f}")
        print(f"F1 surface: P={surf['precision']:.4f} R={surf['recall']:.4f} F1={surf['f1']:.4f}")
        print("\nLaporan per tipe (entity, strict):")
        print(ev.entity_report(y[split], pred))
        if split == "test":
            print("Confusion matrix per tipe (span):")
            print(ev.format_confusion(ev.confusion_by_type(toks[split], y[split], pred, ENTITY_TYPES), ENTITY_TYPES))
            corr, wrong = ev.error_examples(toks[split], y[split], pred, n=5)
            print("\nContoh BENAR:")
            for t, g, p in corr[:3]:
                print("  ", " ".join(t)[:90])
                print("     gold:", ev.render_spans(t, g), "| pred:", ev.render_spans(t, p))
            print("Contoh SALAH:")
            for t, g, p in wrong[:5]:
                print("  ", " ".join(t)[:90])
                print("     gold:", ev.render_spans(t, g), "| pred:", ev.render_spans(t, p))

    (OUT / "crf_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[CRF] tersimpan: {OUT / 'crf_results.json'}")


if __name__ == "__main__":
    main()
