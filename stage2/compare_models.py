"""Rangkum hasil ketiga model menjadi satu tabel perbandingan.

Baca outputs/{crf,svm,bilstm_crf}_results.json (jalankan ketiga train_*.py dulu),
lalu cetak tabel F1 entity dan F1 surface untuk dev dan test.

Jalankan:  python compare_models.py
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "outputs"
MODELS = [("CRF", "crf_results.json"), ("SVM", "svm_results.json"),
          ("BiLSTM-CRF", "bilstm_crf_results.json")]


def main():
    rows = []
    for name, fname in MODELS:
        path = OUT / fname
        if not path.exists():
            print(f"(lewati {name}: {fname} belum ada, jalankan train dulu)")
            continue
        r = json.loads(path.read_text())
        rows.append((name, r))

    if not rows:
        return
    header = f"{'Model':12s} | {'dev entF1':>9s} {'dev surF1':>9s} | {'test entF1':>10s} {'test surF1':>10s}"
    print(header)
    print("-" * len(header))
    for name, r in rows:
        print(f"{name:12s} | {r['dev']['entity']['f1']:9.4f} {r['dev']['surface']['f1']:9.4f} "
              f"| {r['test']['entity']['f1']:10.4f} {r['test']['surface']['f1']:10.4f}")
    print("\nPembanding paper (test, F1 entity): baseline shared task s.d. UH-RiTUAL 0.4186.")


if __name__ == "__main__":
    main()
