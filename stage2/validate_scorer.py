"""Validasi modul evaluasi terhadap 7 submission resmi shared task WNUT17.

Karena rilis data tidak menyertakan scorer resmi, kami menghitung F1 entity dan
F1 surface sendiri, lalu memastikan hasilnya mereproduksi Tabel 2 paper. Jika
angka kami cocok dengan paper, scorer kami tepat.

Jalankan:  python validate_scorer.py
"""
from __future__ import annotations

from pathlib import Path

from ner import evaluate as ev
from ner.data import read_conll

BASE = Path(__file__).resolve().parents[1]
SUB_DIR = BASE / "submissions"
# F1 entity/surface resmi dari Tabel 2 paper (Derczynski et al., 2017)
PAPER = {
    "arcada": (39.98, 37.77), "drexel_cci": (26.30, 25.26), "flytxt": (38.35, 36.31),
    "mic-cis": (37.06, 34.25), "sjtu_adapt": (40.42, 37.62), "spinningbytes": (40.78, 39.33),
    "uh_ritual": (41.86, 40.24),
}


def align(gold, sub):
    yt, yp, toks = [], [], []
    for g, s in zip(gold, sub):
        if len(g) == len(s) and [t for t, _ in g] == [t for t, _ in s]:
            yt.append([l for _, l in g])
            yp.append([l for _, l in s])
            toks.append([t for t, _ in g])
    return yt, yp, toks


def main():
    gold = read_conll(BASE / "emerging.test.annotated")
    print(f"{'sistem':14s} {'entity (kami/paper)':24s} {'surface (kami/paper)':24s}")
    print("-" * 64)
    for name, (pe, ps) in PAPER.items():
        path = SUB_DIR / f"{name}.txt"
        if not path.exists():
            continue
        sub = read_conll(path)
        yt, yp, toks = align(gold, sub)
        if not yt:  # format kolom tidak cocok (mis. arcada tanpa kolom label)
            print(f"{name:14s} (dilewati: format tidak selaras)")
            continue
        e = ev.entity_scores(yt, yp)["f1"] * 100
        s = ev.surface_scores(toks, yt, yp)["f1"] * 100
        print(f"{name:14s} {e:6.2f} / {pe:<6.2f}          {s:6.2f} / {ps:<6.2f}")
    print("\nJika kolom 'kami' cocok dengan 'paper', modul evaluasi sudah benar.")


if __name__ == "__main__":
    main()
