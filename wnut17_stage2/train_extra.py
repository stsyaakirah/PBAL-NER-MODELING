"""Opsi 1: perbesar ensemble Bi-LSTM CRF (Twitter-only, seed 42).

Menambah tiga anggota yang beda STRUKTUR ke enam varian lama (V0-V5), lalu
menyimpan bobotnya sebagai results/ckpt_V6/V7/V8.pkl dan menambahkan catatannya
ke results/improve_bilstm.json, sehingga ensemble_bilstm.py otomatis merata-rata
kesembilan model.

Resep ditetapkan SEBELUM melihat skor test (aturan keras: pilih di dev saja):
  V6  char-CNN lebih lebar   (char_filters=100, dari 50)
  V7  embedding karakter lebih kaya (char_dim=50, dari 30)
  V8  ablasi tanpa kapitalisasi     (cap_dim=0; hanya GloVe + char)

Skor dev dan test dicatat untuk semua varian demi transparansi; test tidak
pernah dipakai untuk memilih apa pun. Jalankan setelah improve_bilstm.py.
"""
import json, time, os, pickle, numpy as np
import jax, jax.numpy as jnp
from wnut_data import load
from bilstm_crf import Encoder, train, predict, make_fns, DEFAULT
from evalx import evaluate

D = {s: load(s) for s in ("train", "dev", "test")}
ALL = D["train"] + D["dev"] + D["test"]
OUT = "results/improve_bilstm.json"
R = json.load(open(OUT))                      # varian lama sudah ada di sini
existing = {v["name"] for v in R["variants"]}

pct = lambda x: round(100 * x, 2)
brief = lambda ev: {"P": pct(ev["entity"]["precision"]), "R": pct(ev["entity"]["recall"]),
                    "F1": pct(ev["entity"]["f1"]), "surf": pct(ev["surface"]["f1"])}
ckpt = lambda name: f"results/ckpt_{name[:2]}.pkl"


def run(name, cfg):
    """Latih satu varian (seed 42, early stopping di dev, maks 35 epoch, patience 8)."""
    if name in existing:
        print("SKIP (sudah ada di json):", name, flush=True); return
    if os.path.exists(ckpt(name)):
        blob = pickle.load(open(ckpt(name), "rb"))
        R["variants"].append(blob["rec"]); json.dump(R, open(OUT, "w"), indent=1, default=float)
        print("RESUMED", name, flush=True); return
    enc = Encoder(D["train"], ALL, pos=False)
    t0 = time.time()
    p, be, hist, dec = train(enc, D["train"], D["dev"], 35, seed=42, cfg=cfg, patience=8,
                             log=lambda r: print(name, r["epoch"], r.get("dev_f1"), flush=True))
    best = max(hist, key=lambda h: h["dev_f1"])
    rt = evaluate(D["test"], predict(p, enc, dec, D["test"]))
    rec = {"name": name, "cfg": dict(cfg), "pos": False, "best_epoch": be,
           "dev_F1": best["dev_f1"], "dev_P": best["dev_P"], "dev_R": best["dev_R"],
           "test": brief(rt), "epochs_run": len(hist), "sec": round(time.time() - t0),
           "curve": [h["dev_f1"] for h in hist]}
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, p), "rec": rec}, open(ckpt(name), "wb"))
    R["variants"].append(rec); json.dump(R, open(OUT, "w"), indent=1, default=float)
    print("DONE", name, "dev", rec["dev_F1"], "test", rec["test"]["F1"], flush=True)


run("V6 wider char-CNN (100 filters)", {"char_filters": 100})
run("V7 richer char embeddings (dim 50)", {"char_dim": 50})
run("V8 no-capitalization ablation", {"cap_dim": 0})
print("EXTRA done. total variants now:", len(R["variants"]), flush=True)
