"""Section B: Bi-LSTM CRF BASELINE on the pooled 80/10/10 split (data_partB), seed 42.

Baseline architecture only (frozen GloVe + char-CNN + capitalization embedding, 128 units/dir,
dropout 0.5, Adam 1e-3, batch 32), early stopping on the new dev split (patience 8, max 35).
No word-dropout / fine-tune / hidden-256 / POS / O-bias variants: those were tuned for the
Section A domain-shift problem, which the pooled split does not have. New test scored once.
"""
import os, json, numpy as np
from wnut_data import read_conll
from bilstm_crf import Encoder, train, predict, DEFAULT
from evalx import evaluate, error_breakdown

os.makedirs("results_partB", exist_ok=True)
D = {sp: read_conll(f"data_partB/{sp}.conll") for sp in ("train", "dev", "test")}
ALL = D["train"] + D["dev"] + D["test"]
brief = lambda r: {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2),
                   "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}

enc = Encoder(D["train"], ALL)                       # vocab/char from NEW train; GloVe keys from full pool
params, be, hist, dec = train(enc, D["train"], D["dev"], 35, seed=42, cfg=dict(DEFAULT), patience=8,
                              log=lambda r: print("bilstm", r["epoch"], r.get("dev_f1"), flush=True))
dev_pred = predict(params, enc, dec, D["dev"])
test_pred = predict(params, enc, dec, D["test"])     # delta=0, baseline, no O-bias shift
R = {"best_epoch": be, "dev": evaluate(D["dev"], dev_pred), "test": evaluate(D["test"], test_pred),
     "test_errors": error_breakdown(D["test"], test_pred), "dev_curve": [h.get("dev_f1") for h in hist]}
json.dump(test_pred, open("results_partB/pred_partB_bilstm_test.json", "w"))
json.dump(dev_pred, open("results_partB/pred_partB_bilstm_dev.json", "w"))
json.dump(R, open("results_partB/partB_bilstm.json", "w"), indent=1, default=float)
print("Bi-LSTM baseline  dev", brief(R["dev"]), "| test", brief(R["test"]), "| best_epoch", be, flush=True)
