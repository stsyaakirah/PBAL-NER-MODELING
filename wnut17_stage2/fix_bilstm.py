"""Rerun of the Bi-LSTM CRF after the normalization fix: early stopping on dev (train only), then train+dev for the best epoch x 3 seeds."""
import json
from wnut_data import load
from bilstm_crf import *
from evalx import evaluate, error_breakdown
D = {s: load(s) for s in ["train", "dev", "test"]}; ALL = D["train"] + D["dev"] + D["test"]
R = {}; save = lambda: json.dump(R, open("results/fix_bilstm.json", "w"), indent=1, default=float)
enc = Encoder(D["train"], ALL)
p, be, hist, dec = train(enc, D["train"], D["dev"], 35, seed=42, log=print, patience=8)          # A: choose epoch on dev
pt = predict(p, enc, dec, D["test"]); R["A"] = {"best_epoch": be, "history": hist, "dev_F1": max(h["dev_f1"] for h in hist), "test": evaluate(D["test"], pt)}; save()
print("A done", be, R["A"]["dev_F1"], round(R["A"]["test"]["entity"]["f1"] * 100, 2), flush=True)
R["B"] = {"epochs": be, "runs": []}
for seed in (42, 43, 44):                                                                          # B: final models on train + dev
    enc2 = Encoder(D["train"] + D["dev"], ALL); p2, _, h2, dec2 = train(enc2, D["train"] + D["dev"], None, be, seed=seed, log=print)
    pt2 = predict(p2, enc2, dec2, D["test"]); R["B"]["runs"].append({"seed": seed, "test": evaluate(D["test"], pt2), "errors": error_breakdown(D["test"], pt2)})
    if seed == 42: json.dump(pt2, open("results/pred_fix_bilstm_test.json", "w"))
    save(); print("B seed", seed, round(R["B"]["runs"][-1]["test"]["entity"]["f1"] * 100, 2), flush=True)
print("BILSTM done", flush=True)
