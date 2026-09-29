import json, sys, time, numpy as np
from wnut_data import load
from bilstm_crf import *
from evalx import evaluate, error_breakdown

D = {s: load(s) for s in ["train", "dev", "test"]}
ALL = D["train"] + D["dev"] + D["test"]
OUT = "results/bilstm.json"; R = {}
def save(): json.dump(R, open(OUT, "w"), indent=1, default=float)
def brief(r): return {"P": round(r["entity"]["precision"] * 100, 2), "R": round(r["entity"]["recall"] * 100, 2),
                      "F1": round(r["entity"]["f1"] * 100, 2), "surfF1": round(r["surface"]["f1"] * 100, 2)}
def log(tag): return lambda rec: print(tag, {k: (round(float(v), 3) if not isinstance(v, int) else v) for k, v in rec.items()}, flush=True)
MAXEP, PAT = 35, 8

# A) BIO, train -> dev (early stopping), lalu uji pada test
enc = Encoder(D["train"], ALL)
t0 = time.time(); p, be, hist, dec = train(enc, D["train"], D["dev"], MAXEP, seed=42, log=log("A"), patience=PAT)
pt = predict(p, enc, dec, D["test"]); rt = evaluate(D["test"], pt)
R["A_bio_trainonly"] = {"best_epoch": be, "history": hist, "test": rt, "dev_F1": max(h["dev_f1"] for h in hist), "sec": round(time.time() - t0)}
json.dump(pt, open("results/pred_bilstm_trainonly_test.json", "w")); save(); print("A TEST", brief(rt), "best_ep", be, flush=True)

# B) BIO, train+dev selama best_epoch epoch -> test (protokol final); tiga seed
R["B_bio_final"] = {"epochs": be, "runs": []}
for seed in (42, 43, 44):
    enc2 = Encoder(D["train"] + D["dev"], ALL)
    p2, _, h2, dec2 = train(enc2, D["train"] + D["dev"], None, be, seed=seed, log=log(f"B{seed}"))
    pt2 = predict(p2, enc2, dec2, D["test"]); r2 = evaluate(D["test"], pt2)
    R["B_bio_final"]["runs"].append({"seed": seed, "test": r2, "errors": error_breakdown(D["test"], pt2)})
    if seed == 42: json.dump(pt2, open("results/pred_bilstm_test.json", "w"))
    f1s = [r["test"]["entity"]["f1"] * 100 for r in R["B_bio_final"]["runs"]]
    R["B_bio_final"]["mean_f1"], R["B_bio_final"]["std_f1"] = round(float(np.mean(f1s)), 2), round(float(np.std(f1s)), 2)
    save(); print("B TEST seed", seed, brief(r2), flush=True)

# C) BIOES (25 label), train -> dev, uji test (dibandingkan dengan A)
encE = Encoder(D["train"], ALL, bioes=True)
t0 = time.time(); pE, beE, hE, decE = train(encE, D["train"], D["dev"], MAXEP, seed=42, log=log("C"), patience=PAT)
ptE = predict(pE, encE, decE, D["test"]); rE = evaluate(D["test"], ptE)
R["C_bioes_trainonly"] = {"best_epoch": beE, "history": hE, "test": rE, "dev_F1": max(h["dev_f1"] for h in hE), "sec": round(time.time() - t0)}
save(); print("C TEST", brief(rE), "best_ep", beE, flush=True)
print("DONE", flush=True)
