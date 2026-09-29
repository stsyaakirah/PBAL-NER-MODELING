"""Improving the Bi-LSTM CRF under the TWITTER-ONLY protocol (seed 42, training on train only).
Every variant is trained on train, stopped on dev and compared on dev. The variant (and the O-bias delta) are chosen on DEV ONLY.
Test scores are recorded for every variant for transparency but are never used to choose."""
import json, time, os, pickle, numpy as np
import jax, jax.numpy as jnp
from wnut_data import load
from features import get_pos
from bilstm_crf import *
from evalx import evaluate, error_breakdown
D = {s: load(s) for s in ("train", "dev", "test")}; ALL = D["train"] + D["dev"] + D["test"]
for s in D:
    for x, p in zip(D[s], get_pos(s, D[s])): x["pos"] = p              # NLTK POS tags, used by the POS variant
OUT = "results/improve_bilstm.json"; R = {"variants": []}; STORE = {}
save = lambda: json.dump(R, open(OUT, "w"), indent=1, default=float)
pct = lambda x: round(100 * x, 2)
brief = lambda ev: {"P": pct(ev["entity"]["precision"]), "R": pct(ev["entity"]["recall"]), "F1": pct(ev["entity"]["f1"]), "surf": pct(ev["surface"]["f1"])}

ckpt = lambda name: f"results/ckpt_{name[:2]}.pkl"          # each finished variant is stored, so an interrupted run can resume

def run(name, cfg, pos=False):
    """Train one variant (seed 42, early stopping on dev, at most 35 epochs, patience 8) and record dev and test scores."""
    enc = Encoder(D["train"], ALL, pos=pos); cfg = dict(cfg)
    if pos: cfg.update(pos_dim=12, n_pos=enc.n_pos)
    if os.path.exists(ckpt(name)):                          # resume: reload the trained parameters instead of retraining
        blob = pickle.load(open(ckpt(name), "rb")); p = jax.tree_util.tree_map(jnp.asarray, blob["params"]); rec = blob["rec"]
        full = {**DEFAULT, **cfg}
        if full.get("finetune"): full.setdefault("emb_scale", 0.1)
        STORE[name] = (p, enc, make_fns(full, len(enc.tags))[2], full, pos); R["variants"].append(rec); save(); print("RESUMED", name, flush=True); return rec
    t0 = time.time()
    p, be, hist, dec = train(enc, D["train"], D["dev"], 35, seed=42, cfg=cfg, patience=8, log=lambda r: print(name, r["epoch"], r["dev_f1"], r["dev_P"], r["dev_R"], flush=True))
    best = max(hist, key=lambda h: h["dev_f1"]); rt = evaluate(D["test"], predict(p, enc, dec, D["test"]))
    rec = {"name": name, "cfg": {k: v for k, v in cfg.items() if k != "n_pos"}, "pos": pos, "best_epoch": be, "dev_F1": best["dev_f1"], "dev_P": best["dev_P"], "dev_R": best["dev_R"], "test": brief(rt), "epochs_run": len(hist), "sec": round(time.time() - t0), "curve": [h["dev_f1"] for h in hist]}
    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, p), "rec": rec}, open(ckpt(name), "wb"))
    STORE[name] = (p, enc, dec, cfg, pos); R["variants"].append(rec); save(); print("DONE", name, "dev", rec["dev_F1"], "test", rec["test"]["F1"], flush=True); return rec

v0 = run("V0 baseline (Twitter only)", {})
v1 = run("V1 word dropout 0.1", {"word_dropout": 0.1})
v2 = run("V2 fine-tuned GloVe (learning rate x0.1)", {"finetune": True, "emb_scale": 0.1})
v3 = run("V3 hidden size 256", {"hidden": 256})
v4 = run("V4 POS-tag input", {}, pos=True)
win = [v for v in (v1, v2, v3, v4) if v["dev_F1"] - v0["dev_F1"] >= 0.3]           # rule fixed in advance: keep a change if it gains at least 0.3 dev F1
R["combination_rule"] = "keep every single change that gains at least 0.3 F1 on dev; combine them if there are two or more"; R["winners"] = [v["name"] for v in win]
if len(win) >= 2:
    cfg = {}; pos = False
    for v in win: cfg.update({k: val for k, val in v["cfg"].items() if k not in ("pos_dim",)}); pos = pos or v["pos"]
    run("V5 combination: " + " + ".join(v["name"].split()[0] for v in win), cfg, pos=pos)
final = max(R["variants"], key=lambda v: v["dev_F1"]); R["final_variant"] = final["name"]; save()

def sweep(name):
    """O-bias sweep: subtract delta from the O score at decoding time; scored on dev (selection) and test (disclosure)."""
    p, enc, dec, cfg, pos = STORE[name]; rows = []
    for delta in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        dv = evaluate(D["dev"], predict(p, enc, dec, D["dev"], delta=delta)); ts = evaluate(D["test"], predict(p, enc, dec, D["test"], delta=delta))
        rows.append({"delta": delta, "dev": brief(dv), "test": brief(ts)}); print("sweep", name, delta, rows[-1]["dev"]["F1"], rows[-1]["test"]["F1"], flush=True)
    best = max(rows, key=lambda r: (r["dev"]["F1"], -r["delta"])); return rows, best["delta"]
R["sweep"] = {}
for name in {v0["name"], final["name"]}:
    rows, d = sweep(name); R["sweep"][name] = {"rows": rows, "best_delta": d}; save()
for tag, name in (("base", v0["name"]), ("final", final["name"])):
    p, enc, dec, cfg, pos = STORE[name]; d = R["sweep"][name]["best_delta"]
    pt0 = predict(p, enc, dec, D["test"], delta=0.0); pt1 = predict(p, enc, dec, D["test"], delta=d)
    json.dump(pt0, open(f"results/pred_to_bilstm_{tag}_nobias_test.json", "w")); json.dump(pt1, open(f"results/pred_to_bilstm_{tag}_test.json", "w"))
    ev1 = evaluate(D["test"], pt1); R[tag] = {"variant": name, "delta": d, "test": ev1, "test_errors": error_breakdown(D["test"], pt1), "test_nobias": evaluate(D["test"], pt0)}
    dvp = predict(p, enc, dec, D["dev"], delta=d); R[tag]["dev"] = evaluate(D["dev"], dvp); save()
print("IMPROVE done", flush=True)
