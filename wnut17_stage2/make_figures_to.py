"""Figures and bootstrap intervals for the Twitter-only report."""
import json, os, shutil, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from wnut_data import load
from evalx import type_confusion
R = lambda p: json.load(open("results/" + p)); IB, TC, TS, AN, IBE = R("improve_bilstm.json"), R("to_crf.json"), R("to_svm.json"), R("analysis_to.json")["group_recall"], R("ensemble_bilstm.json")
os.makedirs("report_figs_to", exist_ok=True)
for f in ("f_architecture.png", "f_ablation.png"): shutil.copy("report_figs/" + f, "report_figs_to/" + f)
NAVY, AMB, SLATE, LNAVY, INK, MUT, RED, GRN = "#12406A", "#E0A63A", "#8FA3B8", "#9DB4CB", "#1F2937", "#5B6B7C", "#B85042", "#2E7D5B"
M4 = ["CRF", "SVM", "Bi-LSTM CRF (baseline)", "Bi-LSTM CRF (ensemble)"]; COL = dict(zip(M4, [SLATE, AMB, LNAVY, NAVY])); T6 = ["person", "location", "corporation", "product", "creative-work", "group"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
save = lambda fig, n: (fig.savefig(f"report_figs_to/{n}.png", dpi=200, bbox_inches="tight", facecolor="white"), plt.close(fig))
def lab(ax, bars, fmt="{:.1f}", fs=7):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5, fmt.format(b.get_height()), ha="center", va="bottom", fontsize=fs, color=INK)
pct = lambda x: 100 * x
ev = {"CRF": TC["test"], "SVM": TS["test"], "Bi-LSTM CRF (baseline)": IB["base"]["test_nobias"], "Bi-LSTM CRF (ensemble)": IBE["test_delta0"]}
PF = {"CRF": "pred_to_crf_test.json", "SVM": "pred_to_svm_test.json", "Bi-LSTM CRF (baseline)": "pred_to_bilstm_base_nobias_test.json", "Bi-LSTM CRF (ensemble)": "pred_to_bilstm_ens_nobias_test.json"}

# ---- pipeline (Twitter-only protocol)
fig, ax = plt.subplots(figsize=(6.9, 2.75)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 45)
boxes = [("1  Data and splits", "WNUT17 official splits\ntrain = Twitter 2010 only\nCoNLL, IOB2 labels"), ("2  Reading and checks", "Tab-only and blank lines\nsplit sequences\nBIO check: 0 violations"), ("3  Normalization", "NFKC and HTML unescape\nURL, @user, digits, repeats\nlowercase lookup form"),
         ("4  Representations", "CRF and SVM: features\nBi-LSTM CRF: GloVe,\nchar-CNN, capitalization"), ("5  Protocol", "Train on train only\ntune and stop on dev\ntest once, seed 42"), ("6  Evaluation", "Entity F1 (strict), surface F1\nerror analysis\nbootstrap intervals")]
w, h, gx = 29, 19, 5.5; pos = [(1 + i * (w + gx), 25) for i in range(3)] + [(1 + i * (w + gx), 1) for i in range(3)]
for (t, d), (x, y) in zip(boxes, pos):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#F4F5F7", ec="#D9DEE5", lw=0.8)); ax.text(x + w / 2, y + h - 4.2, t, ha="center", va="center", fontsize=8, fontweight="bold", color=NAVY); ax.text(x + w / 2, y + 7.4, d, ha="center", va="center", fontsize=6.4, color=INK, linespacing=1.45)
for i in (0, 1, 3, 4): x, y = pos[i]; ax.add_patch(FancyArrowPatch((x + w + 0.4, y + h / 2), (x + w + gx - 0.4, y + h / 2), arrowstyle="-|>", mutation_scale=9, color=NAVY, lw=1.2))
xm = 1 + 2 * (w + gx) + w / 2; ax.plot([xm, xm], [24.7, 22.6], color=NAVY, lw=1.2); ax.plot([xm, pos[3][0] + w / 2], [22.6, 22.6], color=NAVY, lw=1.2); ax.add_patch(FancyArrowPatch((pos[3][0] + w / 2, 22.6), (pos[3][0] + w / 2, pos[3][1] + h + 0.4), arrowstyle="-|>", mutation_scale=9, color=NAVY, lw=1.2)); save(fig, "f_pipeline_to")

# ---- improvement ladder (dev and test F1 per step)
V = IB["variants"]; short = {"V0": "Baseline", "V1": "+ word\ndropout", "V2": "+ fine-tuned\nGloVe", "V3": "+ hidden\n256", "V4": "+ POS\ntags", "V5": "All four\ncombined"}
names = [short[v["name"][:2]] for v in V]; dev = [v["dev_F1"] for v in V]; test = [v["test"]["F1"] for v in V]
fin = IB["final"]; base = IB["base"]; sw = IB["sweep"]
def sweep_at(name, delta): return next(r for r in sw[name]["rows"] if r["delta"] == delta)
er = {r["delta"]: r for r in IBE["rows"]}; eb = er[IBE["best_delta"]]
names += ["Ensemble\nof V0 to V5", "Baseline\n+ shift", "V3 + shift\n(dev choice)", "Ensemble\n+ shift"]
dev += [er[0.0]["dev"]["F1"], sweep_at(base["variant"], base["delta"])["dev"]["F1"], sweep_at(fin["variant"], fin["delta"])["dev"]["F1"], eb["dev"]["F1"]]
test += [er[0.0]["test"]["F1"], sweep_at(base["variant"], base["delta"])["test"]["F1"], sweep_at(fin["variant"], fin["delta"])["test"]["F1"], eb["test"]["F1"]]
fig, ax = plt.subplots(figsize=(6.9, 3.0)); x = np.arange(len(names)); wd = 0.38
b1 = ax.bar(x - wd / 2, dev, wd, color=LNAVY, label="Dev F1 (used to choose)"); b2 = ax.bar(x + wd / 2, test, wd, color=NAVY, label="Test F1 (scored once, for reporting)"); lab(ax, b1, fs=5.6); lab(ax, b2, fs=5.6)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=6); ax.set_ylim(30, max(dev + test) + 6); ax.set_ylabel("F1 entity (%)"); ax.legend(loc="upper left", fontsize=7, ncol=2); ax.grid(axis="x", visible=False); save(fig, "f_ladder")

# ---- main results
fig, ax = plt.subplots(figsize=(6.6, 2.7)); x = np.arange(4); wd = 0.26
for j, (k, nm, c) in enumerate([("precision", "Precision", LNAVY), ("recall", "Recall", AMB), ("f1", "F1 entity", NAVY)]):
    b = ax.bar(x + (j - 1) * wd, [pct(ev[m]["entity"][k]) for m in M4], wd, label=nm, color=c); lab(ax, b)
ax.set_xticks(x); ax.set_xticklabels([m.replace(" (", "\n(") for m in M4], fontsize=7.6); ax.set_ylim(0, 68); ax.set_ylabel("Score (%)"); ax.legend(ncol=3, loc="upper left"); ax.grid(axis="x", visible=False); save(fig, "f_main_to")

# ---- per type
fig, ax = plt.subplots(figsize=(6.6, 2.8)); x = np.arange(6); wd = 0.2
for j, m in enumerate(M4): b = ax.bar(x + (j - 1.5) * wd, [100 * ev[m]["per_type"][t]["f1-score"] for t in T6], wd, label=m, color=COL[m]); lab(ax, b, fs=5.8)
ax.set_xticks(x); ax.set_xticklabels(T6); ax.set_ylim(0, 72); ax.set_ylabel("F1 entity (%)"); ax.legend(ncol=2, loc="upper right", fontsize=7); ax.grid(axis="x", visible=False); save(fig, "f_types_to")

# ---- error categories
from evalx import error_breakdown
E = {m: error_breakdown(load("test"), json.load(open("results/" + PF[m]))) for m in M4}
json.dump(E, open("results/errors_to.json", "w"), indent=1)
fig, ax = plt.subplots(figsize=(6.6, 2.7)); N = 1079; left = np.zeros(4)
for nmz, fn, c in [("Correct", lambda e: e["benar"], GRN), ("Wrong boundary or type", lambda e: e["salah_batas"] + e["salah_tipe"] + e["salah_batas_dan_tipe"], AMB), ("Missed", lambda e: e["terlewat"], "#C9CED6")]:
    v = np.array([100 * fn(E[m]) / N for m in M4]); ax.barh(range(4), v, left=left, color=c, label=nmz, height=0.55)
    for i, (l, w_) in enumerate(zip(left, v)): ax.text(l + w_ / 2, i, f"{w_:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if c == GRN else INK)
    left += v
for i, m in enumerate(M4): ax.text(101, i, f"+{E[m]['prediksi_palsu']} spurious", va="center", fontsize=7.5, color=RED)
ax.set_yticks(range(4)); ax.set_yticklabels(M4, fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 120); ax.set_xlabel("Share of the 1,079 gold entities (%)"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.4, 1.2), fontsize=7.5); ax.grid(axis="y", visible=False); save(fig, "f_errors_to")

# ---- analysis panels
fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.7), sharey=True)
for ax, (ttl, items) in zip(axs, [("Seen in train?", [("seen", "seen"), ("unseen", "unseen")]), ("Capitalization", [("kapital", "all\ncapitalized"), ("ada huruf kecil", "some\nlowercase")]), ("Entity length", [("1 token", "1 token"), (">=2 token", "2+ tokens")])]):
    x = np.arange(2); wd = 0.2
    for j, m in enumerate(M4): b = ax.bar(x + (j - 1.5) * wd, [AN[m][k]["recall"] for k, _ in items], wd, color=COL[m], label=m); lab(ax, b, "{:.0f}", 5.8)
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n({AN[M4[0]][k]['n']})" for k, l in items], fontsize=7); ax.set_title(ttl, fontsize=8.5, color=NAVY, loc="left"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Recall of gold entities (%)"); axs[2].legend(fontsize=6.2, loc="upper right"); fig.tight_layout(); save(fig, "f_analysis_to")

# ---- confusion matrices (improved Bi-LSTM CRF, CRF, SVM)
te_ = load("test"); cl = ["person", "location", "corporation", "product", "creative-\nwork", "group", "wrong\nboundary", "missed"]
fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.15), sharey=True)
for ax, m in zip(axs, ["CRF", "SVM", "Bi-LSTM CRF (ensemble)"]):
    M, rows_, cols_ = type_confusion(te_, json.load(open("results/" + PF[m]))); ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1)); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6.2, color="white" if M[i, j] > 0.55 * M.max() else INK)
    ax.axhline(5.5, color=MUT, lw=0.8); ax.set_xticks(range(8)); ax.set_xticklabels(cl, rotation=60, ha="right", fontsize=6.2); ax.set_title(m.replace(" (", "\n("), fontsize=8, color=NAVY)
axs[0].set_yticks(range(7)); axs[0].set_yticklabels(T6 + ["SPURIOUS"], fontsize=7); axs[0].set_ylabel("Gold type", fontsize=8); fig.tight_layout(); save(fig, "f_confusion_to")

# ---- learning curves (dev F1 per epoch): baseline, hidden 256 (dev choice), combination
fig, ax = plt.subplots(figsize=(6.6, 2.5))
for k, c, nm in (("V0", LNAVY, "V0 baseline"), ("V3", AMB, "V3 hidden 256 (dev choice)"), ("V5", NAVY, "V5 all four combined")):
    v = next(v for v in V if v["name"].startswith(k)); ax.plot(range(1, len(v["curve"]) + 1), v["curve"], color=c, lw=1.6, marker="o", ms=2.2, label=nm)
ax.set_xlabel("Epoch"); ax.set_ylabel("Dev F1 entity (%)"); ax.set_ylim(0, 62); ax.legend(loc="lower right", fontsize=7.5); save(fig, "f_curves_to")

# ---- decoding shift: dev (solid) and test (dashed) F1 against delta
fig, ax = plt.subplots(figsize=(6.6, 2.7))
for nm, rows_, c in (("baseline", sw[base["variant"]]["rows"], LNAVY), ("V3 (dev choice)", sw[fin["variant"]]["rows"], AMB), ("ensemble", IBE["rows"], NAVY)):
    ax.plot([r["delta"] for r in rows_], [r["dev"]["F1"] for r in rows_], color=c, ls="-", marker="o", ms=3, label=f"{nm}, dev"); ax.plot([r["delta"] for r in rows_], [r["test"]["F1"] for r in rows_], color=c, ls="--", marker="s", ms=3, label=f"{nm}, test")
ax.set_xlabel("delta subtracted from the O score at decoding"); ax.set_ylabel("F1 entity (%)"); ax.legend(ncol=3, fontsize=6.6, loc="lower center"); ax.set_ylim(25, 62); save(fig, "f_sweep_to")

# ---- bootstrap intervals (sentence resampling, 2000 draws)
def strict_spans(labs):
    out, i, n = set(), 0, len(labs)
    while i < n:
        if labs[i].startswith("B-"):
            t = labs[i][2:]; j = i + 1
            while j < n and labs[j] == "I-" + t: j += 1
            out.add((i, j - 1, t)); i = j
        else: i += 1
    return out
gold = [x["labels"] for x in te_]; st = {}
for m in M4:
    pr = json.load(open("results/" + PF[m])); st[m] = tuple(np.array(v) for v in ([len(strict_spans(g) & strict_spans(p)) for g, p in zip(gold, pr)], [len(strict_spans(p)) for p in pr], [len(strict_spans(g)) for g in gold]))
rng = np.random.RandomState(0); idx = rng.randint(0, len(gold), (2000, len(gold))); bf = {m: np.array([2 * s[0][i].sum() / (s[1][i].sum() + s[2][i].sum()) for i in idx]) * 100 for m, s in st.items()}
out = {m: [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)] for m, v in bf.items()}
pr = json.load(open("results/pred_to_bilstm_final_test.json")); st["V3 + shift (dev choice)"] = tuple(np.array(v) for v in ([len(strict_spans(g) & strict_spans(p)) for g, p in zip(gold, pr)], [len(strict_spans(p)) for p in pr], [len(strict_spans(g)) for g in gold]))
bf["V3 + shift (dev choice)"] = np.array([2 * st["V3 + shift (dev choice)"][0][i].sum() / (st["V3 + shift (dev choice)"][1][i].sum() + st["V3 + shift (dev choice)"][2][i].sum()) for i in idx]) * 100
out["V3 + shift (dev choice)"] = [round(float(np.percentile(bf["V3 + shift (dev choice)"], 2.5)), 1), round(float(np.percentile(bf["V3 + shift (dev choice)"], 97.5)), 1)]
for a, b_ in [("Bi-LSTM CRF (ensemble)", "Bi-LSTM CRF (baseline)"), ("V3 + shift (dev choice)", "Bi-LSTM CRF (baseline)"), ("Bi-LSTM CRF (baseline)", "SVM"), ("SVM", "CRF")]:
    d = bf[a] - bf[b_]; out[f"{a} minus {b_}"] = [round(float(d.mean()), 1), round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]
json.dump(out, open("results/bootstrap_to.json", "w"), indent=1); print(out)
