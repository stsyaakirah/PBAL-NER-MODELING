import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
R = lambda p: json.load(open("results/" + p))
S, A, SV, C2, BL, SF = R("summary.json"), R("analysis.json"), R("svm.json"), R("crf2.json"), R("bilstm.json"), R("svm_final.json")
NAVY, AMB, SLATE, INK, MUT, RED, GRN = "#12406A", "#E0A63A", "#8FA3B8", "#1F2937", "#5B6B7C", "#B85042", "#2E7D5B"
COL = {"CRF": SLATE, "SVM": AMB, "Bi-LSTM CRF": NAVY}; M3 = ["CRF", "SVM", "Bi-LSTM CRF"]; key = {"CRF": "crf", "SVM": "svm", "Bi-LSTM CRF": "bilstm"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9CA3AF", "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": "#E5E7EB", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False})
def save(fig, name): fig.savefig(f"report_figs/{name}.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
def lab(ax, bars, fmt="{:.1f}", fs=7.5):
    for b in bars: ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, fmt.format(b.get_height()), ha="center", va="bottom", fontsize=fs, color=INK)

# ---- F1 pipeline (2 rows x 3)
fig, ax = plt.subplots(figsize=(6.9, 2.75)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 45)
boxes = [("1  Data and splits", "WNUT17 official splits\ntrain / dev / test\nCoNLL, IOB2 labels"), ("2  Reading and checks", "Tab-only and blank lines\nsplit sequences\nBIO check: 0 violations"), ("3  Normalization", "NFKC and HTML unescape\nURL, @user, digits, repeats\nlowercase lookup form"),
         ("4  Representations", "CRF and SVM: features\nBi-LSTM CRF: GloVe,\nchar-CNN, capitalization"), ("5  Protocol", "Tune on dev\nretrain on train+dev\ntest once"), ("6  Evaluation", "Entity F1 (strict), surface F1\nerror analysis\nbootstrap intervals")]
w, h, gx = 29, 19, 5.5
pos = [(1 + i * (w + gx), 25) for i in range(3)] + [(1 + i * (w + gx), 1) for i in range(3)]
for (t, d), (x, y) in zip(boxes, pos):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1.2", fc="#F4F5F7", ec="#D9DEE5", lw=0.8))
    ax.text(x + w / 2, y + h - 4.2, t, ha="center", va="center", fontsize=8, fontweight="bold", color=NAVY)
    ax.text(x + w / 2, y + 7.4, d, ha="center", va="center", fontsize=6.4, color=INK, linespacing=1.45)
for i in (0, 1, 3, 4):
    (x, y) = pos[i]; ax.add_patch(FancyArrowPatch((x + w + 0.4, y + h / 2), (x + w + gx - 0.4, y + h / 2), arrowstyle="-|>", mutation_scale=9, color=NAVY, lw=1.2))
x3, y3 = pos[2]; x4, y4 = pos[3]; xm = 1 + 2 * (w + gx) + w / 2
ax.plot([xm, xm], [y3 - 0.3, 22.6], color=NAVY, lw=1.2); ax.plot([xm, x4 + w / 2], [22.6, 22.6], color=NAVY, lw=1.2)
ax.add_patch(FancyArrowPatch((x4 + w / 2, 22.6), (x4 + w / 2, y4 + h + 0.4), arrowstyle="-|>", mutation_scale=9, color=NAVY, lw=1.2))
save(fig, "f_pipeline")

# ---- F2 architecture
fig, ax = plt.subplots(figsize=(6.9, 2.6)); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(-5, 36)
def blk(x, y, w, h, t, d, fc, tc="white", dc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=1", fc=fc, ec=fc, lw=0.8)); ax.text(x + w / 2, y + h * 0.66, t, ha="center", va="center", fontsize=7.2, fontweight="bold", color=tc); ax.text(x + w / 2, y + h * 0.28, d, ha="center", va="center", fontsize=6.3, color=dc or tc)
ins = [("Word", "GloVe Twitter 200d, frozen"), ("Characters", "char-CNN, 50 filters"), ("Capitalization", "embedding, 8 dims")]
for i, (t, d) in enumerate(ins): blk(1, 24 - i * 11, 24, 9, t, d, "#F4F5F7", INK, MUT); ax.add_patch(FancyArrowPatch((25.6, 28.5 - i * 11), (29.6, 20), arrowstyle="-|>", mutation_scale=8, color=NAVY, lw=1))
blk(30, 12, 15, 16, "Concatenate", "258 dims\ndropout 0.5", "#E6EEF6", INK, INK)
blk(50, 12, 15, 16, "Bi-LSTM", "128 units\nper direction", NAVY)
blk(70, 12, 15, 16, "Linear + CRF", "13 labels\nViterbi", NAVY)
blk(89, 12, 10, 16, "Labels", "BIO per\ntoken", "#FAD400", "#0B2A48")
for x1, x2 in [(45.4, 49.6), (65.4, 69.6), (85.4, 88.6)]: ax.add_patch(FancyArrowPatch((x1, 20), (x2, 20), arrowstyle="-|>", mutation_scale=9, color=NAVY, lw=1.2))
ax.text(50, -2.5, "Trained with Adam (lr 0.001), batch 32, gradient clipping 5, early stopping on dev entity F1", ha="center", fontsize=6.8, color=MUT)
save(fig, "f_architecture")

# ---- F3 main results
fig, ax = plt.subplots(figsize=(6.6, 2.7)); x = np.arange(3); wd = 0.25
for j, (k, nm, c) in enumerate([("P", "Precision", "#9DB4CB"), ("R", "Recall", AMB), ("F1", "F1 entity", NAVY)]):
    b = ax.bar(x + (j - 1) * wd, [S[key[m]][k] for m in M3], wd, label=nm, color=c); lab(ax, b)
ax.set_xticks(x); ax.set_xticklabels(M3); ax.set_ylim(0, 68); ax.set_ylabel("Score (%)"); ax.legend(ncol=3, loc="upper left"); ax.grid(axis="x", visible=False)
save(fig, "f_main")

# ---- F4 per type
T6 = ["person", "location", "corporation", "product", "creative-work", "group"]
fig, ax = plt.subplots(figsize=(6.6, 2.8)); x = np.arange(6); wd = 0.26
for j, m in enumerate(M3): b = ax.bar(x + (j - 1) * wd, [S[key[m]]["types"][t] for t in T6], wd, label=m, color=COL[m]); lab(ax, b, fs=6.8)
ax.set_xticks(x); ax.set_xticklabels(T6); ax.set_ylim(0, 68); ax.set_ylabel("F1 entity (%)"); ax.legend(ncol=3, loc="upper right"); ax.grid(axis="x", visible=False)
save(fig, "f_types")

# ---- F5 ablations
fig, axs = plt.subplots(1, 2, figsize=(6.9, 2.7), gridspec_kw={"width_ratios": [1.15, 1]})
ab = C2["ablation"]; nm = ["Word", "+ shape,\naffix, social", "+ context\n±2, bigram", "+ POS", "+ cluster"]
b = axs[0].bar(range(5), [a["F1"] for a in ab], color=NAVY, width=0.62); lab(axs[0], b); axs[0].set_xticks(range(5)); axs[0].set_xticklabels(nm, fontsize=7); axs[0].set_ylabel("Dev F1 entity (%)"); axs[0].set_ylim(0, 31); axs[0].set_title("CRF, cumulative features", fontsize=8.5, color=NAVY, loc="left"); axs[0].grid(axis="x", visible=False)
sa = {a["name"]: a for a in SV["ablation"]}; full = sa["LinearSVC balanced + GloVe (lengkap)"]["F1"]; noW = sa["Tanpa pembobotan kelas"]["F1"]; noG = sa["Tanpa vektor GloVe"]["F1"]; noR = R("svm2.json")["dev"][1]["no_repair"]["F1"]
vals = [("Full model\n(BIO repair)", full, AMB), ("Without\nBIO repair", noR, NAVY), ("Without\nclass weights", noW, SLATE), ("Without\nGloVe", noG, SLATE)]
b = axs[1].bar(range(4), [v[1] for v in vals], color=[v[2] for v in vals], width=0.62); lab(axs[1], b); axs[1].set_xticks(range(4)); axs[1].set_xticklabels([v[0] for v in vals], fontsize=7); axs[1].set_ylim(0, 52); axs[1].set_title("SVM, dev F1 entity (%)", fontsize=8.5, color=NAVY, loc="left"); axs[1].grid(axis="x", visible=False)
fig.tight_layout(); save(fig, "f_ablation")

# ---- F6 learning curves
fig, ax = plt.subplots(figsize=(6.6, 2.6)); cb, ce, cp = S["bilstm"]["curve_bio"], S["bilstm"]["curve_bioes"], S["bilstm"]["curve_bio_prefix"]
ax.plot(range(1, len(cb) + 1), cb, color=NAVY, lw=1.9, marker="o", ms=3, label="BIO (after fix)"); ax.plot(range(1, len(cp) + 1), cp, color=SLATE, lw=1.4, ls="--", label="BIO (before fix)"); ax.plot(range(1, len(ce) + 1), ce, color=AMB, lw=1.4, marker="o", ms=2.5, label="BIOES (before fix)")
ax.axvline(S["bilstm"]["best_ep_bio"], color=NAVY, ls=":", lw=1); ax.text(S["bilstm"]["best_ep_bio"] + 0.3, 54, "best BIO epoch (%d)" % S["bilstm"]["best_ep_bio"], fontsize=7, color=NAVY)
ax.set_xlabel("Epoch"); ax.set_ylabel("Dev F1 entity (%)"); ax.set_ylim(0, 60); ax.legend(loc="lower right", fontsize=7.5); save(fig, "f_curves")

# ---- F7 error breakdown
fig, ax = plt.subplots(figsize=(6.6, 2.4)); E = {m: S[key[m]]["errors"] for m in M3}; N = 1079
parts = [("Correct", lambda e: e["benar"], GRN), ("Wrong boundary or type", lambda e: e["salah_batas"] + e["salah_tipe"] + e["salah_batas_dan_tipe"], AMB), ("Missed", lambda e: e["terlewat"], "#C9CED6")]
left = np.zeros(3)
for nmz, fn, c in parts:
    v = np.array([100 * fn(E[m]) / N for m in M3]); ax.barh(range(3), v, left=left, color=c, label=nmz, height=0.55)
    for i, (l, w_) in enumerate(zip(left, v)): ax.text(l + w_ / 2, i, f"{w_:.0f}%", ha="center", va="center", fontsize=7.5, color="white" if c == GRN else INK)
    left += v
for i, m in enumerate(M3): ax.text(101, i, f"+{E[m]['prediksi_palsu']} spurious", va="center", fontsize=7.5, color=RED)
ax.set_yticks(range(3)); ax.set_yticklabels(M3); ax.invert_yaxis(); ax.set_xlim(0, 118); ax.set_xlabel("Share of the 1,079 gold entities (%)"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.42, 1.22), fontsize=7.5); ax.grid(axis="y", visible=False)
save(fig, "f_errors")

# ---- F8 analysis panels
G = S["analysis"]; fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.6), sharey=True)
panels = [("Seen in train+dev?", [("seen", "seen (109)"), ("unseen", "unseen (970)")]), ("Capitalization", [("kapital", "all capitalized\n(800)"), ("ada huruf kecil", "some lowercase\n(279)")]), ("Entity length", [("1 token", "1 token\n(718)"), (">=2 token", "2+ tokens\n(361)")])]
for ax, (ttl, items) in zip(axs, panels):
    x = np.arange(len(items)); wd = 0.26
    for j, m in enumerate(M3): b = ax.bar(x + (j - 1) * wd, [G[m][k]["recall"] for k, _ in items], wd, color=COL[m], label=m); lab(ax, b, "{:.0f}", 6.8)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in items], fontsize=7); ax.set_title(ttl, fontsize=8.5, color=NAVY, loc="left"); ax.set_ylim(0, 100); ax.grid(axis="x", visible=False)
axs[0].set_ylabel("Recall of gold entities (%)"); axs[2].legend(fontsize=7, loc="upper right"); fig.tight_layout(); save(fig, "f_analysis")

# ---- F9 confusion (gold type x predicted, with SPURIOUS row)
from evalx import type_confusion
from wnut_data import load
te_ = load("test"); PFC = {"CRF": "pred_fix_crf_test.json", "SVM": "pred_fix_svm_test.json", "Bi-LSTM CRF": "pred_fix_bilstm_test.json"}
cl = ["person", "location", "corporation", "product", "creative-\nwork", "group", "wrong\nboundary", "missed"]
fig, axs = plt.subplots(1, 3, figsize=(6.9, 3.15), sharey=True)
for ax, m in zip(axs, M3):
    M, rows_, cols_ = type_confusion(te_, json.load(open("results/" + PFC[m]))); ax.imshow(M, cmap="Blues", vmin=0, vmax=max(M.max(), 1)); ax.grid(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=6.2, color="white" if M[i, j] > 0.55 * M.max() else INK)
    ax.axhline(5.5, color=MUT, lw=0.8); ax.set_xticks(range(8)); ax.set_xticklabels(cl, rotation=60, ha="right", fontsize=6.2); ax.set_title(m, fontsize=8.5, color=NAVY)
axs[0].set_yticks(range(7)); axs[0].set_yticklabels(T6 + ["SPURIOUS"], fontsize=7); axs[0].set_ylabel("Gold type", fontsize=8)
fig.tight_layout(); save(fig, "f_confusion")

# ---- bootstrap CIs
from seqeval.metrics.sequence_labeling import get_entities
from wnut_data import load
def strict_spans(labs):
    out, i, n = set(), 0, len(labs)
    while i < n:
        if labs[i].startswith("B-"):
            t = labs[i][2:]; j = i + 1
            while j < n and labs[j] == "I-" + t: j += 1
            out.add((i, j - 1, t)); i = j
        else: i += 1
    return out
gold = [x["labels"] for x in load("test")]; PF = {"CRF": "pred_fix_crf_test.json", "SVM": "pred_fix_svm_test.json", "Bi-LSTM CRF": "pred_fix_bilstm_test.json"}
st = {}
for m, f in PF.items():
    pr = json.load(open("results/" + f)); st[m] = tuple(np.array(v) for v in ([len(strict_spans(g) & strict_spans(p)) for g, p in zip(gold, pr)], [len(strict_spans(p)) for p in pr], [len(strict_spans(g)) for g in gold]))
rng = np.random.RandomState(0); idx = rng.randint(0, len(gold), (2000, len(gold)))
bf = {m: np.array([2 * s[0][i].sum() / (s[1][i].sum() + s[2][i].sum()) for i in idx]) * 100 for m, s in st.items()}
out = {m: [round(float(np.percentile(v, 2.5)), 1), round(float(np.percentile(v, 97.5)), 1)] for m, v in bf.items()}
for a, b_ in [("SVM", "CRF"), ("Bi-LSTM CRF", "SVM")]:
    d = bf[a] - bf[b_]; out[f"{a} minus {b_}"] = [round(float(d.mean()), 1), round(float(np.percentile(d, 2.5)), 1), round(float(np.percentile(d, 97.5)), 1)]
json.dump(out, open("results/bootstrap.json", "w"), indent=1); print(out)
