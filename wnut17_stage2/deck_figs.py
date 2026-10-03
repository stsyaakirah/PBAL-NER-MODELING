"""Regenerate the key RESULT visualizations in the deck color palette, from the real
experiment outputs. Writes PNGs to the folder given by OUT env (the deck scratchpad)."""
import os, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from wnut_data import load
from evalx import evaluate, error_breakdown, type_confusion

OUT = os.environ.get("OUT", "deck_figs"); os.makedirs(OUT, exist_ok=True)
NAVY="#2A356E"; VIOLET="#7A5AF8"; TEAL="#0C9E9A"; CORAL="#EF6A4E"; AMBER="#D98A12"
GREEN="#1F9D6B"; SECB="#4F5FC4"; INK="#1A2036"; INK2="#5B6480"; LINE="#D8DEEE"
CRF,SVM,BIL,ROB=AMBER,CORAL,TEAL,VIOLET
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":14,"text.color":INK,
  "axes.labelcolor":INK,"xtick.color":INK,"ytick.color":INK,"axes.edgecolor":LINE,
  "axes.titlecolor":INK,"figure.dpi":150,"savefig.bbox":"tight"})
def style(ax):
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.grid(axis="y",color=LINE,lw=1); ax.set_axisbelow(True)

test=load("test")
base=json.load(open("results/pred_to_bilstm_base_nobias_test.json"))
rob=json.load(open("results/pred_to_bilstm_roberta_test.json"))
TYPES=["person","location","corporation","product","creative-work","group"]
TY_ID=["person","location","corp.","product","creative","group"]

# 1 per-type F1: baseline vs RoBERTa
evb=evaluate(test,base); evr=evaluate(test,rob)
bt=[evb["per_type"].get(t,{}).get("f1-score",0)*100 for t in TYPES]
rt=[evr["per_type"].get(t,{}).get("f1-score",0)*100 for t in TYPES]
fig,ax=plt.subplots(figsize=(8.2,4.3)); x=np.arange(6); w=0.38
ax.bar(x-w/2,bt,w,color=BIL,label="Bi-LSTM baseline",edgecolor="white")
ax.bar(x+w/2,rt,w,color=ROB,label="+ RoBERTa",edgecolor="white")
for xi,v in zip(x-w/2,bt): ax.text(xi,v+1,f"{v:.0f}",ha="center",fontsize=11.5,color=INK2)
for xi,v in zip(x+w/2,rt): ax.text(xi,v+1,f"{v:.0f}",ha="center",fontsize=11.5,color=ROB)
ax.set_xticks(x); ax.set_xticklabels(TY_ID); ax.set_ylabel("F1 (%)"); ax.set_ylim(0,80)
ax.legend(frameon=False,fontsize=13,loc="upper right"); style(ax)
fig.savefig(f"{OUT}/fig_pertype.png"); plt.close(fig)

# 2 confusion matrix RoBERTa
M,rows,cols=type_confusion(test,rob)
cmap=LinearSegmentedColormap.from_list("v",["#FFFFFF","#C9CBF0","#7A5AF8","#2A356E"])
fig,ax=plt.subplots(figsize=(7.6,5.4))
im=ax.imshow(M,cmap=cmap,aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace("creative-work","creative").replace("wrong boundary","batas salah").replace("missed","terlewat") for c in cols],rotation=35,ha="right",fontsize=12)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r.replace("creative-work","creative") for r in rows],fontsize=12)
mx=M.max() or 1
for i in range(len(rows)):
  for j in range(len(cols)):
    if M[i,j]: ax.text(j,i,int(M[i,j]),ha="center",va="center",fontsize=11,
      color="white" if M[i,j]>0.45*mx else INK)
ax.set_xlabel("diprediksi"); ax.set_ylabel("gold (baris SPURIOUS = prediksi palsu)")
fig.savefig(f"{OUT}/fig_confusion.png"); plt.close(fig)

# 3 error breakdown baseline vs RoBERTa
cats=["benar","salah_batas","salah_tipe","salah_batas_dan_tipe","terlewat","prediksi_palsu"]
labs=["benar","batas salah","tipe salah","batas+tipe","terlewat","palsu"]
eb=error_breakdown(test,base); er=error_breakdown(test,rob)
bv=[eb.get(c,0) for c in cats]; rv=[er.get(c,0) for c in cats]
fig,ax=plt.subplots(figsize=(8.4,4.3)); x=np.arange(len(cats)); w=0.38
ax.bar(x-w/2,bv,w,color=BIL,label="Bi-LSTM baseline",edgecolor="white")
ax.bar(x+w/2,rv,w,color=ROB,label="+ RoBERTa",edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labs,rotation=18,ha="right"); ax.set_ylabel("jumlah entitas")
ax.legend(frameon=False,fontsize=13); style(ax)
fig.savefig(f"{OUT}/fig_errors.png"); plt.close(fig)

# 4 training curves: baseline vs RoBERTa (dev F1 per epoch)
bc=json.load(open("results/improve_bilstm.json"))["variants"][0]["curve"]
rc=json.load(open("results/bilstm_roberta.json"))["dev_curve"]
fig,ax=plt.subplots(figsize=(8.2,4.3))
ax.plot(range(1,len(bc)+1),bc,"-o",color=BIL,lw=2.5,ms=5,label="Bi-LSTM baseline")
ax.plot(range(1,len(rc)+1),rc,"-o",color=ROB,lw=2.5,ms=5,label="+ RoBERTa")
ax.set_xlabel("epoch"); ax.set_ylabel("F1 dev (%)"); ax.legend(frameon=False,fontsize=13,loc="lower right")
style(ax); ax.grid(axis="both",color=LINE,lw=1)
fig.savefig(f"{OUT}/fig_curve.png"); plt.close(fig)

# 6 Section A main results
fig,ax=plt.subplots(figsize=(8.2,4.3))
names=["CRF","SVM","Bi-LSTM\nbaseline","Bi-LSTM\nensemble","+ RoBERTa"]
vals=[16.8,29.8,40.4,41.4,54.1]; cols2=[CRF,SVM,BIL,BIL,ROB]
b=ax.bar(names,vals,color=cols2,edgecolor="white")
for bi,v in zip(b,vals): ax.text(bi.get_x()+bi.get_width()/2,v+0.8,f"{v:.1f}".replace(".",","),ha="center",fontsize=12.5,color=INK)
ax.axhline(41.86,color=INK2,ls="--",lw=1.2); ax.text(4.4,42.6,"shared task 41,86",ha="right",fontsize=11.5,color=INK2)
ax.set_ylabel("F1 entity test (%)"); ax.set_ylim(0,62); style(ax)
fig.savefig(f"{OUT}/fig_main.png"); plt.close(fig)

# 7 Section B leaked vs unseen recall (3 models)
B=json.load(open("results_partB/partB_analysis.json"))
mods=["CRF","SVM","Bi-LSTM CRF"]
lk=[B[m]["recall_leaked_vs_unseen"].get("leaked (seen in new train)",{}).get("recall",0) for m in mods]
un=[B[m]["recall_leaked_vs_unseen"].get("genuinely unseen",{}).get("recall",0) for m in mods]
fig,ax=plt.subplots(figsize=(8.0,4.3)); x=np.arange(3); w=0.38
ax.bar(x-w/2,lk,w,color=SECB,label="entitas bocor",edgecolor="white")
ax.bar(x+w/2,un,w,color=INK2,label="benar-benar baru",edgecolor="white")
for xi,v in zip(x-w/2,lk): ax.text(xi,v+1.5,f"{v:.0f}",ha="center",fontsize=12,color=INK)
for xi,v in zip(x+w/2,un): ax.text(xi,v+1.5,f"{v:.0f}",ha="center",fontsize=12,color=INK)
ax.set_xticks(x); ax.set_xticklabels(mods); ax.set_ylabel("recall (%)"); ax.set_ylim(0,90)
ax.legend(frameon=False,fontsize=13); style(ax)
fig.savefig(f"{OUT}/fig_novelty.png"); plt.close(fig)

# 8 Section A vs B per model
fig,ax=plt.subplots(figsize=(8.0,4.3)); x=np.arange(3); w=0.38
aF=[16.8,29.8,40.4]; bF=[40.6,42.1,51.8]
ax.bar(x-w/2,aF,w,color=BIL,label="A (split resmi)",edgecolor="white")
ax.bar(x+w/2,bF,w,color=SECB,label="B (split gabungan)",edgecolor="white")
for xi,v in zip(x-w/2,aF): ax.text(xi,v+0.8,f"{v:.1f}".replace(".",","),ha="center",fontsize=12,color=INK)
for xi,v in zip(x+w/2,bF): ax.text(xi,v+0.8,f"{v:.1f}".replace(".",","),ha="center",fontsize=12,color=SECB)
ax.set_xticks(x); ax.set_xticklabels(mods); ax.set_ylabel("F1 entity (%)"); ax.set_ylim(0,62)
ax.legend(frameon=False,fontsize=13); style(ax)
fig.savefig(f"{OUT}/fig_AvsB.png"); plt.close(fig)

# 9 leakage bar
st=json.load(open("data_partB/split_stats.json"))
fig,ax=plt.subplots(figsize=(7.4,4.0))
labs2=["Test resmi\n(vs train)","Dev gabungan","Test gabungan"]
vals2=[2.0,st["leakage"]["dev"]["leak_pct"],st["leakage"]["test"]["leak_pct"]]
b=ax.bar(labs2,vals2,color=[BIL,SECB,SECB],edgecolor="white")
for bi,v in zip(b,vals2): ax.text(bi.get_x()+bi.get_width()/2,v+0.5,f"{v:.1f}%".replace(".",","),ha="center",fontsize=12.5,color=INK)
ax.set_ylabel("entitas bocor ke train (%)"); ax.set_ylim(0,26); style(ax)
fig.savefig(f"{OUT}/fig_leakage.png"); plt.close(fig)

print("figures written to",OUT,":",sorted(os.listdir(OUT)))
