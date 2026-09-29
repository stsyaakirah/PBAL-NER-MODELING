"""Builds results/summary.json from the post-fix runs (fix_*.json). The pre-fix summary stays in results_prefix/.
PRIMARY=seed42 (default): the Bi-LSTM CRF result reported is the seed fixed in Stage 1 (42); seeds 43 and 44 are still disclosed.
PRIMARY=mean: the mean of the three seeds is reported instead."""
import json, os, numpy as np
PRIMARY = os.environ.get("PRIMARY", "seed42")
R = lambda p: json.load(open("results/" + p)); PRE = json.load(open("results_prefix/summary.json"))
crf, svm, bl, cv = R("fix_crf.json"), R("fix_svm.json"), R("fix_bilstm.json"), R("cv_final.json")
T = ["person", "location", "corporation", "product", "creative-work", "group"]; pct = lambda x: round(100 * x, 2)
def row(ev): return {"P": pct(ev["entity"]["precision"]), "R": pct(ev["entity"]["recall"]), "F1": pct(ev["entity"]["f1"]), "surf": pct(ev["surface"]["f1"]), "tokacc": pct(ev["token_acc"]), "tokF1": pct(ev["token_f1_nonO"]["f1"]),
                      "types": {t: round(100 * ev["per_type"][t]["f1-score"], 1) for t in T}, "typeP": {t: round(100 * ev["per_type"][t]["precision"], 1) for t in T}, "typeR": {t: round(100 * ev["per_type"][t]["recall"], 1) for t in T}}
runs = [x["test"] for x in bl["B"]["runs"]]; mean_ = lambda k: round(float(np.mean([pct(e["entity"][k]) for e in runs])), 2)
mean3 = {"P": mean_("precision"), "R": mean_("recall"), "F1": mean_("f1"), "surf": round(float(np.mean([pct(e["surface"]["f1"]) for e in runs])), 2), "tokacc": round(float(np.mean([pct(e["token_acc"]) for e in runs])), 2),
         "types": {t: round(float(np.mean([100 * e["per_type"][t]["f1-score"] for e in runs])), 1) for t in T}}
seed42 = row(runs[0])
pre_runs = json.load(open("results_prefix/bilstm.json"))["B_bio_final"]["runs"]; pre_seed42 = {k: row(pre_runs[0]["test"])[k] for k in ("P", "R", "F1")}
pre_mean = {k: PRE["bilstm"][k] for k in ("P", "R", "F1")}
prim = seed42 if PRIMARY == "seed42" else mean3
bilstm = {**prim, "primary": "seed 42" if PRIMARY == "seed42" else "mean of 3 seeds", "mean3": mean3, "seed42": seed42, "pre": pre_seed42 if PRIMARY == "seed42" else pre_mean,
          "std": round(float(np.std([pct(e["entity"]["f1"]) for e in runs])), 2), "seeds": [pct(e["entity"]["f1"]) for e in runs], "epochs": bl["B"]["epochs"], "errors": bl["B"]["runs"][0]["errors"],
          "trainonly_bio": pct(bl["A"]["test"]["entity"]["f1"]), "dev_bio": bl["A"]["dev_F1"], "best_ep_bio": bl["A"]["best_epoch"], "bioes_test": PRE["bilstm"]["bioes_test"], "bioes_dev": PRE["bilstm"]["bioes_dev"], "bioes_ep": PRE["bilstm"]["bioes_ep"],
          "curve_bio": [h["dev_f1"] for h in bl["A"]["history"]], "curve_bioes": PRE["bilstm"]["curve_bioes"], "curve_bio_prefix": PRE["bilstm"]["curve_bio"]}
S = {"crf": {**row(crf["test"]), "dev": pct(crf["dev_trainonly"]["entity"]["f1"]), "cfg": PRE["crf"]["cfg"], "errors": crf["test_errors"], "cv": {"folds": cv["crf"], "mean": cv["crf_mean"], "std": cv["crf_std"]}, "trainonly": pct(crf["test_trainonly"]["entity"]["f1"])},
     "svm": {**row(svm["test"]), "dev": pct(svm["dev_trainonly"]["no_repair"]["entity"]["f1"]), "dev_repair": pct(svm["dev_trainonly"]["repair"]["entity"]["f1"]), "errors": svm["test_errors"], "cv": {"folds": cv["svm"], "mean": cv["svm_mean"], "std": cv["svm_std"]},
             "repair_test": row(svm["test_repair"]), "invalid_I": svm["invalid_I"], "trainonly": pct(svm["test_trainonly"]["no_repair"]["entity"]["f1"])},
     "bilstm": bilstm, "crf_grid": PRE["crf_grid"], "crf_ablation": PRE["crf_ablation"], "crf_first_grid_best": PRE["crf_first_grid_best"], "crf_nocluster_test": PRE["crf_nocluster_test"]}
S["analysis"] = R("analysis.json")["group_recall"]; json.dump(S, open("results/summary.json", "w"), indent=1, default=float)
print("PRIMARY =", PRIMARY, "| Bi-LSTM CRF reported:", {k: bilstm[k] for k in ("P", "R", "F1", "surf")}, "| mean of 3:", mean3["F1"], "| seeds", bilstm["seeds"])
