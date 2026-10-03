# Additions: Section B (pooled split) and the RoBERTa enhancement

Two pieces of work added on top of the Section A (official-split) pipeline. Section A is
untouched: its results in `results/` (`to_crf.json`, `to_svm.json`, `improve_bilstm.json`,
etc.) and its report PDFs are exactly as before.

Environment note: the Bi-LSTM code is JAX; `pip install jax jaxlib optax` forced NumPy 2,
which required upgrading scikit-learn and matplotlib. Always run with `python -W ignore`
(unrelated packages print harmless NumPy-2 warnings). Heavy training was run in a local
(non-iCloud) copy of this folder to avoid iCloud I/O stalls, then results were copied back.

## 1. Opsi 1: improving the Bi-LSTM CRF (Twitter-only, seed 42)

Goal was F1 entity/surface above the plateau. Two levers tried.

### a) Bigger and multi-seed ensembles: did NOT help (variance already reduced)
- `train_extra.py`     trains 3 extra structural variants V6/V7/V8 (wider char-CNN, char-dim 50, no-cap).
- `ensemble_multiseed.py`  trains baselines with seeds 43-46 and ensembles seeds 42-46.
- Results (test, delta 0): 6-model 41.35 / 9-model 41.11 / 5-seed 41.14 entity F1; surface F1 is
  identical (39.63) in all three. Ensembling has plateaued. Files:
  `results/ensemble_{6model,9model,multiseed}.json`, `results/ckpt_V6..V8`, `results/ckpt_seed43..46`,
  `results/improve_bilstm_opsi1.json`.
- Reproduce: `run_local.sh` (needs `glove_sub.npz`, regenerated and included).

### b) Frozen RoBERTa features: LARGE gain on both metrics (this is the improvement)
- `extract_roberta.py`  runs roberta-base over each sentence, mean-pools subwords to CoNLL tokens,
  caches `results/roberta_{train,dev,test}.pkl` (768d, frozen, not fine-tuned).
- `opsi1_roberta.py`    same Bi-LSTM CRF baseline + the RoBERTa vector as an extra input.
- Results (test): delta 0 -> entity 52.40, surface 51.65; dev-tuned delta 1.0 -> entity 54.12,
  surface 52.94; bootstrap entity CI [51.0, 57.2]. About +13 points on both metrics vs the
  ensemble. Files: `results/bilstm_roberta.json`, `results/pred_to_bilstm_roberta{,_nobias}_test.json`.
- Honest framing: RoBERTa is a 2019 contextual model the 2017 shared task did not have, so this
  is not an equal-footing win over 41.86. It is a labeled enhancement to the Bi-LSTM CRF (frozen
  feature extractor), not a 4th model. Still Twitter-only, seed 42, delta chosen on dev only.
- Needs `transformers` (installed) and roberta-base (cached in `~/.cache/huggingface`).
- Reproduce: `python -W ignore extract_roberta.py && python -W ignore opsi1_roberta.py`.

## 2. Section B: pooled 80/10/10 random split (ablation, mixing authorized for this section only)

Not comparable to the shared task. Shows how much of WNUT17's difficulty is entity novelty.

- `partB_split.py`     pools all three official files, splits by document (seed 42), writes
  `data_partB/{train,dev,test}.conll` + `data_partB/split_stats.json`. Conservation exact
  (5690 seq / 101857 tok / 3890 ent), BIO violations 0.
- `partB_models.py`    CRF + SVM, same finalized hyperparameters (not re-searched).
- `partB_bilstm.py`    Bi-LSTM CRF baseline only.
- `partB_analysis.py`  bootstrap CI, per-type, confusion (SPURIOUS row), error breakdown, and the
  leaked-vs-unseen recall decomposition.
- `partB_figures.py` / `partB_report.py`  black-only figures and the addendum PDF
  `NER_WNUT17_PartB_MixedSplit.pdf`.
- Results (test entity F1): CRF 16.8 -> 40.6, SVM 29.8 -> 42.1, Bi-LSTM 40.4 -> 51.8. Entity
  leakage into train rises from ~2% (official) to 21.1% (pooled); the gain concentrates on leaked
  entities, so it is leakage plus feature coverage, not better modeling. Files: `results_partB/`.
- Reproduce: `python -W ignore partB_split.py && python -W ignore partB_models.py && python -W ignore partB_bilstm.py && python -W ignore partB_analysis.py && python -W ignore partB_figures.py && python -W ignore partB_report.py`.
