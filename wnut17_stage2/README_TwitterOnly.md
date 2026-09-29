# NER on WNUT17, Twitter-only training, seed 42 (Assignment #2, Stage 2, Kelompok 1)

This archive belongs to the second Stage 2 run: the models are trained on the **Twitter train file only** (no YouTube or Reddit data in training), every run uses **seed 42**, and several ways of raising the Bi-LSTM CRF F1 are tried. The first run (train + dev, three seeds) is a separate deliverable; its files are kept in `results/` and `results_prefix/`.

## Layout (flat: run everything from this folder)
- `Stage2_NER_WNUT17_Kelompok1_TwitterOnly.ipynb` executed notebook (reloads saved results in about a minute; `RUN_TRAINING = True` retrains)
- `NER_WNUT17_Comprehensive_Report_TwitterOnly_Seed42.pdf` the report
- `data/` the three official WNUT17 files (CC BY 4.0); `report_figs_to/` figures; `results/` metrics and predictions

## Files of this run
- `trainonly_models.py` CRF and SVM trained on train only (predictions in `results/pred_to_*.json`, metrics in `results/to_crf.json`, `results/to_svm.json`)
- `improve_bilstm.py` Bi-LSTM CRF baseline and variants V1 to V5 (word dropout, fine-tuned GloVe, hidden 256, POS input, combination) and the decoding-shift sweep; choices are made on dev only (`results/improve_bilstm.json`)
- `ensemble_bilstm.py` ensemble of the six seed-42 variants (needs the checkpoints `results/ckpt_V*.pkl`, which `improve_bilstm.py` writes; they are not included to keep the archive small)
- `analysis_to.py`, `make_figures_to.py`, `build_report3.py`, `build_notebook_to.py` analysis, figures, report and notebook
- `bilstm_crf.py` now has the options `word_dropout`, `finetune`, `pos_dim`/POS input and an O-bias `delta` at decoding (all off by default = the baseline)

## Quick start
Open the notebook and run all cells. Needs numpy, pandas, matplotlib, scikit-learn, seqeval, nltk (jax and optax only to print the model source).

## Retraining (about one hour on one CPU core)
1. `GLOVE_GZ=/path/glove-twitter-200.gz python prep_glove.py` (gensim-data release `glove-twitter-200`)
2. `python trainonly_models.py`, `python improve_bilstm.py`, `python ensemble_bilstm.py`
3. `python analysis_to.py`, `python make_figures_to.py`, `python build_report3.py`

## Protocol notes
- Dev (YouTube) is used only to choose settings and stop training; it never updates a weight. The test file (`emerging.test.annotated`) is scored on all 1,287 sequences; the repository file `emerging.test.conll` is not used.
- Test scores of every variant are printed for transparency; nothing was chosen on test.
- Search grids, ablations and cross-validation of the CRF and SVM were run before the normalization fix (report Appendix B); the Bi-LSTM CRF runs use the fixed preprocessing.
