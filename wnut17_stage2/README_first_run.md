# NER on WNUT17: CRF, SVM and Bi-LSTM CRF (Assignment #2, Stage 2, Kelompok 1)

## Folder layout of the archive (flat: run everything from this folder)
- (root)      Python modules, run scripts and the notebook `Stage2_NER_WNUT17_Kelompok1.ipynb`
- `data/`      the three official WNUT17 files used (CC BY 4.0): `wnut17train.conll`, `emerging.dev.conll`, `emerging.test.annotated`
- `results/`   metrics and test predictions of the final runs (after the preprocessing fix), search logs, cross-validation
- `results_prefix/` the same files from the first full run, before the preprocessing fix (kept for the before/after comparison)
- `report_figs/` figures used in the report

## Quick start (about one minute, no GloVe file needed)
Open `Stage2_NER_WNUT17_Kelompok1.ipynb` and run all cells. It reloads the saved predictions and recomputes every table and figure.
Needs: numpy, pandas, matplotlib, scikit-learn, seqeval, nltk (jax and optax only to print the Bi-LSTM CRF source).

## Retraining
1. Download `glove-twitter-200.gz` (gensim-data release `glove-twitter-200`), then `GLOVE_GZ=/path/glove-twitter-200.gz python prep_glove.py`.
2. `python fix_crf.py`, `python fix_svm.py`, `python fix_bilstm.py` (final settings, seed 42; Bi-LSTM CRF also seeds 43 and 44).
3. Hyperparameter search and ablations: `run_crf2.py`, `run_svm.py`, `run_svm2.py`; cross-validation: `run_cv_final.py`.
4. `python analysis.py`, `python consolidate_fix.py`, `python make_figures.py`, `python build_report2.py`.
Set `WNUT_DIR` to the folder that holds the three data files if it is not `./data`.

## Modules
- `wnut_data.py` reader (blank line = document end, tab-only line = sentence end), surface F1
- `prep_glove.py` normalization and GloVe lookup keys (contains the fixed digit, domain and hashtag rules)
- `features.py` CRF/SVM feature dictionaries, POS tags, cluster IDs
- `evalx.py` strict entity metrics (seqeval), error categories, confusion matrix with a SPURIOUS row
- `bilstm_crf.py` Bi-LSTM CRF in JAX (LSTM, CRF loss, Viterbi; checked against brute force)

## Notes
- The test file is `emerging.test.annotated`. The repository file `emerging.test.conll` holds raw multi-annotator labels and is not used.
- Hyperparameters are chosen on dev only. Final models train on train + dev and are scored once on the full test file.
- Grids, ablations and cross-validation were run before the preprocessing fix; final models were retrained after it with the same hyperparameters.

- Reported Bi-LSTM CRF result: seed 42 (fixed in Stage 1); seeds 43 and 44 are disclosed. `PRIMARY=mean python consolidate_fix.py` switches the summary to the mean of the three seeds (rebuild the figures and the report afterwards).
- `domain_mixing_crf.py` and `target_domain_adaptation_crf.py` are analyses of training-data mixes (report Appendix G); they are not part of the official protocol.
