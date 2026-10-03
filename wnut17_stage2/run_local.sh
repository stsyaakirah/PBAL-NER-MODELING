#!/bin/zsh
# Opsi 1 compute stages, run in a LOCAL (non-iCloud) dir. glove_sub.npz already present.
set -e
cd "/private/tmp/claude-501/-Users-adityaiko-Desktop-MCs-in-AI-Class-Semester-2-NLP-Thursday-Assignment-2/96e3b24d-4f38-478e-8e27-c3fb9ee5fa32/scratchpad/wnut_run"
PY=/opt/anaconda3/bin/python
log(){ echo "[$(date +%H:%M:%S)] $*"; }

log "improve_bilstm.py (train V0-V5)"; $PY improve_bilstm.py; log "improve done"
log "ensemble_bilstm.py (6-model)"; $PY ensemble_bilstm.py
cp results/ensemble_bilstm.json results/ensemble_6model.json; log "saved ensemble_6model.json"
log "train_extra.py (V6-V8)"; $PY train_extra.py; log "extra done"
log "ensemble_bilstm.py (9-model enlarged)"; $PY ensemble_bilstm.py
cp results/ensemble_bilstm.json results/ensemble_9model.json; log "saved ensemble_9model.json"
log "ALL DONE"
