#!/bin/zsh
# Autonomous Opsi 1 chain (robust single-connection resume version).
# Resumes glove-twitter-200.gz through connection drops, verifies, then
# prep -> improve(V0-V5) -> 6-model ensemble -> train extra(V6-V8) -> 9-model ensemble.
set -e
cd "/Users/adityaiko/Desktop/MCs in AI/Class/Semester 2/NLP Thursday/Assignment 2/wnut17_stage2"
PY=/opt/anaconda3/bin/python
URL="https://github.com/RaRe-Technologies/gensim-data/releases/download/glove-twitter-200/glove-twitter-200.gz"
LEN=795373100
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# 1) resumable single-connection download; loop until the file reaches full size
log "download/resume glove-twitter-200.gz (have $(stat -f%z glove-twitter-200.gz 2>/dev/null || echo 0) / $LEN)"
tries=0
while true; do
  sz=$(stat -f%z glove-twitter-200.gz 2>/dev/null || echo 0)
  [ "$sz" = "$LEN" ] && { log "download complete: $sz bytes"; break; }
  tries=$((tries+1)); [ $tries -gt 200 ] && { log "too many retries, abort"; exit 9; }
  log "resume attempt $tries (at $((sz/1024/1024)) MB)"
  curl -L -C - --retry 100 --retry-delay 3 --retry-all-errors --retry-connrefused \
       --connect-timeout 30 -o glove-twitter-200.gz "$URL" || true
  sleep 2
done

# 2) verify
sz=$(stat -f%z glove-twitter-200.gz); [ "$sz" = "$LEN" ] || { log "SIZE MISMATCH $sz"; exit 2; }
log "gzip integrity check"; gzip -t glove-twitter-200.gz && log "gzip OK" || { log "gzip CORRUPT"; exit 3; }

# 3) prep_glove -> glove_sub.npz
log "prep_glove.py"
GLOVE_GZ="$PWD/glove-twitter-200.gz" $PY prep_glove.py
[ -f glove_sub.npz ] && log "glove_sub.npz created ($(stat -f%z glove_sub.npz) bytes)" || { log "no glove_sub.npz"; exit 4; }

# 4) improve_bilstm -> V0..V5 (reproduce baseline / ensemble inputs)
log "improve_bilstm.py (train V0-V5, ~15-20 min)"; $PY improve_bilstm.py; log "improve done"

# 5) 6-model ensemble (reproduction check ~41.5)
log "ensemble_bilstm.py (6-model)"; $PY ensemble_bilstm.py
cp results/ensemble_bilstm.json results/ensemble_6model.json; log "saved ensemble_6model.json"

# 6) extra structural variants V6,V7,V8
log "train_extra.py (V6-V8, ~10 min)"; $PY train_extra.py; log "extra done"

# 7) 9-model ensemble
log "ensemble_bilstm.py (9-model enlarged)"; $PY ensemble_bilstm.py
cp results/ensemble_bilstm.json results/ensemble_9model.json; log "saved ensemble_9model.json"
log "ALL DONE"
