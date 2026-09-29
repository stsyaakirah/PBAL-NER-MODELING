#!/bin/sh
cd /home/claude/proj
rm -f cache_clusters.pkl
python3 prep_glove.py > results/fix_glove.log 2>&1
python3 fix_crf.py > results/fix_crf.log 2>&1
python3 fix_svm.py > results/fix_svm.log 2>&1
python3 fix_bilstm.py > results/fix_bilstm.log 2>&1
echo ALLDONE > results/fix_done.flag
