#!/bin/sh
cd /home/claude/proj
python3 trainonly_models.py > results/to_models.log 2>&1
python3 improve_bilstm.py > results/improve_bilstm.log 2>&1
echo ALLDONE > results/to_done.flag
