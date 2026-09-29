#!/bin/sh
cd /home/claude/proj
python3 improve_bilstm.py >> results/improve_bilstm.log 2>&1
echo ALLDONE > results/to_done.flag
