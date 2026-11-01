#!/bin/bash
# Final MS ensemble (greedy selection on dev, one test decode), then BLEURT-20 of the chosen set.
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
LOG=out/coco133/greedy_ens.log
echo "start $(date)" >> $LOG
python scripts/greedy_ens.py >> $LOG 2>&1 || { echo "greedy failed $(date)" >> $LOG; exit 1; }
read -r _ dir _ tag < <(grep "^DIR " $LOG | tail -1)
${SPAMO_PJM:-../SpaMo-PJM}/.venv/bin/python scripts/bleurt_plain.py "$dir" "$tag" >> $LOG 2>&1
echo "end $(date)" >> $LOG
