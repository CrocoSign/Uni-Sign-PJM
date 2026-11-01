#!/bin/bash
# Uni-Sign How2Sign transfer with Polish targets (pl_fixed of texts_fixed.csv), MS and filtered SI,
# same hyperparameters as the English rows in queue_coco133_filtered.sh. No BLEURT: it is English-only.
# Start: setsid nohup bash scripts/queue_polish.sh > queue_polish.log 2>&1 < /dev/null &
set -o pipefail
cd "$(dirname "$0")/.."
echo $$ > queue_polish.pid

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$(pwd)/.venv/bin:$PATH
export WANDB_MODE=offline HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

OUT=out/polish
TEXTS=../CrocoSign/data/texts_pl_fixed.h5
mkdir -p "$OUT"
echo "=== start $(date) ==="

RUNS=(
  "pjm_pl_ms_how2sign ms 15 1 pretrained_weight/how2sign_pose_only_slt.pth"
  "pjm_pl_si_how2sign filtered 30 2 pretrained_weight/how2sign_pose_only_slt.pth"
)

for spec in "${RUNS[@]}"; do
    read -r name split epochs warmup ft <<< "$spec"
    run_dir="$OUT/$name"; eval_dir="$OUT/eval_$name"
    if [ -f "$run_dir/DONE" ]; then
        echo "SKIP train $name"
    else
        echo "=== train $name ($split, $epochs ep) $(date) ==="
        rm -rf "$run_dir"; mkdir -p "$run_dir"
        deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
            --batch-size 8 --gradient-accumulation-steps 2 \
            --epochs "$epochs" --warmup-epochs "$warmup" \
            --opt AdamW --lr 3e-4 --weight-decay 0.01 \
            --finetune "$ft" \
            --dataset PJM --task SLT --pjm_split "$split" \
            --pjm_texts "$TEXTS" --pjm_lang Polish \
            --num_examples 5 \
            --output_dir "$run_dir" \
            --wandb --wandb_project uni-sign-pjm-polish --wandb_dir ./wandb_logs \
            > "$run_dir/train.log" 2>&1
        st=$?
        echo "=== train $name exit=$st $(date) ==="
        if [ $st -ne 0 ] || [ ! -f "$run_dir/best_checkpoint.pth" ]; then
            tail -n 40 "$run_dir/train.log"; continue
        fi
        rm -f "$run_dir"/checkpoint_*.pth
        touch "$run_dir/DONE"
    fi
    if [ ! -f "$eval_dir/test_tmp_pres.txt" ]; then
        echo "=== eval $name $(date) ==="
        rm -rf "$eval_dir"; mkdir -p "$eval_dir"
        torchrun --nproc_per_node=1 --master_port=29501 fine_tuning.py \
            --eval --finetune "$run_dir/best_checkpoint.pth" \
            --dataset PJM --task SLT --pjm_split "$split" \
            --pjm_texts "$TEXTS" --pjm_lang Polish \
            --num_examples 5 --batch-size 8 --num_workers 4 \
            --output_dir "$eval_dir" \
            > "$eval_dir/log.txt" 2>&1 || { echo "eval $name failed"; tail -n 40 "$eval_dir/log.txt"; continue; }
    fi
    tail -n 1 "$eval_dir/log.txt"
done
echo "=== done $(date) ==="
