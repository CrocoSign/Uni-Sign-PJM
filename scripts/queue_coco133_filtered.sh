#!/bin/bash
# Retrain the 8 Uni-Sign phase-3 runs (SI on the filtered split, as in the paper) on COCO-133 poses (the originals used rtmlib
# to_openpose=True, 134 points, against COCO indices in datasets.py), then eval + BLEURT-20.
# Same hyperparameters as script/train_pjm_phase3_*.sh. HF offline. Training runs without wandb (every run with
# wandb online died in the dev eval with a DataLoader SIGPIPE).
# Start: setsid nohup bash scripts/queue_coco133_filtered.sh > queue_coco133_filtered.log 2>&1 < /dev/null &
set -o pipefail
cd "$(dirname "$0")/.."
echo $$ > queue_coco133_filtered.pid

export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$(pwd)/.venv/bin:$PATH
export WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1

OUT=out/coco133
SPAMO=${SPAMO_PJM:-../SpaMo-PJM}
PJM=dataset/PJM
mkdir -p "$OUT"
echo "=== start $(date) ==="

# 1. Poses: convert 134-point files once (the poses on HF are already COCO-133), keep the originals next to them.
first=$(ls "$PJM/pose_format" | grep -m1 '\.pkl$')
[ -n "$first" ] || { echo "no pose files in $PJM/pose_format"; exit 2; }
n_points=$(python -c "import pickle, numpy as np; d=pickle.load(open('$PJM/pose_format/$first','rb')); print(np.asarray(d['keypoints'][0]).shape[-2])")
if [ "$n_points" = 134 ]; then
    python scripts/openpose134_to_coco133.py "$PJM/pose_format" "$PJM/pose_format_coco133" --selftest || exit 2
    n_src=$(ls "$PJM/pose_format" | wc -l); n_dst=$(ls "$PJM/pose_format_coco133" | wc -l)
    [ "$n_src" = "$n_dst" ] || { echo "count mismatch $n_src vs $n_dst"; exit 2; }
    mv "$PJM/pose_format" "$PJM/pose_format_openpose134"
    mv "$PJM/pose_format_coco133" "$PJM/pose_format"
fi
python -c "import pickle, numpy as np; d=pickle.load(open('$PJM/pose_format/$first','rb')); assert np.asarray(d['keypoints'][0]).shape[-2]==133" || { echo "poses are not COCO-133"; exit 2; }

# name split epochs warmup finetune(empty = scratch); main transfer rows first
RUNS=(
  "pjm_phase3_full_ms_how2sign ms 15 1 pretrained_weight/how2sign_pose_only_slt.pth"
  "pjm_phase3_full_ms_openasl ms 15 1 pretrained_weight/openasl_pose_only_slt.pth"
  "pjm_phase3_full_si_how2sign filtered 30 2 pretrained_weight/how2sign_pose_only_slt.pth"
  "pjm_phase3_full_si_openasl filtered 30 2 pretrained_weight/openasl_pose_only_slt.pth"
  "pjm_phase3_full_ms ms 15 1 out/csl_news_stage2/csl_stage2_weight.pth"
  "pjm_phase3_full_si filtered 30 2 out/csl_news_stage2/csl_stage2_weight.pth"
  "pjm_phase3_full_ms_scratch ms 15 1"
  "pjm_phase3_full_si_scratch filtered 15 1"
)

for spec in "${RUNS[@]}"; do
    read -r name split epochs warmup ft <<< "$spec"
    run_dir="$OUT/$name"; eval_dir="$OUT/eval_$name"
    ft_arg=(); [ -n "$ft" ] && ft_arg=(--finetune "$ft")

    if [ -f "$run_dir/DONE" ]; then
        echo "SKIP train $name"
    else
        echo "=== train $name ($split, $epochs ep) $(date) ==="
        rm -rf "$run_dir"; mkdir -p "$run_dir"
        deepspeed --include localhost:0 --master_port 29514 fine_tuning.py \
            --batch-size 8 --gradient-accumulation-steps 2 \
            --epochs "$epochs" --warmup-epochs "$warmup" \
            --opt AdamW --lr 3e-4 --weight-decay 0.01 \
            "${ft_arg[@]}" \
            --dataset PJM --task SLT --pjm_split "$split" \
            --bertscore --num_examples 5 \
            --output_dir "$run_dir" \
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
            --bertscore --num_examples 5 --batch-size 8 --num_workers 4 \
            --output_dir "$eval_dir" \
            > "$eval_dir/log.txt" 2>&1 || { echo "eval $name failed"; tail -n 40 "$eval_dir/log.txt"; continue; }
    fi
    if [ ! -f "$eval_dir/bleurt_BLEURT-20.json" ]; then
        echo "=== bleurt $name $(date) ==="
        "$SPAMO/.venv/bin/python" "$SPAMO/scripts/bleurt_unisign.py" "$eval_dir" >> "$eval_dir/log.txt" 2>&1 \
            || echo "bleurt $name failed"
    fi
    tail -n 1 "$eval_dir/log.txt"
    df -h / | tail -1
done
echo "=== done $(date) ==="
