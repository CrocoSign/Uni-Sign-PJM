#!/bin/bash
# Final-ensemble queue (amended): k-fold teachers at 10 epochs, then the mixed distillation student only.
# Teachers: {How2Sign, OpenASL} trained on 3 of 4 folds of the MS train set; their ensemble translates the held-out fold.
# Student: How2Sign init, standard MS recipe; each draw takes the held-out teacher output with p=0.5, else the reference.
set -o pipefail
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
eval "$(sed -n '/^train() {/,/^}/p' scripts/run_chain.sh)"

evaluate() {  # name
    local name=$1 eval_dir=out/coco133/eval_$1
    [ -f "$eval_dir/test_tmp_pres.txt" ] || {
        rm -rf "$eval_dir"; mkdir -p "$eval_dir"
        torchrun --nproc_per_node=1 --master_port=29501 fine_tuning.py \
            --eval --finetune out/coco133/$name/best_checkpoint.pth \
            --dataset PJM --task SLT --pjm_split ms \
            --bertscore --num_examples 5 --batch-size 8 --num_workers 4 \
            --output_dir "$eval_dir" > "$eval_dir/log.txt" 2>&1 || { echo "eval failed"; tail -n 40 "$eval_dir/log.txt"; return 1; }
    }
    ${SPAMO_PJM:-../SpaMo-PJM}/.venv/bin/python ${SPAMO_PJM:-../SpaMo-PJM}/scripts/bleurt_unisign.py "$eval_dir" >> "$eval_dir/log.txt" 2>&1 || echo "bleurt failed"
    grep "bleu4" "$eval_dir/log.txt" | tail -1
    tail -n 1 "$eval_dir/log.txt"
}

echo "start $(date)"

K=out/coco133/kfold_ms
mkdir -p $K
[ -f $K/heldout_fold3.csv ] || python scripts/make_folds.py ../CrocoSign/data/split_train_ms.csv $K || exit 1
for k in 0 1 2 3; do
    train kfold_ms_how2sign_f$k 10 1 pretrained_weight/how2sign_pose_only_slt.pth PJM --pjm_split ms --pjm_train_csv $K/train_fold$k.csv || exit 1
    train kfold_ms_openasl_f$k 10 1 pretrained_weight/openasl_pose_only_slt.pth PJM --pjm_split ms --pjm_train_csv $K/train_fold$k.csv || exit 1
    [ -f $K/f$k/train_2m_pres.txt ] || python scripts/ensemble_eval.py --dataset PJM --task SLT --pjm_split ms --batch-size 8 --num_workers 4 \
        --phase train --csv $K/heldout_fold$k.csv \
        --ckpt out/coco133/kfold_ms_how2sign_f$k/best_checkpoint.pth --ckpt out/coco133/kfold_ms_openasl_f$k/best_checkpoint.pth \
        --out_dir $K/f$k || exit 1
    # Disk is tight: fold teachers are not ensemble members, drop their weights once the fold is decoded.
    rm -f out/coco133/kfold_ms_how2sign_f$k/best_checkpoint.pth out/coco133/kfold_ms_openasl_f$k/best_checkpoint.pth
done
python - $K <<'PY' || exit 1
import json, sys
d = sys.argv[1]
targets, same = {}, 0
for k in range(4):
    names = open(f"{d}/f{k}/train_2m_names.txt").read().splitlines()
    hyps = open(f"{d}/f{k}/train_2m_pres.txt").read().splitlines()
    refs = open(f"{d}/f{k}/train_2m_refs.txt").read().splitlines()
    assert len(names) == len(hyps) == len(refs) and not set(names) & set(targets), k
    targets.update(zip(names, hyps))
    same += sum(h.strip().lower() == r.strip().lower() for h, r in zip(hyps, refs))
print(f"held-out teacher outputs {len(targets)}, identical to reference {same}")
json.dump(targets, open(f"{d}/train_targets.json", "w"), ensure_ascii=False)
PY
echo "teachers done $(date)"

train pjm_phase3_full_ms_how2sign_kfold_mix 15 1 pretrained_weight/how2sign_pose_only_slt.pth PJM --pjm_split ms \
    --pjm_train_texts $K/train_targets.json --pjm_train_texts_p 0.5 || exit 1
evaluate pjm_phase3_full_ms_how2sign_kfold_mix || exit 1
echo "end $(date)"
