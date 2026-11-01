#!/bin/bash
# After the transfer chain: MS ensembles with the chain model (subset picked on dev, one test decode),
# SI H2S+OpenASL ensemble for the table, and the authors' How2Sign checkpoint on How2Sign test (eval sanity check).
cd "$(dirname "$0")/.."
export LD_LIBRARY_PATH=/usr/local/cuda/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export PATH=$(pwd)/.venv/bin:$PATH
export USE_TF=0 WANDB_MODE=disabled HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
echo "waiting for chain $(date)"
while pgrep -f "[r]un_chain.sh" > /dev/null; do sleep 60; done
echo "start $(date)"
BLEURT=${SPAMO_PJM:-../SpaMo-PJM}/.venv/bin/python
H=out/coco133/pjm_phase3_full_ms_how2sign/best_checkpoint.pth
O=out/coco133/pjm_phase3_full_ms_openasl/best_checkpoint.pth
C=out/coco133/pjm_phase3_full_ms_chain/best_checkpoint.pth
E=out/coco133/ensemble_ms
MS="--dataset PJM --task SLT --pjm_split ms --batch-size 8 --num_workers 4"
exec > >(tee -a $E/run_after_chain.log) 2>&1  # read back below and by csl_ens.sh

if [ -f out/coco133/pjm_phase3_full_ms_chain/DONE ]; then
    python scripts/ensemble_eval.py $MS --phase val --ckpt $H --ckpt $C --out_dir $E/h2s_chain || exit 1
    python scripts/ensemble_eval.py $MS --phase val --ckpt $O --ckpt $C --out_dir $E/openasl_chain || exit 1
    python scripts/ensemble_eval.py $MS --phase val --ckpt $H --ckpt $O --ckpt $C --out_dir $E/h2s_openasl_chain || exit 1
    # Registered rule: highest dev BLEU-4 among the four subsets, ties go to fewer models; only that one is decoded on test.
    best=$(grep -h "^RESULT val_" $E/run.log $E/run_after_chain.log | python -c '
import re, sys
names = {"H": "h2s", "O": "openasl", "C": "chain"}
best = None
for line in sys.stdin:
    cks = re.findall(r"pjm_phase3_full_ms_(how2sign|openasl|chain)/", line)
    if len(cks) < 2:
        continue
    bleu = float(re.search(r"bleu4=([0-9.]+)", line).group(1))
    key = (bleu, -len(cks))
    if best is None or key > best[0]:
        best = (key, "_".join({"how2sign": "h2s"}.get(c, c) for c in cks))
print(best[1])')
    echo "dev-best subset: $best"
    if [ "$best" != "h2s_openasl" ]; then
        ck=(); for m in ${best//_/ }; do case $m in h2s) ck+=(--ckpt $H);; openasl) ck+=(--ckpt $O);; chain) ck+=(--ckpt $C);; esac; done
        python scripts/ensemble_eval.py $MS --phase test "${ck[@]}" --out_dir $E/$best || exit 1
        n=$(ls $E/$best/test_*m_pres.txt | sed -E 's/.*test_([0-9]+)m.*/\1/')
        $BLEURT scripts/bleurt_plain.py $E/$best test_${n}m || echo "bleurt failed"
    fi
else
    echo "chain not DONE, skipping MS ensembles"
fi

S=out/coco133/ensemble_si/h2s_openasl
SI="--dataset PJM --task SLT --pjm_split filtered --batch-size 8 --num_workers 4"
HS=out/coco133/pjm_phase3_full_si_how2sign/best_checkpoint.pth
OS=out/coco133/pjm_phase3_full_si_openasl/best_checkpoint.pth
python scripts/ensemble_eval.py $SI --phase val --ckpt $HS --ckpt $OS --out_dir $S || exit 1
python scripts/ensemble_eval.py $SI --phase test --ckpt $HS --ckpt $OS --out_dir $S || exit 1
$BLEURT scripts/bleurt_plain.py $S test_2m || echo "bleurt failed"

A=out/coco133/eval_authors_how2sign
mkdir -p $A
torchrun --nproc_per_node=1 --master_port=29501 fine_tuning.py \
    --eval --finetune pretrained_weight/how2sign_pose_only_slt.pth \
    --dataset How2Sign --task SLT --batch-size 8 --num_workers 4 \
    --output_dir $A > $A/log.txt 2>&1 || echo "authors eval failed"
grep "bleu4" $A/log.txt | tail -1
echo "end $(date)"
