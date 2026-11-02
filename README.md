# Uni-Sign-PJM

Fork of [Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) (Li et al., ICLR 2025, [arXiv:2501.15187](https://arxiv.org/abs/2501.15187)) adapted to translate Polish Sign Language (PJM) into English. All model code, pretrained checkpoints and the method are the work of the Uni-Sign authors; please cite their paper (see Citation). Our changes are listed below.

Part of CrocoSign, our PJATK engineering thesis. All repositories are under [github.com/crocosign](https://github.com/crocosign), project site https://crocosign.pl:

CrocoSign (data preparation) -> **Uni-Sign-PJM** / SpaMo-PJM (training) -> pjm-models (inference) -> croco-sign-backend / croco-sign-frontend (app).

## PJM adaptation

What we changed relative to upstream:

- `datasets.py`: new `S2T_Dataset_PJM` that reads split CSVs, English texts from an HDF5 file and pre-extracted pose files; optional replacement of the training targets by a teacher's outputs (`--pjm_train_texts`, mixed with the references by `--pjm_train_texts_p`).
- `config.py`, `fine_tuning.py`, `utils.py`: `--dataset PJM` with `--pjm_split {si,ms,filtered}` and `--pjm_train_csv` (k-fold teachers); optional LoRA on the mT5 decoder (`--lora`, `--lora_target`, `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--lora_ckpt`); `--freeze_visual`; BERTScore (`--bertscore`); W&B logging (`--wandb`, `--wandb_project`, `--wandb_dir`); early stopping on dev BLEU-4 (`--patience`, off by default, single GPU only; a run stopped early writes no `final_test_*`); the test set is evaluated only after the last epoch and written to `log.txt` as `final_test_*`; `--quick_break` now actually stops an epoch early.
- `SLRT_metrics.py`: chrF and a few output diagnostics (empty, very short and repetitive hypotheses); ROUGE no longer crashes on empty strings.
- `demo/extract_pose_pjm.py`: runs RTMPose (via the bundled `rtmlib`) over mp4 clips stored in tar shards and writes one pose `.pkl` per clip in the COCO-WholeBody layout of 133 points.
- `scripts/`: the runners behind every number in the paper (see Reproduce), the ensemble decoder `scripts/ensemble_eval.py` and `scripts/openpose134_to_coco133.py`.
- `pyproject.toml`, `uv.lock`: Python 3.10 environment managed with uv.

We use the pose-only setting. The RGB stream and Prior-Guided Fusion are not used. Targets are English: our Polish transcripts were machine-translated to English (DeepL) and used as references in all experiments.

### Data layout expected by the code

The paths are hard-coded relative to the repository root, so adapt them if your layout differs.

- Split files: `../CrocoSign/data/split_{train,val,test}{suffix}.csv` with a `key` column. `--pjm_split si` uses no suffix, `ms` uses `_ms`, `filtered` uses `_filtered`. The paper's SI rows use `filtered`.
- Texts: `../CrocoSign/data/texts_eng.h5` (key -> English sentence).
- Poses: `./dataset/PJM/pose_format/<key>.pkl`. Samples with a missing pose file are skipped with a printed count.
- Splits and texts are produced by [pjm-data](https://github.com/crocosign/pjm-data). [CrocoSign/pjm-data](https://huggingface.co/datasets/CrocoSign/pjm-data) publishes the features, splits, translations, videos and RTMPose poses of the filtered set (29,932 pairs); [CrocoSign/pjm-segments](https://huggingface.co/datasets/CrocoSign/pjm-segments) holds the unfiltered segments (about 40k) as WebDataset shards.
- The CSV layout above (`../CrocoSign/data/split_*.csv`) differs from the `splits/` folder of pjm-data: download the files from there and place them at the paths the code expects, or change the paths in `fine_tuning.py`.

Our poses are in `poses/` of CrocoSign/pjm-data; unpack the shards and move `pose_coco133/` to `dataset/PJM/pose_format/`. To extract them yourself from the tar shards of pjm-segments:

```bash
python demo/extract_pose_pjm.py --tars_dir /path/to/tar_shards --tgt_dir dataset/PJM/pose_format
```

Optional flags: `--device {cpu,cuda,mps}`, `--backend {opencv,onnxruntime,openvino}`, `--mode {performance,lightweight,balanced}`, `--max_workers`, `--overwrite`.

### Starting checkpoints

Download from the upstream model page [ZechengLi19/Uni-Sign](https://huggingface.co/ZechengLi19/Uni-Sign) and place them where the scripts expect:

- CSL-News stage 2: `out/csl_news_stage2/csl_stage2_weight.pth`
- How2Sign pose-only SLT: `pretrained_weight/how2sign_pose_only_slt.pth`
- OpenASL pose-only SLT: `pretrained_weight/openasl_pose_only_slt.pth`

According to the Uni-Sign paper, the two ASL checkpoints were pretrained on YouTube-ASL and fine-tuned on How2Sign or OpenASL; the CSL-News one comes from the RGB-pose pretraining stage, so its RGB branch is dropped when loading.

### Environment

```bash
uv sync
```

Python 3.10, CUDA 12.1 wheels for PyTorch 2.1.1, DeepSpeed 0.16.3. Training is launched with `deepspeed` on a single GPU. We trained on one RTX 3090 (24 GB). The scripts set CUDA paths for our workstation (`LD_LIBRARY_PATH`, `CUDA_HOME`); edit or remove these lines. BLEURT-20 is computed with the scorer of [SpaMo-PJM](https://github.com/crocosign/SpaMo-PJM) (`lucadiliello/BLEURT-20`), expected as a sibling clone `../SpaMo-PJM` with its own environment (override with `SPAMO_PJM=/path`). The runners set `HF_HUB_OFFLINE=1`, so download `lucadiliello/BLEURT-20` (in the SpaMo-PJM environment) and `roberta-large` (BERTScore, here) into the Hugging Face cache once before the first run; without them BLEURT prints `bleurt failed` and `--bertscore` cannot load its model.

## Results

Test set, beam 4, the checkpoint with the best dev BLEU-4. **MS** (multi-signer): test signers also appear in training. **SI** (signer-independent, filtered split): test signers are never seen in training. Data: 29,928 video-sentence pairs, 151 signers. BLEU is corpus-level SacreBLEU (13a); BLEURT is BLEURT-20 scaled to 0 to 100.

| Transfer source | Split | BLEU-4 | ROUGE-L | chrF | BLEURT |
| --- | --- | --- | --- | --- | --- |
| How2Sign | MS | 8.61 | 21.90 | 25.86 | 41.98 |
| How2Sign | SI | 5.91 | 17.37 | 22.36 | 37.83 |
| OpenASL | MS | 8.53 | 21.72 | 26.04 | 42.43 |
| OpenASL | SI | 6.09 | 17.62 | 22.82 | 38.16 |
| CSL-News | MS | 5.16 | 16.35 | 20.56 | 36.36 |
| CSL-News | SI | 2.41 | 11.36 | 16.80 | 31.31 |
| scratch | MS | 1.82 | 8.48 | 13.36 | 30.04 |
| scratch | SI | 0.85 | 6.11 | 13.07 | 30.08 |

Further MS runs, test BLEU-4 (BLEURT): a second seed of How2Sign 8.47 (41.74) and of OpenASL 8.67 (42.08); How2Sign started from OpenASL fine-tuned on How2Sign poses ("chain") 8.50 (41.85); a How2Sign student trained on a mix of references and k-fold ensemble outputs 8.01 (41.83). Ensembles (one beam search over the averaged token probabilities, `scripts/ensemble_eval.py`): MS {How2Sign, OpenASL} 9.58 (43.31), {How2Sign, OpenASL, chain} 9.81 (43.78), {How2Sign, OpenASL, chain, CSL-News} 10.01 (43.74, not significantly above 9.81); SI {How2Sign, OpenASL} 6.68 (39.19).

## Reproduce

Every run writes to `out/coco133/<name>/` (training, `best_checkpoint.pth`) and `out/coco133/eval_<name>/` (test outputs, `log.txt`, BLEURT). Our checkpoints are in [CrocoSign/pjm-models](https://huggingface.co/CrocoSign/pjm-models) under `checkpoints/unisign/<name>/best_checkpoint.pth`.

| Rows | Command | Run names |
| --- | --- | --- |
| The eight table rows | `bash scripts/queue_coco133_filtered.sh` | `pjm_phase3_full_{ms,si}_how2sign`, `..._openasl`, `pjm_phase3_full_{ms,si}` (CSL-News), `..._scratch` |
| Second seeds | `bash scripts/run_seed.sh how2sign`, `bash scripts/run_seed.sh openasl` | `pjm_phase3_full_ms_how2sign_s43`, `pjm_phase3_full_ms_openasl_s43` |
| Chain | `bash scripts/get_h2s_pose.sh`, unzip so that `dataset/How2Sign/{train,test}-pose-rtmpose-192/` exist (`config.py`), then `bash scripts/run_chain.sh` | `chain_openasl_how2sign`, `pjm_phase3_full_ms_chain` |
| Ensembles | `bash scripts/run_ensemble.sh`, then `run_after_chain.sh`, then `csl_ens.sh` (each reads the dev logs of the previous ones) | `out/coco133/ensemble_{ms,si}/` |
| k-fold student | `bash scripts/run_kd.sh` (makes the folds with `scripts/make_folds.py`; teachers 10 epochs) | `kfold_ms_{how2sign,openasl}_f{0..3}`, `pjm_phase3_full_ms_how2sign_kfold_mix` |
| Final ensemble (greedy on dev) | `bash scripts/run_greedy.sh`, after all of the above (pool of seven models, bootstrap against the ensembles) | `out/coco133/ensemble_ms/greedy/` |
| Polish targets (How2Sign, MS and SI) | `python scripts/make_pl_texts.py texts_fixed.csv ../CrocoSign/data/texts_pl_fixed.h5` (`texts/texts_fixed.csv` from CrocoSign/pjm-data), then `bash scripts/queue_polish.sh`; it adds `--pjm_texts` and `--pjm_lang Polish` to the English How2Sign settings | `out/polish/pjm_pl_{ms,si}_how2sign/` |

Paired bootstrap tests: `scripts/boot_si.py` and `scripts/boot_csl.py` (run from `out/coco133/`), `scripts/boot_chain.py` (from the repository root), and the end of `scripts/greedy_ens.py`. `run_kfold.sh` (15-epoch teachers), `run_distill.sh` and `run_mix.sh` are earlier versions of the k-fold queue, kept for the record; do not run them before `run_kd.sh`, which would reuse their teachers. The early stopping in `run_chain.sh` picks the How2Sign epoch on How2Sign test, the authors' protocol (their code has no How2Sign dev set). The older `script/*pjm*.sh` commands have the same hyperparameters but were run on 134-point OpenPose-order poses, which the Uni-Sign code indexes as COCO-WholeBody (see Caveats); their results are superseded.

## Caveats

- Our first pose extraction stored 134 points in OpenPose order, while the code selects body and hand points by COCO-WholeBody indices, so the pose encoders received shifted points. All results above come from models retrained on 133-point poses; `scripts/openpose134_to_coco133.py` converts old files exactly.
- LoRA on frozen pose encoders (phase 2) collapsed into repeating tokens in our runs. Full fine-tuning fixed it, so the final models use no LoRA.
- Generalisation to unseen signers is weak (SI). The SI training set is about a third of the MS one, and about 30 percent of distinct test words never occur in training.
- One run per table row; two seeds moved MS test BLEU-4 by 0.14.
- `docs/PJM_TRANSFER_LEARNING_PLAN.md` and `docs/PHASE3_IMPLEMENTATION_REPORT.md` are our working notes (in Polish).

## Upstream

For Uni-Sign pretraining (stages 1 to 3 on CSL-News), the CSL-News dataset, other datasets, online inference and everything not listed above, see the [upstream README](https://github.com/ZechengLi19/Uni-Sign#readme). Upstream's own BLEURT evaluation uses a different BLEURT-20 checkpoint, as described there; our numbers come from the scorer above. Uni-Sign builds on GFSLT-VLP, CoSign, SSVP-SLT, MMPose and FunASR (see upstream acknowledgements).

## Citation

```bibtex
@article{li2025uni,
  title={Uni-Sign: Toward Unified Sign Language Understanding at Scale},
  author={Li, Zecheng and Zhou, Wengang and Zhao, Weichao and Wu, Kepeng and Hu, Hezhen and Li, Houqiang},
  journal={arXiv preprint arXiv:2501.15187},
  year={2025}
}
```

## License

Our changes are under the MIT licence; the upstream code is not (see `LICENSE`). The upstream repository has no licence; only the bundled `demo/rtmlib-main` carries its own Apache-2.0 licence. Upstream terms must be respected: check [ZechengLi19/Uni-Sign](https://github.com/ZechengLi19/Uni-Sign) before reuse. The PJM Corpus data is licensed under CC BY-SA 4.0. Our model weights are under CC BY-NC-SA 4.0, as the upstream checkpoints they start from are for non-commercial use.

## Authors

[Wojciech Sieńko](https://www.linkedin.com/in/wojciech-sie%C5%84ko/) and [Rostyslav Kovalenko](https://www.linkedin.com/in/rost-koff/), PJATK engineering thesis, supervised by dr Albert Śledzianowski, MBA.
