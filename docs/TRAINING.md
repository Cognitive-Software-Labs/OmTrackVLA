# Training and simulator evaluation

Core inference does not require Habitat. Simulator reproduction additionally
requires Habitat-Sim 0.3.1 with Bullet, the bundled modified Habitat-Lab, licensed
scene datasets, avatars and robot assets referenced by the episode/config files.

## Optional simulator environment

```bash
conda env create -f requirements/simulation-environment.yml
conda activate omtrack-sim
python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/simulation.txt
python -m pip check
```

Use `OMTRACK_PYTHON="$(command -v python)"` with the launchers. The simulator
combination is preserved from upstream; installation/scene compatibility on a
fresh host must be checked independently of the core inference validation.

Obtain the HM3D/MP3D scene files and humanoid assets following the respective
providers' access/license instructions and the original TrackVLA dataset setup.
Populate `data/scene_datasets/` and `data/humanoids/`. Do not replace these with
unrelated scenes and describe the resulting demo as the published benchmark.
The upstream README references a humanoid downloader absent from its source
snapshot; this repository does not pretend it is available.

## Data conversion and cached features

```bash
python make_tracking_data.py --input_root sim_data/sample \
  --output_root data/sample --history 31 --horizon 8 --dt 0.1 \
  --instruction 'Follow the target person without collision.'
python precache_frames.py --data_root data/sample \
  --cache_root data/sample/vision_cache --batch_size 1 --image_size 384
```

Use local vision checkpoints by setting `DINOV3_MODEL_PATH` and
`SIGLIP_MODEL_PATH` to the downloaded folders. The launcher exports them for
its subprocesses. Verify frame/pose alignment, units, axes, dt and waypoint
shape before training. Audit found that the converter's `trajectory` contains
nine post-action points despite its origin-inclusive docstring. The trainer
ignores that field and derives eight origin-inclusive points from `actions`.
Resolve the target timing convention explicitly before preparing a new dataset;
do not compare the stored nine points directly against the model's eight outputs.
See [the waypoint contract audit](WAYPOINT_DATA_CONTRACT.md) for the verified
legacy timing, action scaling, history differences and explicit pose-label format.
Private stationary recordings alone are not expert movement labels. Preserve a
held-out split by person/session/scene; never split overlapping windows across
train and test. Do not label model predictions as human-approved ground truth.

## Existing trainer

```bash
python -m pip install -r requirements/training.txt
python train.py --train_json data/sample/jsonl \
  --cache_root data/sample/vision_cache --out_dir ckpts/example \
  --llm_name "$PWD/models/Qwen3-0.6B" --batch_size 1 \
  --epochs 1 --lr 2e-5 --mixed_precision --save_trajectories
```

This is the upstream training entry point, not yet a complete fine-tuning command
for the released waypoint checkpoint. It resumes training `.pt` state through
`--resume --resume_ckpt PATH`; it does not automatically initialize the complete
released safetensors model. `--freeze_llm` freezes Qwen while keeping the vision
projector and waypoint head trainable. The unused bbox projection is retained for
checkpoint compatibility and frozen until bbox features are consumed. A read-only
strict-load and prediction-parity check is available with
`scripts/validate_checkpoint_compat.py`; that check does not initialize or train a
new model. Pretrained fine-tuning still requires a trainer initialization path,
approved labels, session-level splits, and a measured single-step memory check.
Do not assume an 8 GB inference fit implies training fits; measure a single step.

Convert an approved trainer checkpoint with `convert_ckpt_to_hf.py --ckpt-path
PATH --out-dir PATH --llm-name LOCAL_QWEN_PATH`. Keep the original checkpoint,
configuration, source revision and dataset split. Re-run held-out accuracy and
compute tests before deploying a modified model.

## Simulator evaluation

```bash
HF_MODEL_DIR="$PWD/models/OmTrackVLA-0.6B" \
  OMTRACK_PYTHON="$(command -v python)" CUDA_VISIBLE_DEVICES=0 bash eval.sh
```

`CHUNKS`, `SAVE_PATH` and `EXP_CONFIG` select partitions/output/task config.
The default is the bundled STT configuration. Evaluation requires its referenced
scene files; Git cloning cannot provide restricted datasets. Ctrl-C stops only
the child owned by that launcher. No broad process-name kill is used.

## Accuracy benchmark

Report average/final waypoint displacement and heading error against valid
reference trajectories; also report wrong-target behavior, empty/lost-target
movement proposals and closed-loop success/collisions where actually tested.
Model-only mode provides no explicit person ID or absence confidence. Target-lock
counts from the optional wrapper and numerical output checks are not substitutes
for ground-truth trajectory accuracy or physical-following qualification.
