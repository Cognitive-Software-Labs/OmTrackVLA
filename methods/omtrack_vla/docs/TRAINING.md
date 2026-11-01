# Training

The Habitat simulator stack (EVT-Bench episodes, modified Habitat-Lab, rollout
and evaluation scripts) was removed from this repository; simulation for the
leader-following work uses the DyNAMO-Ridgeback simulators (Gazebo, Isaac Sim)
instead. It remains in Git history before the removal commit. The two sample
rollouts in `samples/` were rendered by it and are kept as training-data and
inference fixtures.

## Data conversion and cached features

Run every command from the repository root. Library code lives in
`methods/omtrack_vla/` and training scripts in `methods/omtrack_vla/training/`;
`PYTHONPATH=.` makes the package importable.

```bash
PYTHONPATH=. python -m methods.omtrack_vla.training.make_tracking_data --input_root methods/omtrack_vla/samples \
  --output_root outputs/omtrack-data/sample --history 31 --horizon 8 --dt 0.1 \
  --instruction 'Follow the target person without collision.'
PYTHONPATH=. python -m methods.omtrack_vla.training.precache_frames --data_root outputs/omtrack-data/sample \
  --cache_root outputs/omtrack-data/sample/vision_cache --batch_size 1 --image_size 384
```

Use local vision checkpoints by setting `DINOV3_MODEL_PATH` and
`SIGLIP_MODEL_PATH` to the downloaded folders. Verify frame/pose alignment, units, axes, dt and waypoint
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
PYTHONPATH=. python -m methods.omtrack_vla.training.train --train_json outputs/omtrack-data/sample/jsonl \
  --cache_root outputs/omtrack-data/sample/vision_cache --out_dir ckpts/example \
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
`methods/omtrack_vla/scripts/validate_checkpoint_compat.py`; that check does not initialize or train a
new model. Pretrained fine-tuning still requires a trainer initialization path,
approved labels, session-level splits, and a measured single-step memory check.
Do not assume an 8 GB inference fit implies training fits; measure a single step.

Convert an approved trainer checkpoint with `python -m methods.omtrack_vla.training.convert_ckpt_to_hf --ckpt-path
PATH --out-dir PATH --llm-name LOCAL_QWEN_PATH`. Keep the original checkpoint,
configuration, source revision and dataset split. Re-run held-out accuracy and
compute tests before deploying a modified model.

## Checking a code change does not change outputs

Before refactoring model, loader or preprocessing code, record exact outputs at
the current commit, then compare after each change on the same GPU and stack:

```bash
PYTHONPATH=. .venv/bin/python methods/omtrack_vla/scripts/capture_reference_outputs.py \
  --output outputs/reference/before.json
PYTHONPATH=. .venv/bin/python methods/omtrack_vla/scripts/capture_reference_outputs.py \
  --compare outputs/reference/before.json
```

The `planner` section needs only the OmTrackVLA and Qwen weights and runs both
the released loader and the trainer class on fixed random features. The
`pipeline` section also needs DINOv3 and SigLIP; use `--sections planner` until
DINOv3 access is set up. The comparison requires bit-identical outputs.

## Accuracy benchmark

Report average/final waypoint displacement and heading error against valid
reference trajectories; also report wrong-target behavior, empty/lost-target
movement proposals and closed-loop success/collisions where actually tested.
Model-only mode provides no explicit person ID or absence confidence. Target-lock
counts from the optional wrapper and numerical output checks are not substitutes
for ground-truth trajectory accuracy or physical-following qualification.
