"""Read-only compatibility check: released HF weights versus the trainer model."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Use a fresh output report path')
    assets = a.assets.resolve()
    checkpoint = assets/'models'/'OmTrackVLA-0.6B'
    qwen = assets/'models'/'Qwen3-0.6B'
    for path in (checkpoint/'model.safetensors', qwen/'model.safetensors'):
        if not path.is_file():
            raise FileNotFoundError(path)

    import torch
    import transformers
    from open_trackvla_hf import OpenTrackVLAConfig, OpenTrackVLAForWaypoint
    from train import ModelConfig as TrainModelConfig, OpenTrackVLA as TrainOpenTrackVLA
    torch.set_num_threads(4)
    torch.manual_seed(187)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    config = OpenTrackVLAConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.llm_name = str(qwen)
    config.freeze_llm = True
    device = torch.device(a.device)
    start = time.perf_counter()
    released = OpenTrackVLAForWaypoint.from_pretrained(
        str(checkpoint), config=config, local_files_only=True,
        low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if device.type == 'cuda' else None,
    ).eval().to(device)
    load_released_s = time.perf_counter()-start

    train_cfg = TrainModelConfig(llm_name=str(qwen), freeze_llm=True,
        n_waypoints=config.n_waypoints, max_time=config.max_time,
        beta_nav=config.beta_nav, use_angle_tvi=config.use_angle_tvi,
        use_tanh_actions=config.use_tanh_actions, alpha_xy=config.alpha_xy)
    start = time.perf_counter()
    trainer = TrainOpenTrackVLA(train_cfg, vision_feat_dim=config.vision_feat_dim).eval().to(device)
    load_trainer_s = time.perf_counter()-start
    source = released.model.state_dict()
    target = trainer.state_dict()
    missing = sorted(set(target)-set(source))
    unexpected = sorted(set(source)-set(target))
    shape_mismatches = {key:{'released':list(source[key].shape),'trainer':list(target[key].shape)}
                        for key in set(source)&set(target) if source[key].shape != target[key].shape}
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(json.dumps(dict(missing=missing, unexpected=unexpected,
                                           shape_mismatches=shape_mismatches), indent=2))
    trainer.load_state_dict(source, strict=True)

    generator = torch.Generator(device='cpu').manual_seed(593)
    coarse = torch.randn((1,31*4,config.vision_feat_dim), generator=generator).to(device)
    fine = torch.randn((1,64,config.vision_feat_dim), generator=generator).to(device)
    coarse_t = torch.arange(31, device=device).repeat_interleave(4).unsqueeze(0)
    fine_t = torch.full((1,64),31,dtype=torch.long,device=device)
    instruction = ['Follow the person wearing a green jacket.']
    with torch.inference_mode():
        a_pred = released(coarse,coarse_t,fine,fine_t,instruction)
        b_pred = trainer(coarse,coarse_t,fine,fine_t,instruction)
    max_abs = float((a_pred.float()-b_pred.float()).abs().max().cpu())
    if a_pred.shape != (1,8,3) or b_pred.shape != (1,8,3) or not torch.isfinite(a_pred).all() or max_abs > 1e-5:
        raise RuntimeError(f'Pre-training prediction parity failed: shape={list(a_pred.shape)}, max_abs={max_abs}')
    if any(param.requires_grad for param in trainer.llm.parameters()):
        raise RuntimeError('Requested frozen language backbone still has trainable parameters')
    if any(param.requires_grad for param in trainer.tvi.bbox_proj.parameters()):
        raise RuntimeError('Unused bbox projection must stay frozen until the trainer consumes bbox features')
    if not any(param.requires_grad for param in trainer.proj.parameters()) or not any(param.requires_grad for param in trainer.planner.parameters()):
        raise RuntimeError('Projector and waypoint head should remain trainable')
    report = dict(status='PASS', timestamp_utc=datetime.now(timezone.utc).isoformat(),
        read_only=True, training_steps=0, optimizer_created=False,
        device=str(device), torch=torch.__version__, transformers=transformers.__version__,
        released_checkpoint_sha256=sha256(checkpoint/'model.safetensors'),
        qwen_checkpoint_sha256=sha256(qwen/'model.safetensors'),
        released_load_seconds=load_released_s, trainer_init_seconds=load_trainer_s,
        released_state_keys=len(source), trainer_state_keys=len(target),
        missing_keys=missing, unexpected_keys=unexpected, shape_mismatches=shape_mismatches,
        strict_state_load=True, synthetic_feature_forward_used=True,
        prediction_shape=list(a_pred.shape), prediction_max_absolute_difference=max_abs,
        trainable_groups={'qwen_frozen':True,
                          'projector':any(p.requires_grad for p in trainer.proj.parameters()),
                          'waypoint_head':any(p.requires_grad for p in trainer.planner.parameters()),
                          'unused_bbox_projection_frozen':not any(p.requires_grad for p in trainer.tvi.bbox_proj.parameters())},
        limits=['Feature inputs were analytical random tensors, not an image or accuracy test.',
                'Does not validate waypoint label semantics, dataset quality or training memory.'])
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as stream:
        stream.write(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    del trainer,released
    if device.type == 'cuda':
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
