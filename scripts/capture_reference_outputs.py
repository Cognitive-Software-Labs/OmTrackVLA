"""Record, or compare against, exact model outputs before and after a refactor.

The planner section feeds fixed random vision features through both the released
Hugging Face loader (what inference_server uses) and the trainer's model class.
The pipeline section, which needs the gated DINOv3 weights, runs model_only
inference on frames of the bundled sample video. Same GPU and software stack
required: a reference is only comparable to itself, not across machines.
"""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROMPTS = ['Follow the person wearing a green jacket.',
           'Follow the person in the black T-shirt.',
           'Follow the target person without collision.']
SAMPLE_VIDEO = ROOT / 'sim_data/sample/seed_100/17DRP5sb8fy/4.mp4'
PIPELINE_FRAMES = 12


def deterministic_torch():
    # Must be set before CUDA initialises; cuBLAS needs it for deterministic GEMMs.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    return torch


def tensor_digest(torch, tensor):
    import hashlib
    return hashlib.sha256(tensor.detach().float().contiguous().cpu().numpy().tobytes()).hexdigest()


def planner_section(torch, assets, device):
    from omtrack.hf import OmTrackVLAConfig, OmTrackVLAForWaypoint
    from training.train import ModelConfig as TrainModelConfig, OmTrackVLA as TrainOpenTrackVLA
    checkpoint = assets / 'models/OmTrackVLA-0.6B'
    qwen = assets / 'models/Qwen3-0.6B'
    config = OmTrackVLAConfig.from_pretrained(str(checkpoint), local_files_only=True)
    config.llm_name = str(qwen)
    # Loaded exactly as inference_server.py does (default dtype).
    released = OmTrackVLAForWaypoint.from_pretrained(
        str(checkpoint), config=config, low_cpu_mem_usage=True).eval().to(device)
    state = released.model.state_dict()
    trainer = TrainOpenTrackVLA(TrainModelConfig(
        llm_name=str(qwen), freeze_llm=config.freeze_llm, n_waypoints=config.n_waypoints,
        max_time=config.max_time, beta_nav=config.beta_nav, use_angle_tvi=config.use_angle_tvi,
        use_tanh_actions=config.use_tanh_actions, alpha_xy=config.alpha_xy),
        vision_feat_dim=config.vision_feat_dim).eval().to(device)
    trainer.load_state_dict(state, strict=True)

    cases = []
    for seed, prompt in enumerate(PROMPTS):
        generator = torch.Generator(device='cpu').manual_seed(1000 + seed)
        coarse = torch.randn((1, 31 * 4, config.vision_feat_dim), generator=generator).to(device)
        fine = torch.randn((1, 64, config.vision_feat_dim), generator=generator).to(device)
        coarse_t = torch.arange(31, device=device).repeat_interleave(4).unsqueeze(0)
        fine_t = torch.full((1, 64), 31, dtype=torch.long, device=device)
        with torch.inference_mode():
            released_out = released(coarse, coarse_t, fine, fine_t, [prompt])
            trainer_out = trainer(coarse, coarse_t, fine, fine_t, [prompt])
        cases.append({'seed': 1000 + seed, 'prompt': prompt,
                      'released': released_out[0].float().cpu().tolist(),
                      'trainer': trainer_out[0].float().cpu().tolist()})
    section = {
        'config': {k: getattr(config, k) for k in ('freeze_llm', 'n_waypoints', 'max_time', 'beta_nav',
                                                  'use_angle_tvi', 'use_tanh_actions', 'alpha_xy',
                                                  'vision_feat_dim')},
        'state_dict': {k: {'shape': list(v.shape), 'dtype': str(v.dtype), 'sha256': tensor_digest(torch, v)}
                       for k, v in sorted(state.items())},
        'trainer_trainable': {name: any(p.requires_grad for p in module.parameters())
                              for name, module in trainer.named_children()},
        'cases': cases,
    }
    del released, trainer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return section


def pipeline_section(torch, assets, device):
    import cv2
    from real_robot.inference_server import OmTrackInference
    capture = cv2.VideoCapture(str(SAMPLE_VIDEO))
    frames = []
    while len(frames) < PIPELINE_FRAMES:
        ok, bgr = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) < PIPELINE_FRAMES:
        raise RuntimeError(f'Sample video gave only {len(frames)} frames')
    section = {'video': str(SAMPLE_VIDEO.relative_to(ROOT)), 'prompt': PROMPTS[2]}
    for name, cuda_preprocess in (('legacy_pil', False), ('cuda_preprocess', device.type == 'cuda')):
        if name == 'cuda_preprocess' and not cuda_preprocess:
            continue
        model = OmTrackInference(assets, str(device), 31, .1, 1, .35, .25, .25, 8, 3, .72,
                                 mode='model_only', cuda_vision_preprocess=cuda_preprocess)
        outputs = []
        for index, rgb in enumerate(frames):
            model.next_planner_due = 0.0  # every frame must run the planner, never a cached reply
            reply = model.infer(rgb, PROMPTS[2], reset_target=index == 0)
            outputs.append(reply['trajectory'])
        section[name] = outputs
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return section


def max_difference(a, b):
    import numpy as np
    return float(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)).max())


def compare(reference, current):
    """Return a list of human-readable mismatches; empty means bit-identical outputs."""
    problems = []
    for section in ('planner', 'pipeline'):
        if section not in reference:
            continue
        if section not in current:
            problems.append(f'{section}: missing from current run')
            continue
        ref, cur = reference[section], current[section]
        if section == 'planner':
            for key in ('config', 'state_dict', 'trainer_trainable'):
                if ref[key] != cur[key]:
                    changed = ([k for k in set(ref[key]) | set(cur[key]) if ref[key].get(k) != cur[key].get(k)])
                    problems.append(f'planner.{key} differs: {sorted(changed)[:10]}')
            for r, c in zip(ref['cases'], cur['cases']):
                for path in ('released', 'trainer'):
                    if r[path] != c[path]:
                        problems.append(f"planner seed {r['seed']} {path}: max abs diff "
                                        f"{max_difference(r[path], c[path]):.3e}")
        else:
            for path in ('legacy_pil', 'cuda_preprocess'):
                if path in ref and ref[path] != cur.get(path):
                    problems.append(f'pipeline {path}: max abs diff {max_difference(ref[path], cur[path]):.3e}')
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, default=Path(os.environ.get('OMTRACK_ASSETS', ROOT)))
    p.add_argument('--device', default='cuda')
    p.add_argument('--sections', nargs='+', choices=('planner', 'pipeline'), default=['planner', 'pipeline'])
    p.add_argument('--output', type=Path, help='Write a new reference JSON (never overwritten)')
    p.add_argument('--compare', type=Path, help='Reference JSON to compare against; exits 1 on any difference')
    a = p.parse_args()
    if bool(a.output) == bool(a.compare):
        p.error('Give exactly one of --output or --compare')
    if a.output and a.output.exists():
        p.error(f'{a.output} already exists; choose a new path')
    torch = deterministic_torch()
    import transformers
    device = torch.device(a.device)
    assets = a.assets.resolve()
    result = {'torch': torch.__version__, 'transformers': transformers.__version__,
              'device_name': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}
    if 'planner' in a.sections:
        result['planner'] = planner_section(torch, assets, device)
    if 'pipeline' in a.sections:
        result['pipeline'] = pipeline_section(torch, assets, device)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        with a.output.open('x') as stream:
            stream.write(json.dumps(result, indent=1) + '\n')
        print(f'Saved reference: {a.output}')
        return
    reference = json.loads(a.compare.read_text())
    for key in ('torch', 'transformers', 'device_name'):
        if reference[key] != result[key]:
            print(f'Warning: {key} differs ({reference[key]} vs {result[key]}); exact match not expected.')
    problems = compare(reference, result)
    for problem in problems:
        print('MISMATCH', problem)
    print('IDENTICAL' if not problems else f'{len(problems)} mismatches')
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
