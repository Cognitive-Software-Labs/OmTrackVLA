"""Replay identical recorded RGB under A/B/B/A prompts; no identity accuracy claim."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def load_clip(folder):
    manifest = json.loads((folder / 'clip.json').read_text())
    rows = manifest['samples']
    if not manifest.get('complete') or len(rows) < 2:
        raise ValueError('Clip must be complete with at least two frames')
    for row in rows:
        path = (folder / row['image']).resolve()
        if path.parent != folder.resolve():
            raise ValueError('Frame must be directly inside the clip directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['image_sha256']:
            raise ValueError('Input checksum mismatch: ' + str(path))
    times = [row['elapsed_s'] for row in rows]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('Clip timestamps must increase')
    return manifest


def compare_runs(runs):
    import numpy as np
    values = [np.asarray([r['trajectory'] for r in run], dtype=float) for run in runs]
    if any(v.ndim != 3 or v.shape[1:] != (8, 3) or not np.isfinite(v).all() for v in values):
        raise ValueError('Expected finite frame x 8 x 3 arrays')
    def difference(a, b):
        xy = np.linalg.norm(a[:, :, :2] - b[:, :, :2], axis=2)
        return dict(median_xy_difference=float(np.median(xy)),
                    max_xy_difference=float(np.max(xy)),
                    median_absolute_yaw_difference=float(np.median(np.abs(np.arctan2(
                        np.sin(a[:, :, 2]-b[:, :, 2]), np.cos(a[:, :, 2]-b[:, :, 2]))))))
    return dict(prompt_A_vs_B=difference(values[0], values[1]),
                repeat_A=difference(values[0], values[3]),
                repeat_B=difference(values[1], values[2]),
                qualification='Output differences only; not identity accuracy, waypoint error or confidence.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clip', type=Path, required=True)
    p.add_argument('--prompt-a', required=True)
    p.add_argument('--prompt-b', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--assets', type=Path, default=Path(os.environ.get('OMTRACK_ASSETS', Path(__file__).resolve().parents[1])))
    p.add_argument('--device', default='cuda')
    p.add_argument('--validate-only', action='store_true')
    a = p.parse_args()
    if not a.prompt_a.strip() or not a.prompt_b.strip() or a.prompt_a.strip() == a.prompt_b.strip():
        p.error('Provide two distinct nonempty prompts')
    manifest = load_clip(a.clip)
    if a.validate_only:
        print(json.dumps(dict(validated_frames=len(manifest['samples']), complete=True)))
        return
    if a.output.exists():
        p.error('Use a new output directory')
    import cv2
    import numpy as np
    import torch
    from real_robot.inference_server import OmTrackInference
    frames = []
    for row in manifest['samples']:
        frame = cv2.imread(str(a.clip / row['image']))
        if frame is None:
            raise ValueError('Cannot decode ' + row['image'])
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    torch.set_num_threads(4)
    torch.manual_seed(0)
    model = OmTrackInference(a.assets.resolve(), a.device, 31, .1, 1, .35, .25, .25, 8, 3, .72,
                             mode='model_only', cuda_vision_preprocess=False)
    a.output.mkdir(parents=True)
    metadata = dict(clip=str(a.clip.resolve()), manifest=manifest,
                    prompt_a=a.prompt_a, prompt_b=a.prompt_b, order=['A', 'B', 'B', 'A'],
                    history=31, preprocessing='legacy_cpu_pil', mode='model_only',
                    playback='One fresh inference per sampled frame; history preserved within each run, reset between runs.',
                    timing='Source timestamps recorded; no wall-clock replay pacing or robot movement.',
                    torch_version=torch.__version__, device=a.device)
    (a.output / 'experiment.json').write_text(json.dumps(metadata, indent=2)+'\n')
    runs = []
    for index, prompt in enumerate([a.prompt_a, a.prompt_b, a.prompt_b, a.prompt_a]):
        results = []
        with (a.output / f'run-{index+1}.jsonl').open('w') as log:
            for i, (row, rgb) in enumerate(zip(manifest['samples'], frames)):
                # Offline replay must not substitute cached outputs based on GPU speed.
                model.next_planner_due = 0.0
                started = time.perf_counter()
                reply = model.infer(rgb, prompt, reset_target=(i == 0))
                points = np.asarray(reply['trajectory'], dtype=float)
                if not reply['planner_updated'] or points.shape != (8, 3) or not np.isfinite(points).all():
                    raise RuntimeError('Expected a fresh finite 8x3 prediction')
                result = dict(frame=row, prompt=prompt, elapsed_s=time.perf_counter()-started, **reply)
                log.write(json.dumps(result)+'\n')
                log.flush()
                results.append(result)
        runs.append(results)
        print(json.dumps(dict(completed_run=index+1, frames=len(results))), flush=True)
    summary = compare_runs(runs)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
