"""Measure this checkout's native perception/full branches without DyNAMO or ROS."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import platform
import signal
import socket
import sys
import time

from real_robot.defaults import DEFAULT_PROMPT
from real_robot.files import sha256_bytes, sha256_file
from real_robot.latency import percentiles
from real_robot.model_client import load_image_for_model
from real_robot.owned_processes import GPU_MONITOR, OwnedProcesses, require_idle_gpu

ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = [
    'real_robot/target_perception.py',
    'real_robot/inference_server.py',
    'cache_gridpool.py',
    'real_robot/benchmark_native_modes.py',
    'real_robot/model_client.py',
]
VLA_MODEL_NAMES = (
    'OmTrackVLA-0.6B',
    'Qwen3-0.6B',
    'dinov3-vits16-pretrain-lvd1689m',
    'siglip-so400m-patch14-384',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('case', choices=['perception', 'full'])
    p.add_argument('--samples', type=Path, default=ROOT / 'data/benchmark/samples.json')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data/benchmark')
    p.add_argument('--assets', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(a.output.resolve() / 'yolo-config')
    import cv2  # noqa: F401  (unused here; kept first so native libraries load in the original order)
    import numpy as np
    import torch
    import transformers
    from real_robot.inference_server import OmTrackInference
    import ultralytics
    torch.set_num_threads(4)

    report = {
        'case': a.case,
        'status': 'STARTING',
        'host': socket.gethostname(),
        'prompt': DEFAULT_PROMPT,
        'repetitions': 2,
        'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        'runtime': {
            'python': sys.version,
            'architecture': platform.machine(),
            'torch': torch.__version__,
            'cuda': torch.version.cuda,
            'transformers': transformers.__version__,
            'ultralytics': ultralytics.__version__,
        },
        'settings': {
            'grounding_box_threshold': .35,
            'grounding_text_threshold': .25,
            'person_threshold': .25,
            'grounding_interval': 8,
            'target_acquire_frames': 3,
            'reid_match_threshold': .72,
            'planner_rate': 1e9,
            'history': 31,
            'preprocessing': 'legacy PIL',
        },
        'sample_manifest_sha256': sha256_file(a.samples),
        'reset': 'clip boundary and repetition',
        'scope': 'Native OmTrackVLA target-perception and full modes; no DyNAMO, distance estimators, ROS or robot motion',
        'started_unix': time.time(),
    }
    rows = []
    owned_record = []
    processes = OwnedProcesses(a.output, owned_record)

    def stop(*_):
        raise KeyboardInterrupt('Stopping owned native-mode test')

    signal.signal(signal.SIGTERM, stop)
    try:
        require_idle_gpu(report)
        report['owned_processes'] = owned_record
        processes.spawn('gpu', GPU_MONITOR)
        processes.spawn('host-metrics', [sys.executable, str(ROOT / 'real_robot/sample_compute_host.py')])
        if platform.machine() == 'aarch64':
            processes.spawn('tegrastats', ['tegrastats', '--interval', '500'])
        report['runtime']['device'] = torch.cuda.get_device_name(0)

        samples = json.loads(a.samples.read_text())['samples']
        prepared = []
        for row in samples:
            path = a.data_root / row['clip'] / Path(row['image']).name
            if sha256_file(path) != row['image_sha256']:
                raise RuntimeError('RGB source checksum mismatch')
            jpeg, rgb = load_image_for_model(path)
            prepared.append((row, rgb))
            report.setdefault('jpeg_sha256', {})[row['id']] = sha256_bytes(jpeg)

        models = {
            'grounding-dino-tiny': a.assets / 'models/grounding-dino-tiny/model.safetensors',
            'yolo11n.pt': a.assets / 'models/yolo11n.pt',
        }
        if a.case == 'full':
            models.update({name: a.assets / 'models' / name / 'model.safetensors' for name in VLA_MODEL_NAMES})
        report['model_sha256'] = {name: sha256_file(path) for name, path in models.items()}

        begin = time.perf_counter()
        model = OmTrackInference(a.assets, 'cuda', 31, .1, 1, .35, .25, .25, 8, 3, .72,
                                 planner_rate=1e9, cuda_vision_preprocess=False, mode=a.case)
        torch.cuda.synchronize()
        report['first_observed_load_s'] = time.perf_counter() - begin

        # Count grounding-model forwards so per-frame latency can be split by whether grounding ran.
        ground_calls = [0]

        def grounding_hook(*_):
            ground_calls[0] += 1

        hook = model.target_perception.grounding_model.register_forward_pre_hook(grounding_hook)

        # Warm up on the standing clip (reset on its first frame), then reset once more on frame 0.
        warm = []
        for row, rgb in prepared:
            if row['clip'] == '02-standing':
                warm.append(model.infer(rgb, DEFAULT_PROMPT, reset_target=not warm))
        model.infer(prepared[0][1], DEFAULT_PROMPT, reset_target=True)
        report['warmup_person_frames'] = len(warm)
        report['warmup_vla_updates'] = sum(r.get('planner_updated') is True for r in warm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        durations = []
        for rep in range(2):
            previous = None
            began = time.perf_counter()
            for row, rgb in prepared:
                before_ground = ground_calls[0]
                before = time.perf_counter()
                reply = model.infer(rgb, DEFAULT_PROMPT, reset_target=row['clip'] != previous)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter() - before) * 1000
                previous = row['clip']
                fresh = reply.get('planner_updated') is True
                trajectory = reply.get('trajectory')
                if fresh and (np.asarray(trajectory).shape != (8, 3) or not np.isfinite(trajectory).all()):
                    raise RuntimeError('Invalid fresh VLA output')
                rows.append({
                    'id': row['id'],
                    'clip': row['clip'],
                    'repetition': rep,
                    'latency_ms': elapsed,
                    'target_state': reply['target_state'],
                    'target_valid': reply['target_valid'],
                    'target_reason': reply['target_reason'],
                    'people_count': reply['target_people_count'],
                    'target_bbox': reply['target_bbox'],
                    'grounding_forwards': ground_calls[0] - before_ground,
                    'fresh_vla': fresh,
                    'trajectory_valid': reply.get('trajectory_valid'),
                    'trajectory_reason': reply.get('trajectory_reason'),
                    'trajectory': trajectory if fresh else None,
                })
            durations.append(time.perf_counter() - began)
            print(json.dumps({
                'case': a.case,
                'rep': rep,
                'frames': len(prepared),
                'seconds': durations[-1],
                'fresh_vla': sum(r['fresh_vla'] for r in rows if r['repetition'] == rep),
            }), flush=True)
        hook.remove()

        fresh_count = sum(r['fresh_vla'] for r in rows)
        report.update(
            status='COMPLETE',
            sample_count=len(prepared),
            measured_calls=len(rows),
            duration_s=sum(durations),
            processed_frames_s=len(rows) / sum(durations),
            fresh_vla_updates=fresh_count,
            fresh_vla_updates_s=fresh_count / sum(durations),
            valid_fresh_waypoint_outputs=sum(r['fresh_vla'] and r['trajectory_valid'] for r in rows),
            locked_frames=sum(r['target_valid'] for r in rows),
            target_states=dict(Counter(r['target_state'] for r in rows)),
            target_reasons=dict(Counter(r['target_reason'] for r in rows)),
            grounding_forwards=sum(r['grounding_forwards'] for r in rows),
            latency_ms=percentiles([r['latency_ms'] for r in rows]),
            fresh_vla_frame_latency_ms=percentiles([r['latency_ms'] for r in rows if r['fresh_vla']]),
            grounding_frame_latency_ms=percentiles([r['latency_ms'] for r in rows if r['grounding_forwards']]),
            ordinary_tracking_frame_latency_ms=percentiles(
                [r['latency_ms'] for r in rows if not r['grounding_forwards']]),
            peak_cuda_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
        )
    except BaseException as e:
        report.update(status='FAILED', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        processes.stop(timeout=8)
        report['finished_unix'] = time.time()
        (a.output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        (a.output / 'outputs.json').write_text(json.dumps(rows, indent=2) + '\n')
    summary_keys = ('case', 'status', 'latency_ms', 'locked_frames', 'fresh_vla_updates')
    print(json.dumps({k: report[k] for k in summary_keys}), flush=True)


if __name__ == '__main__':
    main()
