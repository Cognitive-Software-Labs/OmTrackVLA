"""Bounded OmTrackVLA-only compute or socket benchmark; no ROS or DyNAMO."""
import argparse
import base64
import json
from pathlib import Path
import platform
import signal
import socket
import sys
import time

from real_robot.defaults import BENCHMARK_PORT, DEFAULT_PROMPT
from real_robot.files import sha256_bytes, sha256_file
from real_robot.latency import percentiles
from real_robot.model_client import ModelClient, load_image_for_model
from real_robot.owned_processes import GPU_MONITOR, OwnedProcesses, require_idle_gpu

ROOT = Path(__file__).resolve().parents[1]

SOURCE_FILES = [
    'real_robot/benchmark_omtrack_only.py',
    'real_robot/inference_server.py',
    'real_robot/protocol.py',
    'real_robot/model_client.py',
    'cache_gridpool.py',
    'open_trackvla_hf/modeling_open_trackvla.py',
]
MODEL_NAMES = (
    'OmTrackVLA-0.6B',
    'Qwen3-0.6B',
    'dinov3-vits16-pretrain-lvd1689m',
    'siglip-so400m-patch14-384',
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['direct', 'socket'])
    p.add_argument('--samples', type=Path, default=ROOT / 'data/benchmark/samples.json')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data/benchmark')
    p.add_argument('--assets', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, choices=range(1, 4), default=2)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=BENCHMARK_PORT)
    p.add_argument('--source-address', help='Bind socket client to an existing local IPv4 address')
    p.add_argument('--launch-server', action='store_true')
    a = p.parse_args()
    if a.launch_server and (a.mode != 'socket' or a.host != '127.0.0.1' or a.assets.resolve() != ROOT):
        p.error('--launch-server requires socket mode, loopback and this checkout assets')
    a.output.mkdir(parents=True, exist_ok=False)
    import numpy as np

    samples = json.loads(a.samples.read_text())['samples']
    prepared = []  # (row, rgb, jpeg_base64, jpeg_sha256); base64 done here, outside timing
    for row in samples:
        path = a.data_root / row['clip'] / Path(row['image']).name
        if sha256_file(path) != row['image_sha256']:
            raise RuntimeError('Source RGB checksum mismatch: ' + row['id'])
        jpeg, rgb = load_image_for_model(path)
        prepared.append((row, rgb, base64.b64encode(jpeg).decode(), sha256_bytes(jpeg)))
    if not prepared:
        raise RuntimeError('Empty sample list')

    owned = a.mode == 'direct' or a.launch_server
    owned_record = []
    processes = OwnedProcesses(a.output, owned_record)
    model = client = None
    manifest = {
        'host': socket.gethostname(),
        'mode': a.mode,
        'model_mode': 'model_only',
        'status': 'STARTING',
        'started_unix': time.time(),
        'prompt': DEFAULT_PROMPT,
        'sample_count': len(samples),
        'repetitions': a.repetitions,
        'sample_manifest_sha256': sha256_file(a.samples),
        'history': 31,
        'reset': 'each clip boundary and repetition; three warmup calls excluded',
        'preprocessing': 'legacy PIL; JPEG quality 95 decoded identically for direct/socket',
        'source_sha256': {name: sha256_file(ROOT / name) for name in SOURCE_FILES},
        'scope': 'OmTrackVLA neural inference only; no target detector, mask, distance estimator, ROS, Nav2 or motion',
        'jpeg_sha256': {row['id']: sha for row, _, _, sha in prepared},
        'server_endpoint': f'{a.host}:{a.port}' if a.mode == 'socket' else None,
        'client_source_address': a.source_address,
    }
    rows = []

    def save():
        (a.output / 'report.json').write_text(json.dumps(manifest, indent=2) + '\n')
        (a.output / 'outputs.json').write_text(json.dumps(rows, indent=2) + '\n')

    def interrupted(*_):
        raise KeyboardInterrupt('Stopped; cleaning up owned benchmark processes')

    signal.signal(signal.SIGTERM, interrupted)
    try:
        if owned:
            require_idle_gpu(manifest)
            manifest['owned_processes'] = owned_record
            processes.spawn('gpu', GPU_MONITOR)
            processes.spawn('host-metrics', [sys.executable, str(ROOT / 'real_robot/sample_compute_host.py')])
        runtime = {}
        if owned:
            import torch
            import transformers
            torch.set_num_threads(4)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA required')
            runtime = {
                'python': sys.version,
                'architecture': platform.machine(),
                'torch': torch.__version__,
                'cuda': torch.version.cuda,
                'transformers': transformers.__version__,
                'device': torch.cuda.get_device_name(0),
            }
            manifest['runtime'] = runtime
            manifest['model_sha256'] = {
                name: sha256_file(a.assets / 'models' / name / 'model.safetensors') for name in MODEL_NAMES}

        began = time.perf_counter()
        if a.mode == 'direct':
            from real_robot.inference_server import OmTrackInference
            model = OmTrackInference(a.assets, 'cuda', 31, .1, 1, .3, .25, .35, 15, 3, .5,
                                     planner_rate=1e9, cuda_vision_preprocess=False, mode='model_only')
            torch.cuda.synchronize()
        else:
            if a.launch_server:
                # Fail fast if the port is taken instead of benchmarking someone else's server.
                with socket.socket() as probe:
                    probe.bind((a.host, a.port))
                server = processes.spawn('server', [
                    sys.executable, '-u', str(ROOT / 'real_robot/inference_server.py'),
                    '--mode', 'model_only', '--port', str(a.port),
                    '--planner-rate', '1000000000', '--legacy-vision-preprocess'])
            # Retry until the server listens; a launched server that exits ends the wait early.
            deadline = time.monotonic() + 90
            while True:
                try:
                    client = ModelClient(a.port, a.host, connect_timeout=3, timeout=30,
                                         source_address=a.source_address)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise
                    if a.launch_server and server.poll() is not None:
                        raise RuntimeError('Server exited; inspect server.log')
                    time.sleep(.1)
        manifest['first_observed_startup_s'] = time.perf_counter() - began

        request_id = 0

        def infer(item, reset):
            nonlocal request_id
            row, rgb, image_b64, _ = item
            if a.mode == 'direct':
                return model.infer(rgb, DEFAULT_PROMPT, reset_target=reset)
            request_id += 1
            return client.infer(image_b64, DEFAULT_PROMPT, request_id, reset, frame_id=row['id'])

        for i in range(3):
            infer(prepared[0], i == 0)
        if a.mode == 'direct':
            torch.cuda.reset_peak_memory_stats()

        durations = []
        for rep in range(a.repetitions):
            previous = None
            started = time.perf_counter()
            for item in prepared:
                row = item[0]
                before = time.perf_counter()
                reply = infer(item, row['clip'] != previous)
                latency = (time.perf_counter() - before) * 1000
                previous = row['clip']
                trajectory = np.asarray(reply['trajectory'], dtype=float)
                if (reply.get('planner_updated') is not True or trajectory.shape != (8, 3)
                        or not np.isfinite(trajectory).all()):
                    raise RuntimeError('Cached/nonfinite/wrong-shaped model output')
                rows.append({
                    'id': row['id'],
                    'clip': row['clip'],
                    'repetition': rep,
                    'latency_ms': latency,
                    'server_ms': reply.get('inference_seconds', 0) * 1000 if a.mode == 'socket' else None,
                    'fresh': True,
                    'trajectory': reply['trajectory'],
                    'trajectory_valid': reply['trajectory_valid'],
                })
            durations.append(time.perf_counter() - started)
            print(json.dumps({'mode': a.mode, 'repetition': rep, 'completed': len(prepared),
                              'seconds': durations[-1]}), flush=True)

        manifest.update(
            status='COMPLETE',
            measured_calls=len(rows),
            fresh_updates=len(rows),
            finite_waypoint_outputs=len(rows),
            latency_ms=percentiles([r['latency_ms'] for r in rows]),
            completed_fresh_updates_s=len(rows) / sum(durations),
            measured_duration_s=sum(durations),
        )
        if a.mode == 'socket':
            manifest['server_latency_ms'] = percentiles([r['server_ms'] for r in rows])
        if a.mode == 'direct':
            manifest['peak_cuda_allocated_mib'] = torch.cuda.max_memory_allocated() / 1024**2
    except BaseException as exc:
        status = 'CANCELLED' if isinstance(exc, KeyboardInterrupt) else 'FAILED'
        manifest.update(status=status, error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if client is not None:
            client.close()
        processes.stop(timeout=10)
        manifest['finished_unix'] = time.time()
        save()
    print(json.dumps(manifest.get('latency_ms')), flush=True)


if __name__ == '__main__':
    main()
