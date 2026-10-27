"""Bounded OmTrackVLA-only compute or socket benchmark; no ROS or DyNAMO."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROMPT = 'Follow the person wearing a black jacket or black T-shirt.'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['direct', 'socket'])
    p.add_argument('--samples', type=Path, default=ROOT/'data/benchmark/samples.json')
    p.add_argument('--data-root', type=Path, default=ROOT/'data/benchmark')
    p.add_argument('--assets', type=Path, default=ROOT)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repetitions', type=int, choices=range(1,4), default=2)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=18779)
    p.add_argument('--source-address', help='Bind socket client to an existing local IPv4 address')
    p.add_argument('--launch-server', action='store_true')
    a = p.parse_args()
    if a.launch_server and (a.mode != 'socket' or a.host != '127.0.0.1' or a.assets.resolve() != ROOT):
        p.error('--launch-server requires socket mode, loopback and this checkout assets')
    a.output.mkdir(parents=True, exist_ok=False)
    import cv2
    import numpy as np
    from real_robot.protocol import send_message, receive_message
    samples = json.loads(a.samples.read_text())['samples']
    prepared = []
    for row in samples:
        path = a.data_root/row['clip']/Path(row['image']).name
        if digest(path) != row['image_sha256']:
            raise RuntimeError('Source RGB checksum mismatch: '+row['id'])
        bgr = cv2.imread(str(path))
        if bgr is None:
            raise RuntimeError('Cannot decode '+str(path))
        ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise RuntimeError('JPEG encode failed')
        raw = jpg.tobytes()
        rgb = cv2.cvtColor(cv2.imdecode(jpg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        prepared.append((row, rgb, base64.b64encode(raw).decode(), hashlib.sha256(raw).hexdigest()))
    if not prepared:
        raise RuntimeError('Empty sample list')
    owned = a.mode == 'direct' or a.launch_server
    children, streams = [], []
    model = connection = None
    manifest = {'host':socket.gethostname(), 'mode':a.mode, 'model_mode':'model_only',
                'status':'STARTING', 'started_unix':time.time(), 'prompt':PROMPT,
                'sample_count':len(samples), 'repetitions':a.repetitions,
                'sample_manifest_sha256':digest(a.samples), 'history':31,
                'reset':'each clip boundary and repetition; three warmup calls excluded',
                'preprocessing':'legacy PIL; JPEG quality 95 decoded identically for direct/socket',
                'source_sha256':{name:digest(ROOT/name) for name in [
                    'real_robot/benchmark_omtrack_only.py','real_robot/inference_server.py',
                    'real_robot/protocol.py','cache_gridpool.py','open_trackvla_hf/modeling_open_trackvla.py']},
                'scope':'OmTrackVLA neural inference only; no target detector, mask, distance estimator, ROS, Nav2 or motion',
                'jpeg_sha256':{row['id']:sha for row,_,_,sha in prepared},
                'server_endpoint':f'{a.host}:{a.port}' if a.mode=='socket' else None,
                'client_source_address':a.source_address}
    rows = []
    def save():
        (a.output/'report.json').write_text(json.dumps(manifest,indent=2)+'\n')
        (a.output/'outputs.json').write_text(json.dumps(rows,indent=2)+'\n')
    def spawn(name, command):
        stream=(a.output/(name+'.log')).open('w');streams.append(stream)
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        children.append(child)
        manifest.setdefault('owned_processes',[]).append({'name':name,'pid':child.pid,'command':command})
        return child
    def interrupted(*_):
        raise KeyboardInterrupt('Stopped; cleaning up owned benchmark processes')
    signal.signal(signal.SIGTERM, interrupted)
    try:
        if owned:
            check=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True)
            manifest['gpu_before']=check.stdout
            if check.stdout.strip():
                raise RuntimeError('Competing GPU processes present: '+check.stdout)
            spawn('gpu',['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv','-lms','500'])
            spawn('host-metrics',[sys.executable,str(ROOT/'real_robot/sample_compute_host.py')])
        runtime = {}
        if owned:
            import torch
            import transformers
            torch.set_num_threads(4)
            if not torch.cuda.is_available():raise RuntimeError('CUDA required')
            runtime={'python':sys.version,'architecture':platform.machine(),'torch':torch.__version__,
                     'cuda':torch.version.cuda,'transformers':transformers.__version__,
                     'device':torch.cuda.get_device_name(0)}
            manifest['runtime']=runtime
            manifest['model_sha256']={name:digest(a.assets/'models'/name/'model.safetensors') for name in (
                'OmTrackVLA-0.6B','Qwen3-0.6B','dinov3-vits16-pretrain-lvd1689m','siglip-so400m-patch14-384')}
        began=time.perf_counter()
        if a.mode=='direct':
            from real_robot.inference_server import OmTrackInference
            model=OmTrackInference(a.assets,'cuda',31,.1,1,.3,.25,.35,15,3,.5,
                                   planner_rate=1e9,cuda_vision_preprocess=False,mode='model_only')
            torch.cuda.synchronize()
        else:
            if a.launch_server:
                with socket.socket() as probe:probe.bind((a.host,a.port))
                spawn('server',[sys.executable,'-u',str(ROOT/'real_robot/inference_server.py'),
                    '--mode','model_only','--port',str(a.port),'--planner-rate','1000000000','--legacy-vision-preprocess'])
            deadline=time.monotonic()+90
            while True:
                try:
                    connection=socket.create_connection((a.host,a.port),timeout=3,source_address=(a.source_address,0) if a.source_address else None);connection.settimeout(30);break
                except OSError:
                    if time.monotonic()>deadline:raise
                    if a.launch_server and children[-1].poll() is not None:raise RuntimeError('Server exited; inspect server.log')
                    time.sleep(.1)
        manifest['first_observed_startup_s']=time.perf_counter()-began
        request_id=0
        def infer(item, reset):
            nonlocal request_id
            row,rgb,encoded,_=item
            if a.mode=='direct':return model.infer(rgb,PROMPT,reset_target=reset)
            request_id+=1
            send_message(connection,{'request_id':request_id,'frame_id':row['id'],'encoding':'jpeg',
                'image_b64':encoded,'prompt':PROMPT,'reset_target':reset})
            reply=receive_message(connection)
            if reply.get('ok') is not True or reply.get('request_id')!=request_id or reply.get('frame_id')!=row['id']:
                raise RuntimeError('Mismatched/error response: '+str(reply))
            return reply
        for i in range(3):infer(prepared[0],i==0)
        if a.mode=='direct':torch.cuda.reset_peak_memory_stats()
        durations=[]
        for rep in range(a.repetitions):
            previous=None;started=time.perf_counter()
            for item in prepared:
                row=item[0];before=time.perf_counter()
                reply=infer(item,row['clip']!=previous)
                latency=(time.perf_counter()-before)*1000;previous=row['clip']
                trajectory=np.asarray(reply['trajectory'],dtype=float)
                if reply.get('planner_updated') is not True or trajectory.shape!=(8,3) or not np.isfinite(trajectory).all():
                    raise RuntimeError('Cached/nonfinite/wrong-shaped model output')
                rows.append({'id':row['id'],'clip':row['clip'],'repetition':rep,'latency_ms':latency,
                    'server_ms':reply.get('inference_seconds',0)*1000 if a.mode=='socket' else None,
                    'fresh':True,'trajectory':reply['trajectory'],'trajectory_valid':reply['trajectory_valid']})
            durations.append(time.perf_counter()-started)
            print(json.dumps({'mode':a.mode,'repetition':rep,'completed':len(prepared),'seconds':durations[-1]}),flush=True)
        latency=np.asarray([r['latency_ms'] for r in rows])
        manifest.update(status='COMPLETE',measured_calls=len(rows),fresh_updates=len(rows),
            finite_waypoint_outputs=len(rows),latency_ms={f'p{q}':float(np.percentile(latency,q)) for q in (50,95,99)},
            completed_fresh_updates_s=len(rows)/sum(durations),measured_duration_s=sum(durations))
        if a.mode=='socket':manifest['server_latency_ms']={f'p{q}':float(np.percentile([r['server_ms'] for r in rows],q)) for q in (50,95,99)}
        if a.mode=='direct':manifest['peak_cuda_allocated_mib']=torch.cuda.max_memory_allocated()/1024**2
    except BaseException as exc:
        manifest.update(status='CANCELLED' if isinstance(exc,KeyboardInterrupt) else 'FAILED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if connection is not None:connection.close()
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        for stream in streams:stream.close()
        manifest['finished_unix']=time.time();save()
    print(json.dumps(manifest.get('latency_ms')),flush=True)


if __name__=='__main__':main()
