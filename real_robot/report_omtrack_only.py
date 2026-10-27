"""Recalculate the focused OmTrackVLA report from saved measurements only."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path)
    p.add_argument('--important-only',action='store_true',help='Report just cases A, B and C; omit the optional loopback check')
    a=p.parse_args()
    names={'ridgeback-direct':'A: Ridgeback direct','thor-direct':'B: Thor direct',
           'ridgeback-socket':'Ridgeback local socket check','ridgeback-to-thor':'C: Ridgeback → Thor'}
    if a.important_only:names.pop('ridgeback-socket')
    data={};raw={}
    for folder,label in names.items():
        path=a.results/folder
        if not (path/'report.json').exists():
            data[folder]={'label':label,'status':'BLOCKED','reason':'Thor SSH at 192.168.131.51 timed out; no measurement collected'}
            continue
        report=json.loads((path/'report.json').read_text())
        if report['status']!='COMPLETE':
            data[folder]={'label':label,**report};continue
        rows=json.loads((path/'outputs.json').read_text())
        assert len(rows)==report['sample_count']*report['repetitions']==report['fresh_updates']
        assert all(r['fresh'] and np.asarray(r['trajectory']).shape==(8,3) and np.isfinite(r['trajectory']).all() for r in rows)
        latency=[r['latency_ms'] for r in rows]
        stats={f'p{q}':float(np.percentile(latency,q)) for q in (50,95,99)}
        assert all(abs(stats[k]-report['latency_ms'][k])<1e-9 for k in stats)
        data[folder]={'label':label,**report,'latency_ms':stats}
        raw[folder]={(r['id'],r['repetition']):r for r in rows}
        telemetry=path/'gpu.log'
        if telemetry.exists():
            samples=[]
            for line in telemetry.read_text().splitlines()[1:]:
                try:samples.append([float(v.strip().split()[0]) for v in line.split(',')[1:]])
                except ValueError:continue
            if samples:
                arr=np.asarray(samples)
                data[folder]['gpu_sampled']={'peak_used_mib':float(arr[:,1].max()),'peak_temperature_c':float(arr[:,2].max()),
                    'mean_gpu_utilization_percent':float(arr[:,0].mean()),'scope':'whole owned trial including model hash/load/warmup; not steady-state average'}
    if 'thor-direct' in raw and 'ridgeback-to-thor' in raw:
        # The remote case launches the same isolated Thor mirror and weights.
        # Keep the collected baseline manifest as provenance for that server.
        data['ridgeback-to-thor']['server_runtime_reference']='thor-direct/report.json'
        data['ridgeback-to-thor']['server_runtime']=data['thor-direct']['runtime']
    parity={}
    for target in ('ridgeback-socket','thor-direct','ridgeback-to-thor'):
        if target not in raw or 'ridgeback-direct' not in raw:continue
        baseline=data['ridgeback-direct'];other=data[target]
        assert baseline['jpeg_sha256']==other['jpeg_sha256']
        assert baseline['prompt']==other['prompt'] and baseline['history']==other['history']
        for source in ('real_robot/inference_server.py','cache_gridpool.py','open_trackvla_hf/modeling_open_trackvla.py'):
            if source in other['source_sha256']:assert baseline['source_sha256'][source]==other['source_sha256'][source]
        if 'model_sha256' in other:assert baseline['model_sha256']==other['model_sha256']
        assert set(raw[target])==set(raw['ridgeback-direct'])
        xy=[];yaw=[]
        for key,row in raw[target].items():
            left=np.asarray(raw['ridgeback-direct'][key]['trajectory']);right=np.asarray(row['trajectory'])
            xy.append(float(np.linalg.norm(left[:,:2]-right[:,:2],axis=1).max()))
            diff=left[:,2]-right[:,2];yaw.append(float(np.abs(np.arctan2(np.sin(diff),np.cos(diff))).max()))
        max_xy=max(xy);max_yaw=float(np.degrees(max(yaw)))
        parity[target]={'max_waypoint_xy_difference_m':max_xy,'max_waypoint_yaw_difference_deg':max_yaw,
                        'within_5cm_3deg':max_xy<=.05 and max_yaw<=3}
    result={'scope':'OmTrackVLA only','cases':data,'parity':parity,'physical_following_qualified':False,
            'campaign_complete':all(r['status']=='COMPLETE' for r in data.values())}
    (a.results/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# OmTrackVLA-only results — 2026-10-27','',
        'Only RGB → DINOv3/SiGLIP → history/pooling → OmTrackVLA/Qwen → waypoints was tested. No DyNAMO detection, segmentation or distance estimation was launched. No ROS, Nav2 or motion was launched.',
        '', '40 frozen RGB frames across four clips, two repetitions: 80 measured calls per completed case. Same JPEG bytes, prompt, history and reset policy. Three warmup calls excluded; cached outputs rejected. These are compute-only outputs and do not validate target identity or following behavior.', '',
        '| Case | Status | Median ms | p95 ms | p99 ms | Fresh updates/s | Finite waypoint outputs |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in data.values():
        if row['status']=='COMPLETE':
            s=row['latency_ms']
            lines.append(f"| {row['label']} | COMPLETE | {s['p50']:.2f} | {s['p95']:.2f} | {s['p99']:.2f} | {row['completed_fresh_updates_s']:.2f} | {row['finite_waypoint_outputs']}/{row['measured_calls']} |")
        else:lines.append(f"| {row['label']} | {row['status']} | — | — | — | — | — |")
    lines+=['','Direct timing includes vision encoding, history, waypoint inference and output conversion. Socket timing additionally includes JSON/base64 request serialization, loopback/network transport, JPEG decode and response. JPEG encoding is prepared once outside both measured paths. Throughput is serial saturated compute throughput, not a configured live planner frequency.', '',
            '## Startup and memory','', '| Case | First observed startup s | Sampled GPU peak MiB |', '|---|---:|---:|']
    for row in data.values():
        if row['status']=='COMPLETE':
            peak=row.get('gpu_sampled',{}).get('peak_used_mib')
            peak_text=f'{peak:.0f}' if peak is not None else 'unavailable'
            lines.append(f"| {row['label']} | {row['first_observed_startup_s']:.2f} | {peak_text} |")
    lines+=['','Direct startup means model initialization; owned socket startup means child Python/model initialization through socket readiness. Filesystem caches were not cleared. GPU peaks include startup/warmup and normal desktop use; the allocator peak is separately recorded for direct inference. CPU/RAM raw telemetry includes normal system services.', '',
            '## Output consistency','',json.dumps(parity,indent=2),'',
            '## Thor connectivity','',
            'Earlier access was recovered with source address 192.168.123.14. The post-repair campaign uses the normal wired 192.168.131.1 source; per-case manifests record the actual source address. Before repair, Thor had duplicate 192.168.131.51 addresses and selected a failing return interface. A targeted repair and rollback are saved in network-repair/. The user applied the permanent correction; ordinary SSH, the enP2p1s0 return route and disabled duplicate-profile autoconnect were verified on 2026-10-27. No previous combined-system timing is substituted. Splitting the neural model across two GPUs is not implemented in this repository.', '',
            'Ridgeback uses Python 3.9 / PyTorch 2.8 CUDA 12.8; Thor uses Python 3.12 / PyTorch 2.11 CUDA 13.0. Transformers 4.57.6 and model hashes match. This is a comparison of the tested runtimes, not isolated hardware capability. Thor nvidia-smi memory/temperature/power fields are unavailable; direct CUDA allocator peak and cross-host tegrastats are retained separately. Socket startup on an already launched remote server includes only the remaining wait/connect time and is not comparable to direct model load.\n\n35 existing fast unit/contract tests passed. Report statistics were recalculated from raw samples. `cleanup.json` records the post-run local GPU/port inspection. Each case stores source/model/input hashes, runtime, raw waypoint outputs and telemetry.']
    (a.results/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:15]))


if __name__=='__main__':main()
