"""Recalculate native OmTrackVLA case results, keeping skipped VLA work explicit."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np


def stats(rows):
    if not rows:return None
    return {f'p{q}':float(np.percentile(rows,q)) for q in (50,95,99)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('native',type=Path)
    p.add_argument('--baseline',type=Path,required=True)
    a=p.parse_args();result={'native':{},'model_only':{},'physical_motion_qualified':False}
    for host in ('ridgeback','thor'):
        result['native'][host]={}
        reference=json.loads((a.baseline/(host+'-direct')/'report.json').read_text())
        result['model_only'][host]=reference
        for mode in ('perception','full'):
            root=a.native/(host+'-'+mode)
            report=json.loads((root/'report.json').read_text())
            rows=json.loads((root/'outputs.json').read_text())
            assert report['status']=='COMPLETE' and len(rows)==80
            assert report['jpeg_sha256']==reference['jpeg_sha256']
            assert report['source_sha256']['real_robot/inference_server.py']==reference['source_sha256']['real_robot/inference_server.py']
            recomputed=stats([r['latency_ms'] for r in rows])
            assert all(abs(report['latency_ms'][k]-v)<1e-9 for k,v in recomputed.items())
            assert report['fresh_vla_updates']==sum(r['fresh_vla'] for r in rows)
            assert report['locked_frames']==sum(r['target_valid'] for r in rows)
            for r in rows:
                if r['fresh_vla']:assert np.asarray(r['trajectory']).shape==(8,3) and np.isfinite(r['trajectory']).all()
            report['per_clip']={clip:{'frames':sum(r['clip']==clip for r in rows),
                'locked':sum(r['clip']==clip and r['target_valid'] for r in rows),
                'fresh_vla':sum(r['clip']==clip and r['fresh_vla'] for r in rows),
                'valid_fresh_waypoints':sum(r['clip']==clip and r['fresh_vla'] and r['trajectory_valid'] for r in rows),
                'states':dict(Counter(r['target_state'] for r in rows if r['clip']==clip))} for clip in sorted(set(r['clip'] for r in rows))}
            result['native'][host][mode]=report
    for mode in ('perception','full'):
        left,right=[result['native'][h][mode] for h in ('ridgeback','thor')]
        assert left['model_sha256']==right['model_sha256']
        assert left['settings']==right['settings']
        assert left['source_sha256']==right['source_sha256']
    cross=json.loads((a.baseline/'ridgeback-to-thor/report.json').read_text())
    result['remote_model_only']=cross
    def fmt(value,key='p50'):return f'{value[key]:.2f}' if value else '—'
    lines=['# OmTrackVLA native cases — 2026-10-27','',
        'Three software cases are distinguished: VLA-only (`model_only`), native target-perception (`perception`), and native target-perception plus gated VLA (`full`). `planner` uses the same VLA model with an externally supplied target result; it is not another learned model. No DyNAMO code, mask estimator, distance estimator, ROS, Nav2 or physical motion was launched.', '',
        'Each host/case uses the same 40 JPEG-decoded RGB frames and prompt, twice: 80 measured calls. Reset per clip and repetition. Native modes use the existing server defaults: Grounding DINO .35/.25, YOLO .25, grounding interval 8, acquisition 3, appearance .72. VLA uses history 31 and legacy preprocessing. Native frames are timed through CUDA completion; loading/warmup are excluded. Inputs are processed back-to-back, not replayed at camera rate.', '',
        '## Main compute comparison','',
        '| Case / timed work | Ridgeback median ms | Ridgeback p95 ms | Thor median ms | Thor p95 ms |',
        '|---|---:|---:|---:|---:|']
    comparisons=[('VLA-only: every call invokes the model',*[result['model_only'][h]['latency_ms'] for h in ('ridgeback','thor')]),
        ('Native perception: all frames',*[result['native'][h]['perception']['latency_ms'] for h in ('ridgeback','thor')]),
        ('Native perception: ordinary frames without grounding',*[result['native'][h]['perception']['ordinary_tracking_frame_latency_ms'] for h in ('ridgeback','thor')]),
        ('Native perception: frames invoking Grounding DINO',*[result['native'][h]['perception']['grounding_frame_latency_ms'] for h in ('ridgeback','thor')]),
        ('Full pipeline: frames where VLA actually ran',*[result['native'][h]['full']['fresh_vla_frame_latency_ms'] for h in ('ridgeback','thor')]),
        ('Full pipeline: all frames, including skipped VLA',*[result['native'][h]['full']['latency_ms'] for h in ('ridgeback','thor')])]
    for label,left,right in comparisons:lines.append(f'| {label} | {fmt(left)} | {fmt(left,"p95")} | {fmt(right)} | {fmt(right,"p95")} |')
    lines+=['','The mixed all-frame full-pipeline median is **not** the cost of full VLA inference: most frames never acquire a target and skip it. Grounding runs periodically rather than on every frame. Model-only and full modes have different admitted histories because full mode encodes only locked frames.', '',
            '## Work completed and target outcomes','',
            '| Host / native mode | Frames | Locked frames | Grounding forwards | Fresh VLA calls | Valid fresh waypoint outputs | CUDA allocator peak MiB |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for host in ('ridgeback','thor'):
        for mode in ('perception','full'):
            r=result['native'][host][mode]
            lines.append(f"| {host} / {mode} | {r['measured_calls']} | {r['locked_frames']} | {r['grounding_forwards']} | {r['fresh_vla_updates']} | {r['valid_fresh_waypoint_outputs']} | {r['peak_cuda_allocated_mib']:.0f} |")
    lines+=['','Perception-only does not include a waypoint model; zero VLA calls is expected. A valid fresh waypoint means it passed this checkout\'s finite/directional wrapper checks; it does not establish collision-free navigation or physical following.', '',
            '## Scene outcomes in full mode','',
            '| Scene, two repetitions | Ridgeback locked / fresh VLA / valid paths | Thor locked / fresh VLA / valid paths |',
            '|---|---|---|']
    for clip in result['native']['ridgeback']['full']['per_clip']:
        values=[result['native'][h]['full']['per_clip'][clip] for h in ('ridgeback','thor')]
        lines.append(f"| {clip}: 20 frames | "+' | '.join(f"{r['locked']} / {r['fresh_vla']} / {r['valid_fresh_waypoints']}" for r in values)+' |')
    lines+=['','Sparse clips contain 10 frames each sampled roughly three seconds apart. Movement includes cropped views; re-entry is partial exit/return. These measurements expose target acquisition limits on these samples and are not a continuous tracking qualification.', '',
        '## Two-host model serving already measured','',
        f"Ridgeback → Thor model-only service: median {cross['latency_ms']['p50']:.2f} ms, p95 {cross['latency_ms']['p95']:.2f} ms, {cross['finite_waypoint_outputs']}/{cross['measured_calls']} fresh finite outputs. This excludes native target-perception and must not be called a distributed native-full-pipeline measurement.", '',
        '## Interpretation and evidence','',
        'Compare the tested implementations, not hardware peak capability: Ridgeback uses Python 3.9 / PyTorch 2.8 CUDA 12.8; Thor uses Python 3.12 / PyTorch 2.11 CUDA 13.0. Model/input hashes match. Native target-perception code/settings and Ultralytics 8.4.159 match. Thor dependency wheels are installed only under the isolated task directory. Grounding DINO uses the existing FP16 autocast path; other native defaults are unchanged.', '',
        'GPU/RAM/CPU raw telemetry and exact runtime/source/model/input hashes are stored per case. GPU monitoring includes startup/warmup. Thor uses unified memory; missing nvidia-smi memory fields are not zero. Reported CUDA memory is allocator peak, not total board memory. VLA-only and remote timings are reused from the immediately preceding post-network-repair campaign and were not rerun unnecessarily.', '',
        'Raw rows and the complete p50/p95/p99 statistics are in `results.json` and each case folder. `cleanup.json` records that all owned benchmark workers were stopped. No physical output was enabled.']
    (a.native/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (a.native/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines[:18]))


if __name__=='__main__':main()
