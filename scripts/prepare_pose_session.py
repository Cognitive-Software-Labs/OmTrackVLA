"""Convert a reviewed timestamped RGB/pose session into explicit waypoint windows."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from real_robot.waypoint_data import prepare_pose_session


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--session', type=Path, required=True, help='Session manifest inside one recording directory')
    p.add_argument('--output', type=Path, required=True, help='New output JSONL path; never overwrite')
    p.add_argument('--prediction-offsets-json', type=Path, required=True, help='JSON array of exactly eight seconds offsets')
    p.add_argument('--max-pose-gap-s', type=float, required=True, help='Reviewed maximum reference-pose interpolation gap')
    p.add_argument('--max-history-gap-s', type=float, required=True, help='Reviewed maximum adjacent RGB history gap')
    a = p.parse_args()
    audit = a.output.with_suffix(a.output.suffix+'.audit.json')
    if a.output.exists() or audit.exists():
        p.error('Output or audit file exists; choose a fresh output path')
    manifest = json.loads(a.session.read_text())
    manifest['root'] = str(a.session.resolve().parent)
    offsets = json.loads(a.prediction_offsets_json.read_text())
    prepared = prepare_pose_session(manifest, prediction_offsets_s=offsets,
                                    max_pose_gap_s=a.max_pose_gap_s,
                                    max_history_gap_s=a.max_history_gap_s)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as stream:
        for example in prepared.pop('examples'):
            stream.write(json.dumps(example, allow_nan=False)+'\n')
    with audit.open('x') as stream:
        stream.write(json.dumps(prepared, indent=2)+'\n')
    print(json.dumps(dict(accepted=prepared['accepted_windows'], rejected=prepared['rejected_windows'],
                          jsonl=str(a.output.resolve()), audit=str(audit.resolve()))))


if __name__ == '__main__':
    main()
