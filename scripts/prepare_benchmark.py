"""Build portable fixed RGB samples from a video, with original-image checksums."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('data/benchmark'))
    p.add_argument('--frames',type=int,default=40)
    a=p.parse_args()
    if a.frames<1:p.error('--frames must be positive')
    if a.output.exists():p.error('Use a fresh output directory')
    import cv2
    import numpy as np
    cap=cv2.VideoCapture(str(a.video))
    total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total<1:raise RuntimeError('No video frames found')
    indices=set(np.linspace(0,total-1,min(a.frames,total),dtype=int).tolist())
    folder=a.output/'video';folder.mkdir(parents=True)
    rows=[];index=0
    try:
        while True:
            ok,bgr=cap.read()
            if not ok:break
            if index in indices:
                path=folder/f'{index:08d}.png'
                if not cv2.imwrite(str(path),bgr):raise RuntimeError('Cannot write frame')
                rows.append({'id':f'video:{index}','clip':'video','image':path.name,
                             'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            index+=1
    finally:cap.release()
    if len(rows)!=len(indices):raise RuntimeError('Video decode stopped before all selected frames')
    (a.output/'samples.json').write_text(json.dumps({'source_video':a.video.name,'samples':rows},indent=2)+'\n')
    print(f'Saved {len(rows)} RGB frames under {a.output}')


if __name__=='__main__':main()
