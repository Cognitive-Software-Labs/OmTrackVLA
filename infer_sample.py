"""Run OmTrackVLA on an image or bundled video frame; no simulator or robot output."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',default='sim_data/sample/seed_100/17DRP5sb8fy/4.mp4')
    p.add_argument('--image',type=Path)
    p.add_argument('--assets',type=Path,default=Path(os.environ.get('OMTRACK_ASSETS',Path(__file__).resolve().parent)))
    p.add_argument('--device',default='cuda')
    p.add_argument('--instruction',default='Follow the target person without collision.')
    p.add_argument('--output',type=Path,default=Path('outputs/sample-inference'))
    a=p.parse_args()
    import cv2
    import numpy as np
    import torch
    from real_robot.inference_server import OmTrackInference
    torch.set_num_threads(4)
    if a.image:
        frame=cv2.imread(str(a.image))
    else:
        video=cv2.VideoCapture(a.video);ok,frame=video.read();video.release()
        if not ok:frame=None
    if frame is None:raise RuntimeError('Cannot read the requested image/video')
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    model=OmTrackInference(a.assets.resolve(),a.device,31,.1,1,.35,.25,.25,8,3,.72,
                            mode='model_only',cuda_vision_preprocess=False)
    result=model.infer(rgb,a.instruction,reset_target=True)
    points=np.asarray(result['trajectory'],dtype=float)
    if result['planner_updated'] is not True or points.shape!=(8,3) or not np.isfinite(points).all():
        raise RuntimeError('Expected a fresh finite 8×3 trajectory')
    a.output.mkdir(parents=True,exist_ok=True)
    result.update(instruction=a.instruction,device=a.device,source=str(a.image or a.video),
                  qualification='Numerical model prediction only; no target identity or motion qualification')
    (a.output/'prediction.json').write_text(json.dumps(result,indent=2)+'\n')
    cv2.imwrite(str(a.output/'input.jpg'),frame)
    print(json.dumps({'output':str(a.output.resolve()),'waypoints':points.tolist()}),flush=True)


if __name__=='__main__':main()
