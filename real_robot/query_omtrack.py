"""Send one RGB image to the local OmTrackVLA model; save the waypoint reply."""
import argparse
import json
from pathlib import Path
import time

from real_robot.defaults import DEFAULT_PROMPT, MODEL_PORT
from real_robot.model_client import ModelClient, encode_jpeg, require_fresh_waypoints


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('image', type=Path)
    p.add_argument('--port', type=int, default=MODEL_PORT)
    p.add_argument('--prompt', default=DEFAULT_PROMPT)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    import cv2
    bgr = cv2.imread(str(a.image))
    if bgr is None:
        raise RuntimeError('Could not decode ' + str(a.image))
    jpeg = encode_jpeg(bgr)
    request_id = time.time_ns()
    with ModelClient(a.port, connect_timeout=3, timeout=30) as client:
        reply = client.infer(jpeg, a.prompt, request_id, reset_target=True, frame_id=str(a.image))
    require_fresh_waypoints(reply)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(reply, indent=2) + '\n')
    print(json.dumps({'fresh_waypoints': len(reply['trajectory']),
                      'inference_ms': reply['inference_seconds'] * 1000,
                      'saved': str(a.output.resolve())}))


if __name__ == '__main__':
    main()
