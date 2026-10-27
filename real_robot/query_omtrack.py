"""Send one RGB image to the local OmTrackVLA model; save the waypoint reply."""
import argparse
import base64
import json
from pathlib import Path
import socket
import time

from real_robot.protocol import receive_message, send_message


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('image',type=Path)
    p.add_argument('--port',type=int,default=18781)
    p.add_argument('--prompt',default='Follow the person wearing a black jacket or black T-shirt.')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import cv2
    import numpy as np
    bgr=cv2.imread(str(a.image))
    if bgr is None:raise RuntimeError('Could not decode '+str(a.image))
    ok,jpg=cv2.imencode('.jpg',bgr,[cv2.IMWRITE_JPEG_QUALITY,95])
    if not ok:raise RuntimeError('JPEG encoding failed')
    request_id=time.time_ns()
    with socket.create_connection(('127.0.0.1',a.port),timeout=3) as conn:
        conn.settimeout(30)
        send_message(conn,{'request_id':request_id,'frame_id':str(a.image),'encoding':'jpeg',
            'image_b64':base64.b64encode(jpg.tobytes()).decode(),'prompt':a.prompt,'reset_target':True})
        reply=receive_message(conn)
    points=np.asarray(reply.get('trajectory'),dtype=float)
    if reply.get('ok') is not True or reply.get('request_id')!=request_id or reply.get('planner_updated') is not True or points.shape!=(8,3) or not np.isfinite(points).all():
        raise RuntimeError('No fresh finite waypoint response: '+str(reply))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(reply,indent=2)+'\n')
    print(json.dumps({'fresh_waypoints':len(points),'inference_ms':reply['inference_seconds']*1000,
                      'saved':str(a.output.resolve())}))


if __name__=='__main__':main()
