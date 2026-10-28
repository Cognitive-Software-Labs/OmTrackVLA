"""Bounded stationary ROS RGB/model recording. Never publishes ROS topics."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import socket
import time

from real_robot.protocol import receive_message, send_message


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prompt',required=True)
    p.add_argument('--seconds',type=float,default=25)
    p.add_argument('--topic',default='/r100_0160/sensors/camera_0/color/image/compressed')
    a=p.parse_args()
    if not 0<a.seconds<=120 or not a.prompt.strip():p.error('Require 0 < seconds <= 120 and a prompt')
    if a.output.exists():p.error('Use a fresh output directory')
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,HistoryPolicy
    from sensor_msgs.msg import CompressedImage
    a.output.mkdir(parents=True)
    latest=None
    def camera(msg):
        nonlocal latest
        latest=(msg.header.stamp.sec*1000000000+msg.header.stamp.nanosec,time.monotonic(),bytes(msg.data),msg.format)
    rclpy.init();node=Node('omtrack_bounded_live_test')
    sub=node.create_subscription(CompressedImage,a.topic,camera,QoSProfile(history=HistoryPolicy.KEEP_LAST,depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
    rows=[];complete=False;conn=None;started=time.monotonic();previous=None;next_due=0
    manifest=dict(schema_version=1,prompt=a.prompt,topic=a.topic,requested_seconds=a.seconds,mode='model_only',publishes_ros_topics=False,history_reset='First request only; live history retained',input_policy='Latest available camera frame, at least 0.55s after each response',samples=rows)
    try:
        conn=socket.create_connection(('127.0.0.1',18781),timeout=3);conn.settimeout(15)
        print('Recording live camera inputs and OmTrackVLA predictions now.',flush=True)
        with (a.output/'predictions.jsonl').open('x') as log:
            while time.monotonic()-started<a.seconds:
                rclpy.spin_once(node,timeout_sec=.03)
                if latest is None or latest[0]==previous or time.monotonic()<next_due:continue
                stamp,arrival,jpeg,fmt=latest
                if time.monotonic()-arrival>1:continue
                if 'jpeg' not in fmt.lower() and 'jpg' not in fmt.lower():raise ValueError('Expected JPEG camera input')
                index=len(rows);name=f'{index:05d}.jpg';(a.output/name).write_bytes(jpeg)
                before=time.perf_counter()
                send_message(conn,dict(request_id=stamp,encoding='jpeg',image_b64=base64.b64encode(jpeg).decode(),prompt=a.prompt,reset_target=index==0))
                reply=receive_message(conn)
                if reply.get('ok') is not True or reply.get('request_id')!=stamp or reply.get('mode')!='model_only':raise ValueError('Unexpected response: '+str(reply))
                sample=dict(image=name,image_sha256=hashlib.sha256(jpeg).hexdigest(),
                            source_stamp_ns=stamp,elapsed_s=time.monotonic()-started)
                rows.append(sample)
                prediction=dict(**sample,roundtrip_ms=(time.perf_counter()-before)*1000,
                                prompt=a.prompt,**reply)
                log.write(json.dumps(prediction,allow_nan=False)+'\n');log.flush()
                previous=stamp;next_due=time.monotonic()+.55
                if index%10==0:print(json.dumps(dict(saved_frames=len(rows),planner_updated=reply.get('planner_updated'))),flush=True)
        complete=True
    finally:
        if conn is not None:conn.close()
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
        manifest.update(complete=complete,actual_seconds=time.monotonic()-started)
        (a.output/'clip.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if len(rows)<2:raise RuntimeError('Insufficient frames')
    print(json.dumps(dict(complete=complete,frames=len(rows),output=str(a.output.resolve()))),flush=True)


if __name__=='__main__':main()
