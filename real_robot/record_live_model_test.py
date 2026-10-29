"""Bounded stationary ROS RGB/model recording. Never publishes ROS topics."""
import argparse
import json
from pathlib import Path
import time

from real_robot.defaults import CAMERA_TOPIC, MODEL_PORT
from real_robot.files import sha256_bytes
from real_robot.model_client import ModelClient
from real_robot.ros_camera import is_jpeg, latest_frame_qos, stamp_ns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--seconds', type=float, default=25)
    p.add_argument('--topic', default=CAMERA_TOPIC)
    p.add_argument('--port', type=int, default=MODEL_PORT)
    a = p.parse_args()
    if not 0 < a.seconds <= 120 or not a.prompt.strip():
        p.error('Require 0 < seconds <= 120 and a prompt')
    if a.output.exists():
        p.error('Use a fresh output directory')
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    a.output.mkdir(parents=True)
    latest = None

    def camera(msg):
        nonlocal latest
        latest = (stamp_ns(msg.header), time.monotonic(), bytes(msg.data), msg.format)

    rclpy.init()
    node = Node('omtrack_bounded_live_test')
    node.create_subscription(CompressedImage, a.topic, camera, latest_frame_qos())
    rows = []
    complete = False
    client = None
    started = time.monotonic()
    previous = None
    next_due = 0
    manifest = dict(schema_version=1, prompt=a.prompt, topic=a.topic, requested_seconds=a.seconds,
                    mode='model_only', publishes_ros_topics=False,
                    history_reset='First request only; live history retained',
                    input_policy='Latest available camera frame, at least 0.55s after each response',
                    samples=rows)
    try:
        client = ModelClient(a.port, connect_timeout=3, timeout=15)
        print('Recording live camera inputs and OmTrackVLA predictions now.', flush=True)
        with (a.output / 'predictions.jsonl').open('x') as log:
            while time.monotonic() - started < a.seconds:
                rclpy.spin_once(node, timeout_sec=.03)
                if latest is None or latest[0] == previous or time.monotonic() < next_due:
                    continue
                stamp, arrival, jpeg, fmt = latest
                if time.monotonic() - arrival > 1:
                    continue
                if not is_jpeg(fmt):
                    raise ValueError('Expected JPEG camera input')
                index = len(rows)
                name = f'{index:05d}.jpg'
                (a.output / name).write_bytes(jpeg)
                before = time.perf_counter()
                reply = client.infer(jpeg, a.prompt, stamp, reset_target=index == 0, mode='model_only')
                sample = dict(image=name, image_sha256=sha256_bytes(jpeg),
                              source_stamp_ns=stamp, elapsed_s=time.monotonic() - started)
                rows.append(sample)
                prediction = dict(**sample, roundtrip_ms=(time.perf_counter() - before) * 1000,
                                  prompt=a.prompt, **reply)
                log.write(json.dumps(prediction, allow_nan=False) + '\n')
                log.flush()
                previous = stamp
                next_due = time.monotonic() + .55
                if index % 10 == 0:
                    print(json.dumps(dict(saved_frames=len(rows),
                                          planner_updated=reply.get('planner_updated'))), flush=True)
        complete = True
    finally:
        if client is not None:
            client.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        manifest.update(complete=complete, actual_seconds=time.monotonic() - started)
        (a.output / 'clip.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if len(rows) < 2:
        raise RuntimeError('Insufficient frames')
    print(json.dumps(dict(complete=complete, frames=len(rows), output=str(a.output.resolve()))), flush=True)


if __name__ == '__main__':
    main()
