"""Record a bounded RGB-only clip from an existing ROS camera; no publishers."""
import argparse
import json
from pathlib import Path
import time

from real_robot.defaults import CAMERA_TOPIC
from real_robot.files import sha256_bytes
from real_robot.ros_camera import is_jpeg, latest_frame_qos, stamp_ns


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--topic', default=CAMERA_TOPIC)
    p.add_argument('--seconds', type=float, default=16)
    p.add_argument('--hz', type=float, default=2)
    p.add_argument('--scene', required=True, help='Observed clothing, positions and movement; not model labels')
    a = p.parse_args()
    if not 0 < a.seconds <= 120 or not 0 < a.hz <= 10:
        p.error('Use 0 < seconds <= 120 and 0 < hz <= 10')
    if a.output.exists():
        p.error('Use a new output directory')
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CompressedImage
    a.output.mkdir(parents=True)
    rows = []
    last_stamp = None
    next_due = 0.0
    started = time.monotonic()
    rclpy.init()
    node = Node('omtrack_prompt_clip_recorder')

    def image(msg):
        nonlocal last_stamp, next_due
        now = time.monotonic()
        stamp = stamp_ns(msg.header)
        if stamp == last_stamp or now < next_due:
            return
        if not is_jpeg(msg.format):
            raise ValueError('Expected JPEG compressed input; got ' + msg.format)
        payload = bytes(msg.data)
        filename = f'{len(rows):05d}.jpg'
        (a.output / filename).write_bytes(payload)
        rows.append(dict(image=filename, image_sha256=sha256_bytes(payload),
                         source_stamp_ns=stamp, elapsed_s=now - started))
        last_stamp = stamp
        next_due = now + 1 / a.hz

    node.create_subscription(CompressedImage, a.topic, image, latest_frame_qos())
    complete = False
    try:
        print('Recording camera only; no model or robot commands.', flush=True)
        while time.monotonic() - started < a.seconds:
            rclpy.spin_once(node, timeout_sec=.1)
        complete = True
    finally:
        node.destroy_node()
        rclpy.shutdown()
        manifest = dict(schema_version=1, complete=complete, scene=a.scene, topic=a.topic,
                        requested_hz=a.hz, requested_seconds=a.seconds, samples=rows)
        (a.output / 'clip.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if len(rows) < 2:
        raise RuntimeError('Too few camera frames; clip cannot be evaluated')
    print(json.dumps(dict(saved_frames=len(rows), clip=str(a.output.resolve()))), flush=True)


if __name__ == '__main__':
    main()
