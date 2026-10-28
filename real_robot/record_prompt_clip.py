"""Record a bounded RGB-only clip from an existing ROS camera; no publishers."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--topic', default='/r100_0160/sensors/camera_0/color/image/compressed')
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
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
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
        stamp = msg.header.stamp.sec * 1000000000 + msg.header.stamp.nanosec
        if stamp == last_stamp or now < next_due:
            return
        if 'jpeg' not in msg.format.lower() and 'jpg' not in msg.format.lower():
            raise ValueError('Expected JPEG compressed input; got ' + msg.format)
        payload = bytes(msg.data)
        filename = f'{len(rows):05d}.jpg'
        (a.output / filename).write_bytes(payload)
        rows.append(dict(image=filename, image_sha256=hashlib.sha256(payload).hexdigest(),
                         source_stamp_ns=stamp, elapsed_s=now-started))
        last_stamp = stamp
        next_due = now + 1/a.hz

    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.BEST_EFFORT)
    node.create_subscription(CompressedImage, a.topic, image, qos)
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
        (a.output / 'clip.json').write_text(json.dumps(manifest, indent=2)+'\n')
    if len(rows) < 2:
        raise RuntimeError('Too few camera frames; clip cannot be evaluated')
    print(json.dumps(dict(saved_frames=len(rows), clip=str(a.output.resolve()))), flush=True)


if __name__ == '__main__':
    main()
