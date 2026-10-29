"""Helpers for subscribing to the existing compressed ROS camera; never publishes."""


def stamp_ns(header):
    return header.stamp.sec * 1_000_000_000 + header.stamp.nanosec


def is_jpeg(image_format):
    fmt = image_format.lower()
    return 'jpeg' in fmt or 'jpg' in fmt


def latest_frame_qos():
    """Keep only the newest frame and drop late ones instead of queueing them."""
    from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
    return QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                      reliability=ReliabilityPolicy.BEST_EFFORT)
