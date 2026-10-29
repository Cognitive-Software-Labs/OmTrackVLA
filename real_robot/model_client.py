"""Client side of the inference_server socket: JPEG frames in, waypoint replies out."""
import base64
import math
import socket

from real_robot.defaults import MODEL_PORT
from real_robot.protocol import receive_message, send_message


def encode_jpeg(bgr, quality=95):
    import cv2
    ok, jpg = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    return jpg.tobytes()


def load_image_for_model(path, quality=95):
    """Read an image file and return (jpeg_bytes, rgb) where rgb is the decoded JPEG.

    Decoding the JPEG again gives direct (in-process) inference exactly the pixels a
    socket client would send.
    """
    import cv2
    import numpy as np
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f'Cannot decode {path}')
    jpeg = encode_jpeg(bgr, quality)
    decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    return jpeg, cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)


def has_finite_waypoints(trajectory, count=8):
    """True for count x 3 finite numbers; accepts JSON lists or in-process arrays."""
    def finite_number(v):
        # Strings and bools would pass float() but break the browser plot.
        return not isinstance(v, (str, bytes, bool)) and math.isfinite(float(v))
    try:
        return (len(trajectory) == count
                and all(not isinstance(p, (str, bytes)) and len(p) == 3
                        and all(finite_number(v) for v in p) for p in trajectory))
    except (TypeError, ValueError):
        return False


def require_fresh_waypoints(reply):
    """Raise unless the reply holds a newly computed, finite 8x3 trajectory."""
    if reply.get('planner_updated') is not True or not has_finite_waypoints(reply.get('trajectory')):
        raise RuntimeError(f'No fresh finite waypoint response: {reply}')


class ModelClient:
    """One persistent connection to inference_server; validates every reply."""

    def __init__(self, port=MODEL_PORT, host='127.0.0.1', connect_timeout=3, timeout=30,
                 source_address=None):
        self.conn = socket.create_connection(
            (host, port), timeout=connect_timeout,
            source_address=(source_address, 0) if source_address else None)
        self.conn.settimeout(timeout)

    def infer(self, jpeg, prompt, request_id, reset_target, frame_id=None, mode=None):
        """jpeg is JPEG bytes, or the same already base64-encoded as str so timed
        loops and displays that need the text anyway can encode it only once."""
        image_b64 = jpeg if isinstance(jpeg, str) else base64.b64encode(jpeg).decode()
        request = {'request_id': request_id, 'encoding': 'jpeg',
                   'image_b64': image_b64,
                   'prompt': prompt, 'reset_target': reset_target}
        if frame_id is not None:
            request['frame_id'] = frame_id
        send_message(self.conn, request)
        reply = receive_message(self.conn)
        if reply.get('ok') is not True or reply.get('request_id') != request_id:
            raise ValueError(f'Error or mismatched model response: {reply}')
        if frame_id is not None and reply.get('frame_id') != frame_id:
            raise ValueError(f'Reply is for another frame: {reply}')
        if mode is not None and reply.get('mode') != mode:
            raise ValueError(f'Expected {mode} server, got: {reply}')
        return reply

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
