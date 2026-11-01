import socket
import sys
import threading
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from methods.omtrack_vla.ridgeback.model_client import ModelClient, has_finite_waypoints, require_fresh_waypoints
from robot.protocol import receive_message, send_message

POINTS = [[0.1 * i, 0.0, 0.0] for i in range(8)]


def serve_once(reply_for):
    """Fake inference_server: answer one request on a free loopback port."""
    listener = socket.socket()
    listener.bind(('127.0.0.1', 0))
    listener.listen(1)

    def run():
        conn, _ = listener.accept()
        with conn:
            send_message(conn, reply_for(receive_message(conn)))
        listener.close()

    threading.Thread(target=run, daemon=True).start()
    return listener.getsockname()[1]


class WaypointTests(unittest.TestCase):
    def test_shape_and_finiteness(self):
        self.assertTrue(has_finite_waypoints(POINTS))
        self.assertFalse(has_finite_waypoints(POINTS[:7]))
        self.assertFalse(has_finite_waypoints(POINTS[:7] + [[0, float('nan'), 0]]))
        self.assertFalse(has_finite_waypoints(None))
        self.assertFalse(has_finite_waypoints([['0', '0', '0']] * 8))
        self.assertFalse(has_finite_waypoints([[True, 0, 0]] * 8))

    def test_cached_reply_is_not_fresh(self):
        require_fresh_waypoints({'planner_updated': True, 'trajectory': POINTS})
        with self.assertRaises(RuntimeError):
            require_fresh_waypoints({'planner_updated': False, 'trajectory': POINTS})


class ModelClientTests(unittest.TestCase):
    def test_round_trip(self):
        port = serve_once(lambda req: {'ok': True, 'request_id': req['request_id'], 'frame_id': req['frame_id'],
                                       'mode': 'model_only', 'prompt_seen': req['prompt'],
                                       'reset_seen': req['reset_target'], 'trajectory': POINTS})
        with ModelClient(port) as client:
            reply = client.infer(b'\xff\xd8jpeg', 'Follow me.', 7, True, frame_id='f7', mode='model_only')
        self.assertEqual((reply['prompt_seen'], reply['reset_seen']), ('Follow me.', True))

    def test_pre_encoded_image_is_sent_unchanged(self):
        port = serve_once(lambda req: {'ok': True, 'request_id': req['request_id'], 'seen': req['image_b64']})
        with ModelClient(port) as client:
            self.assertEqual(client.infer('anBlZw==', 'Follow me.', 1, False)['seen'], 'anBlZw==')

    def test_mismatched_request_id_is_rejected(self):
        port = serve_once(lambda req: {'ok': True, 'request_id': req['request_id'] + 1})
        with ModelClient(port) as client, self.assertRaises(ValueError):
            client.infer(b'jpeg', 'Follow me.', 1, False)

    def test_wrong_server_mode_is_rejected(self):
        port = serve_once(lambda req: {'ok': True, 'request_id': req['request_id'], 'mode': 'full'})
        with ModelClient(port) as client, self.assertRaises(ValueError):
            client.infer(b'jpeg', 'Follow me.', 1, False, mode='model_only')




if __name__ == '__main__':
    unittest.main()
