import json
import socket
import struct
import unittest

from real_robot.protocol import MAX_MESSAGE_BYTES, receive_message, send_message


class ProtocolTests(unittest.TestCase):
    def setUp(self):self.left,self.right=socket.socketpair()
    def tearDown(self):self.left.close();self.right.close()
    def test_round_trip(self):
        message={'request_id':3,'prompt':'follow','trajectory':[[.1,0,0]]*8}
        send_message(self.left,message)
        self.assertEqual(receive_message(self.right),message)
    def test_invalid_length(self):
        for n in (0,MAX_MESSAGE_BYTES+1):
            self.left.sendall(struct.pack('!I',n))
            with self.assertRaises(ValueError):receive_message(self.right)
    def test_non_object(self):
        data=b'[1,2,3]';self.left.sendall(struct.pack('!I',len(data))+data)
        with self.assertRaises(ValueError):receive_message(self.right)
    def test_disconnect_mid_message(self):
        self.left.sendall(struct.pack('!I',10)+b'{}');self.left.shutdown(socket.SHUT_WR)
        with self.assertRaises(ConnectionError):receive_message(self.right)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):send_message(self.left,{'waypoint':float('nan')})
