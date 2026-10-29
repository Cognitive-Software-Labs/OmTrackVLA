"""Subscribe to Ridgeback RGB and display OmTrackVLA waypoints; publishes no ROS topics."""
import argparse
import base64
import json
from pathlib import Path
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

from real_robot.defaults import CAMERA_TOPIC, DEFAULT_PROMPT, MODEL_PORT, PREVIEW_HTTP_PORT
from real_robot.model_client import ModelClient, has_finite_waypoints
from real_robot.ros_camera import latest_frame_qos, stamp_ns

PAGE = '''<!doctype html><html><head><meta charset="utf-8"><title>Ridgeback · OmTrackVLA</title>
<style>body{margin:0;background:#10171e;color:#e8eff4;font:16px system-ui}main{max-width:1360px;margin:auto;padding:26px}h1{margin:0 0 8px;font-size:28px}p{color:#b1c2cf}header{display:flex;justify-content:space-between;align-items:center}.badge{background:#153e39;color:#82e4ce;padding:8px 14px;border-radius:20px}.panes{display:grid;grid-template-columns:1.2fr 1fr;gap:20px}.card{background:#1a2631;border:1px solid #334656;border-radius:12px;padding:18px}img{width:100%;min-height:220px;object-fit:contain;background:#080e14}canvas{width:100%;background:#111d27}h2{margin-top:0;font-size:19px}#status{font-weight:600;color:#80dece}table{width:100%;border-collapse:collapse;font:13px monospace}th,td{text-align:right;padding:5px;border-bottom:1px solid #334656}footer{color:#9db0be;margin-top:18px;font-size:14px}@media(max-width:900px){.panes{grid-template-columns:1fr}}</style></head>
<body><main><header><h1>Ridgeback · OmTrackVLA</h1><span class="badge">Preview only · no robot commands</span></header>
<p>Live camera frames → DINOv3 / SiGLIP → OmTrackVLA → eight predicted waypoints</p>
<div class="panes"><section class="card"><h2>Ridgeback camera · model input</h2><img id="camera" alt="Waiting for Ridgeback camera"><p id="prompt"></p><div id="status">Waiting for camera and model…</div><p id="metrics"></p></section>
<section class="card"><h2>Waypoints predicted by OmTrackVLA</h2><canvas id="plot" width="560" height="430"></canvas><p>Robot-relative view: forward ↑ · left ← · distances in metres</p><table><thead><tr><th>#</th><th>Forward x (m)</th><th>Left y (m)</th><th>Yaw (rad)</th></tr></thead><tbody id="points"></tbody></table></section></div>
<footer>The image and plotted waypoints come from the same inference request. Model-only mode does not verify person identity, obstacles or following distance. This page subscribes to camera data and never publishes movement commands.</footer></main>
<script>
const canvas=document.getElementById('plot'),ctx=canvas.getContext('2d');
function draw(points){ctx.clearRect(0,0,560,430);const scale=340/Math.max(1,...points.map(p=>Math.max(Math.abs(p[0]),Math.abs(p[1]))*1.4));const ox=280,oy=370;ctx.strokeStyle='#2d4354';ctx.fillStyle='#90a9bb';ctx.font='12px system-ui';
for(let v=-3;v<=3;v+=.5){const x=ox-v*scale;if(x>=0&&x<=560){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,430);ctx.stroke()}const y=oy-v*scale;if(y>=0&&y<=430){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(560,y);ctx.stroke();ctx.fillText(v.toFixed(1)+' m',8,y-4)}}
ctx.fillStyle='#70a8ff';ctx.beginPath();ctx.arc(ox,oy,8,0,Math.PI*2);ctx.fill();ctx.fillText('Ridgeback',ox+12,oy+5);
ctx.strokeStyle='#5ce2bd';ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(ox,oy);for(const p of points)ctx.lineTo(ox-p[1]*scale,oy-p[0]*scale);ctx.stroke();ctx.lineWidth=1;
points.forEach((p,i)=>{const x=ox-p[1]*scale,y=oy-p[0]*scale;ctx.fillStyle='#f4d27a';ctx.beginPath();ctx.arc(x,y,4,0,Math.PI*2);ctx.fill();ctx.fillStyle='#dce7ef';ctx.fillText(String(i+1),x+6,y-5)})}
draw([]);
async function refresh(){try{const s=await(await fetch('/state',{cache:'no-store'})).json();document.getElementById('prompt').textContent='Instruction: '+s.prompt;document.getElementById('status').textContent=s.status;document.getElementById('metrics').textContent=`Fresh predictions: ${s.fresh_predictions||0} · inference: ${s.inference_ms?.toFixed(0)||'—'} ms · image age: ${s.image_age_s?.toFixed(2)||'—'} s`;
const usable=s.prediction_age_s!=null&&s.prediction_age_s<2&&s.image_age_s<2;const points=usable?(s.trajectory||[]):[];if(s.image_b64)document.getElementById('camera').src='data:image/jpeg;base64,'+s.image_b64;draw(points);document.getElementById('points').replaceChildren(...points.map((p,i)=>{const tr=document.createElement('tr');for(const v of [i+1,...p.map(v=>v.toFixed(4))]){const td=document.createElement('td');td.textContent=v;tr.append(td)}return tr}));}catch(e){document.getElementById('status').textContent='Preview disconnected';draw([])}setTimeout(refresh,350)}refresh();
</script></body></html>'''


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--topic', default=CAMERA_TOPIC)
    p.add_argument('--model-port', type=int, default=MODEL_PORT)
    p.add_argument('--http-port', type=int, default=PREVIEW_HTTP_PORT)
    p.add_argument('--prompt', default=DEFAULT_PROMPT)
    p.add_argument('--output', type=Path, default=Path('outputs/ridgeback_omtrack'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()
    halt = threading.Event()
    latest = None
    state = {'status': 'Waiting for camera', 'prompt': a.prompt, 'fresh_predictions': 0, 'trajectory': []}
    log = (a.output / 'live-predictions.jsonl').open('a', buffering=1)

    class Camera(Node):
        def __init__(self):
            super().__init__('omtrack_camera_preview')
            self.create_subscription(CompressedImage, a.topic, self.image, latest_frame_qos())

        def image(self, msg):
            nonlocal latest
            stamp = stamp_ns(msg.header)
            with lock:
                latest = (stamp, time.monotonic(), bytes(msg.data))

    def worker():
        client = None
        previous = None
        reset = True
        while not halt.is_set():
            with lock:
                frame = latest
            if frame is None or frame[0] == previous:
                halt.wait(.05)
                continue
            stamp, arrival, jpeg = frame
            if time.monotonic() - arrival > 1:
                with lock:
                    state.update(status='Camera input stale; waiting', trajectory=[])
                reset = True
                halt.wait(.1)
                continue
            try:
                if client is None:
                    client = ModelClient(a.model_port, connect_timeout=2, timeout=15)
                encoded = base64.b64encode(jpeg).decode()
                before = time.monotonic()
                reply = client.infer(encoded, a.prompt, stamp, reset_target=reset,
                                     frame_id=str(stamp), mode='model_only')
                points = reply.get('trajectory')
                # Cached replies are shown too, but must still hold eight finite waypoints.
                if not isinstance(points, list) or not has_finite_waypoints(points):
                    raise ValueError('Nonfinite/wrong-shaped waypoints')
                fresh = reply.get('planner_updated') is True
                previous = stamp
                reset = False
                finished = time.monotonic()
                with lock:
                    count = state['fresh_predictions'] + int(fresh)
                    state.update(
                        status='Live · fresh OmTrackVLA prediction' if fresh else 'Live · cached model response',
                        fresh_predictions=count, trajectory=points, image_b64=encoded, image_arrival=arrival,
                        prediction_time=finished, source_stamp_ns=stamp,
                        inference_ms=reply['inference_seconds'] * 1000)
                log.write(json.dumps({'wall_time': time.time(), 'source_stamp_ns': stamp, 'fresh': fresh,
                                      'inference_ms': reply['inference_seconds'] * 1000,
                                      'roundtrip_ms': (finished - before) * 1000,
                                      'input_arrival_age_s': finished - arrival,
                                      'trajectory': points}) + '\n')
                (a.output / 'latest-prediction.json').write_text(
                    json.dumps({'source_stamp_ns': stamp, **reply}, indent=2) + '\n')
                if count == 1 or count % 10 == 0:
                    print(json.dumps({'fresh_predictions': count, 'source_stamp_ns': stamp,
                                      'inference_ms': reply['inference_seconds'] * 1000}), flush=True)
                halt.wait(.55)
            except (OSError, ValueError, KeyError, ConnectionError) as e:
                if client is not None:
                    client.close()
                    client = None
                reset = True
                with lock:
                    state.update(status='Waiting for model: ' + str(e), trajectory=[])
                halt.wait(1)
        if client is not None:
            client.close()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/':
                body = PAGE.encode()
                kind = 'text/html; charset=utf-8'
            elif self.path == '/state':
                with lock:
                    current = dict(state)
                now = time.monotonic()
                current['prediction_age_s'] = now - current.pop('prediction_time', now - 999)
                current['image_age_s'] = now - current.pop('image_arrival', now - 999)
                if current['image_age_s'] >= 2:
                    current['status'] = 'Camera/prediction stale; waiting for fresh input'
                body = json.dumps(current, allow_nan=False).encode()
                kind = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', kind)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    rclpy.init()
    node = Camera()
    http = ThreadingHTTPServer(('127.0.0.1', a.http_port), Handler)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    print(json.dumps({'preview': f'http://127.0.0.1:{a.http_port}', 'camera_topic': a.topic,
                      'publishes_ros_topics': False}), flush=True)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        halt.set()
        thread.join(timeout=17)
        http.shutdown()
        http.server_close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        log.close()


if __name__ == '__main__':
    main()
