"""Ridgeback deployment defaults shared by the launchers, clients and recorders."""

CAMERA_TOPIC = '/r100_0160/sensors/camera_0/color/image/compressed'
MODEL_PORT = 18781
PREVIEW_HTTP_PORT = 18881
# Benchmarks use their own port so they never collide with a running preview.
BENCHMARK_PORT = 18779
DEFAULT_PROMPT = 'Follow the person wearing a black jacket or black T-shirt.'
