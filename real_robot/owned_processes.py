"""Start helper processes (GPU/host monitors, servers) and stop only those we started."""
import os
import signal
import subprocess


def require_idle_gpu(report):
    """Store the nvidia-smi compute-app listing as report['gpu_before']; raise if any
    process holds the GPU. The listing is stored first so a failed report shows it."""
    listing = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader'], text=True)
    report['gpu_before'] = listing
    if listing.strip():
        raise RuntimeError('Competing GPU processes present: ' + listing)


GPU_MONITOR = ['nvidia-smi', '--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw',
               '--format=csv', '-lms', '500']


class OwnedProcesses:
    """Each child gets its own process group and log file; stop() kills only those groups."""

    def __init__(self, log_dir, record):
        self.log_dir = log_dir
        self.record = record  # list that receives {'name', 'pid', 'command'} for the report
        self.children = []
        self.streams = []

    def spawn(self, name, command):
        stream = (self.log_dir / (name + '.log')).open('w')
        self.streams.append(stream)
        child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children.append(child)
        self.record.append({'name': name, 'pid': child.pid, 'command': command})
        return child

    def stop(self, timeout=10):
        for child in reversed(self.children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        for stream in self.streams:
            stream.close()
