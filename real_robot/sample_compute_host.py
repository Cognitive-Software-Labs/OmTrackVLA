"""Read-only host CPU, RAM, network and process memory telemetry."""
import json
import os
import signal
import time
from pathlib import Path


def main():
    running=True
    def stop(*_):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop)
    previous=None
    while running:
        cpu=[int(v) for v in Path('/proc/stat').read_text().splitlines()[0].split()[1:9]]
        total=sum(cpu);idle=cpu[3]+cpu[4]
        mem={line.split(':')[0]:int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.split()[1].isdigit()}
        processes=[]
        for path in Path('/proc').iterdir():
            if not path.name.isdigit():continue
            try:
                command=(path/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
                if any(name in command for name in ('inference_server.py','benchmark_omtrack_only.py','benchmark_native_modes.py','real_robot.live_omtrack_preview')):
                    fields=(path/'statm').read_text().split()
                    processes.append({'pid':int(path.name),'rss_bytes':int(fields[1])*os.sysconf('SC_PAGE_SIZE')})
            except (OSError,ValueError,IndexError):pass
        percent=100*(1-(idle-previous[1])/(total-previous[0])) if previous and total>previous[0] else None
        print(json.dumps({'wall_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'cpu_busy_percent_of_host':percent,
            'mem_total_bytes':mem['MemTotal'],'mem_available_bytes':mem['MemAvailable'],
            'used_excluding_available_bytes':mem['MemTotal']-mem['MemAvailable'],
            'model_and_controller_process_rss':processes}),flush=True)
        previous=(total,idle)
        time.sleep(.5)


if __name__=='__main__':main()
