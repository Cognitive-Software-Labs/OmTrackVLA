"""Stop only this preview's recorded, identity-checked local processes."""
import json
import os
from pathlib import Path
import signal

ROOT=Path(__file__).resolve().parents[1]


def main():
    path=ROOT/'outputs/ridgeback_omtrack/runtime.json'
    manifest=json.loads(path.read_text())
    for row in sorted(manifest['processes'],key=lambda r:r['role']=='model'):
        pid=row['pid']
        try:
            argv=(Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0')
            if argv!=[v.encode() for v in row['argv']]:
                raise RuntimeError(f'PID {pid} changed identity; not stopping it')
            os.kill(pid,signal.SIGINT)
            print('Stopped',row['role'],'PID',pid)
        except (FileNotFoundError,ProcessLookupError):
            print(row['role'],'already stopped')


if __name__=='__main__':main()
