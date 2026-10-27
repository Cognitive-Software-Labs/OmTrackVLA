"""One bounded Ridgeback-to-Thor OmTrackVLA-only socket case with owned cleanup."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True,help='Local wired IPv4 address reachable from the remote host')
    p.add_argument('--thor',required=True,help='Remote inference host IPv4 address')
    p.add_argument('--remote-user',default='robot')
    p.add_argument('--remote-root',required=True,help='Absolute path of the same checkout on the remote host')
    p.add_argument('--remote-python',required=True,help='Absolute remote CUDA Python executable')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    remote_root=a.remote_root.rstrip('/')
    if not remote_root.startswith('/') or not a.remote_python.startswith('/'):
        p.error('Remote checkout and Python paths must be absolute')
    if a.output.exists():raise FileExistsError(a.output)
    ssh=['ssh','-b',a.source,'-o','BatchMode=yes','-o','ConnectTimeout=5',a.remote_user+'@'+a.thor]
    remote_run=remote_root+'/results/'+a.output.parent.name+'-'+a.output.name+'-server'
    def run_python(code):
        result=subprocess.run(ssh+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,timeout=30)
        if result.returncode:raise RuntimeError(result.stderr or result.stdout)
        return result
    def interrupted(*_):raise KeyboardInterrupt('Cleaning up owned Thor server')
    signal.signal(signal.SIGTERM,interrupted)
    start='''import json,os,socket,subprocess
from pathlib import Path
root=ROOT
out=Path(OUT)
check=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
if check.strip():raise RuntimeError('Competing GPU process: '+check)
with socket.socket() as s:s.bind((HOST,18779))
out.mkdir(parents=True,exist_ok=False)
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
env['PYTHONPATH']=root
commands={'server':[REMOTE_PYTHON,'-u',root+'/real_robot/inference_server.py','--host',HOST,'--allowed-client',CLIENT,'--port','18779','--assets',root,'--mode','model_only','--planner-rate','1000000000','--legacy-vision-preprocess'],
'gpu':['nvidia-smi','--query-gpu=timestamp,utilization.gpu,memory.used,temperature.gpu,power.draw','--format=csv','-lms','500'],
'host-metrics':[REMOTE_PYTHON,root+'/real_robot/sample_compute_host.py']}
owned=[]
try:
 for name,cmd in commands.items():
  with (out/(name+'.log')).open('w') as f:
   child=subprocess.Popen(cmd,cwd=root,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  owned.append({'name':name,'pid':child.pid,'command':cmd})
  (out/'owned.json').write_text(json.dumps(owned,indent=2))
 print(json.dumps(owned))
except BaseException:
 import signal
 for row in owned:
  try:os.killpg(row['pid'],signal.SIGTERM)
  except ProcessLookupError:pass
 raise
'''.replace('root=ROOT', 'root='+repr(remote_root)).replace('REMOTE_PYTHON',repr(a.remote_python)).replace('Path(OUT)', 'Path('+repr(remote_run)+')').replace('HOST',repr(a.thor)).replace('CLIENT',repr(a.source))
    started=False
    try:
        reply=run_python(start);started=True
        print('Started owned Thor model-only server: '+reply.stdout,flush=True)
        command=[sys.executable,'-u',str(ROOT/'real_robot/benchmark_omtrack_only.py'),'socket',
                 '--host',a.thor,'--source-address',a.source,'--output',str(a.output)]
        child=subprocess.Popen(command,cwd=ROOT,start_new_session=True)
        try:child.wait();check=child.returncode
        except BaseException:
            child.send_signal(signal.SIGTERM);child.wait();raise
        if check:raise RuntimeError(f'Client exited with status {check}')
    finally:
        if started:
            stop='''import json,os,signal,time
from pathlib import Path
out=Path(OUT)
rows=json.loads((out/'owned.json').read_text())
for row in reversed(rows):
 try:
  pid=row['pid'];cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
  if row['command'][0].encode() not in cmd:raise RuntimeError('PID identity mismatch')
  os.killpg(pid,signal.SIGTERM)
 except (ProcessLookupError,FileNotFoundError):pass
time.sleep(3)
for row in reversed(rows):
 try:os.killpg(row['pid'],signal.SIGKILL)
 except ProcessLookupError:pass
print('Stopped owned server and monitors')
'''.replace('Path(OUT)', 'Path('+repr(remote_run)+')')
            print(run_python(stop).stdout,flush=True)
            a.output.mkdir(parents=True,exist_ok=True)
            subprocess.run(['scp','-r','-o','BindAddress='+a.source,'-o','BatchMode=yes',
                a.remote_user+'@'+a.thor+':'+remote_run,str(a.output/'thor-server')],check=True)


if __name__=='__main__':main()
