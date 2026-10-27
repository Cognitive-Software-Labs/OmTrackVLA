# Compute benchmark protocol

Measure identical RGB frames, prompts, model hashes, preprocessing and ordered
history. Reset at sequence/repetition boundaries. Warm up before timing. Separate
fresh inference, cached replies, target-acquisition skips, grounding and ordinary
tracking. Capture startup, latency percentiles, throughput, CUDA allocator peaks,
host memory and sampled GPU load. Reject occupied GPUs; preserve unrelated jobs.

The script reads `data/benchmark/samples.json` by default. Generate it using
`scripts/prepare_benchmark.py`; use `--samples` and `--data-root` for another
portable fixture dataset. Original private camera recordings are not included.
Optional native-mode benchmarks require the tracking dependencies and weights.

## Measured reference (2026-10-27)

These are existing measurements on identical private lab RGB fixtures (40 frames,
two repetitions), not a promise for other inputs or hardware. No background GPU
compute processes were detected before each run; ordinary OS/robot services
remained active. An exhaustive continuous process audit was not collected.

| Case | RTX 5060 Ridgeback median / p95 ms | Thor median / p95 ms |
|---|---:|---:|
| Model-only, every call fresh | 167.68 / 169.12 | 273.90 / 278.54 |
| Local optional perception, ordinary frames | 23.50 / 32.40 | 16.70 / 23.00 |
| Local optional perception, grounding frames | 236.17 / 249.28 | 398.34 / 408.45 |
| Local full wrapper, frames invoking VLA | 202.94 / 207.24 | 282.79 / 298.79 |

The optional full wrapper locked and invoked VLA on 18/80 frames per host;
62/80 skipped VLA. All 18 outputs passed its finite/directional checks.
Those samples were sparse and do not qualify continuous identity recovery.
Ridgeback-to-Thor model-only round-trip: 271.09 ms median / 279.58 ms p95,
80/80 fresh finite outputs. This does not test distributed native target tracking.

Ridgeback used Python 3.9, PyTorch 2.8 CUDA 12.8; Thor Python 3.12, PyTorch 2.11
CUDA 13.0. Transformers/model hashes matched. Thor unified memory fields were
unavailable in nvidia-smi; they must not be treated as zero. Native full CUDA
allocator peaks were approximately 4964 MiB Ridgeback / 4695 MiB Thor.

## Remote service benchmark

Prepare a separate clone and compatible CUDA environment on the remote host.
Download the same model revisions there. The helper does not install or modify
other projects. Set all deployment addresses/paths explicitly:

```bash
PYTHONPATH=. .venv/bin/python -m real_robot.run_remote_omtrack_case \
  --source LOCAL_WIRED_IPV4 --thor REMOTE_IPV4 --remote-user USER \
  --remote-root /absolute/path/OmTrackVLA \
  --remote-python /absolute/path/OmTrackVLA/.venv/bin/python \
  --output outputs/benchmark/remote
```

It starts only owned model/telemetry processes, invokes a bounded client and
collects remote logs. The remote socket uses port 18779 and an explicit allowed
client address. Use a trusted restricted network. Output directories are fresh
and are never silently overwritten. ARM64 needs a vendor CUDA build rather
than the x86 setup script. Remote serving is not multi-GPU model parallelism.
