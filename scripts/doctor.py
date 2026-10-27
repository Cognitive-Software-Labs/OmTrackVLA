"""Check runtime and optional model/ROS prerequisites without starting inference."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models-dir',type=Path,default=Path(__file__).resolve().parents[1]/'models')
    p.add_argument('--require-models',action='store_true')
    a=p.parse_args()
    import torch
    versions={n:importlib.metadata.version(n) for n in ('torch','torchvision','numpy','transformers','huggingface-hub')}
    missing=[str(a.models_dir/name/'model.safetensors') for name in (
        'OmTrackVLA-0.6B','Qwen3-0.6B','dinov3-vits16-pretrain-lvd1689m','siglip-so400m-patch14-384')
        if not (a.models_dir/name/'model.safetensors').is_file()]
    print(json.dumps({'python':sys.version,'versions':versions,'cuda_available':torch.cuda.is_available(),
        'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'missing_model_weights':missing},indent=2))
    if a.require_models and missing:raise SystemExit(1)


if __name__=='__main__':main()
