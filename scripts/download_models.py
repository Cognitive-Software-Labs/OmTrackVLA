"""Download required pretrained assets; never write through shared model symlinks."""
import argparse
import hashlib
import json
from pathlib import Path

MODELS = {
    'OmTrackVLA-0.6B': 'omlab/OmTrackVLA-0.6B',
    'Qwen3-0.6B': 'Qwen/Qwen3-0.6B',
    'dinov3-vits16-pretrain-lvd1689m': 'facebook/dinov3-vits16-pretrain-lvd1689m',
    'siglip-so400m-patch14-384': 'google/siglip-so400m-patch14-384',
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models-dir', type=Path, default=Path(__file__).resolve().parents[1]/'models')
    p.add_argument('--tracking', action='store_true', help='Also download optional local grounding/YOLO assets')
    p.add_argument('--revisions', type=Path, default=Path(__file__).resolve().parents[1]/'requirements/model-revisions.json', help='JSON mapping local model names to exact Hugging Face commit SHAs')
    a = p.parse_args()
    if a.models_dir.is_symlink():
        p.error('Destination is a shared symlink. Use a new --models-dir; existing shared weights must remain read-only.')
    from huggingface_hub import HfApi, snapshot_download
    names = dict(MODELS)
    if a.tracking:
        names['grounding-dino-tiny'] = 'IDEA-Research/grounding-dino-tiny'
    pinned = json.loads(a.revisions.read_text()) if a.revisions else {}
    manifest = {}
    for name, repo in names.items():
        dest = a.models_dir/name
        if dest.is_symlink():
            p.error(f'Refusing shared symlink destination: {dest}')
        sha = HfApi().model_info(repo, revision=pinned[name]).sha
        snapshot_download(repo_id=repo, revision=sha, local_dir=dest,
                          allow_patterns=['*.json', '*.txt', '*.model', '*.safetensors', '*.md'])
        manifest[name] = {'repo_id': repo, 'revision': sha}
        print(name, sha, flush=True)
    if a.tracking:
        from ultralytics import YOLO
        a.models_dir.mkdir(parents=True, exist_ok=True)
        import os
        previous = Path.cwd()
        try:
            os.chdir(a.models_dir)
            YOLO('yolo11n.pt')
        finally:
            os.chdir(previous)
        path = a.models_dir/'yolo11n.pt'
        manifest['yolo11n.pt'] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (a.models_dir/'download-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Downloaded. Keep download-manifest.json to reproduce these exact revisions.')


if __name__ == '__main__':
    main()
