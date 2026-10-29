"""Training/evaluation data: cached vision tokens, action- or waypoint-labelled examples.

Single source for train.py. Examples come from a JSON list, a JSONL file or a
directory of JSONL files; vision tokens are read from <root>/vision_cache and
encoded online (then cached) when missing. See docs/WAYPOINT_DATA_CONTRACT.md for
how labels and short histories are formed.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json
from pathlib import Path
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset

from cache_gridpool import VisionFeatureCacher, VisionCacheConfig


def load_tokens_file(path: str) -> torch.Tensor:
    try:
        obj = torch.load(path, map_location='cpu')
    except Exception:
        # PyTorch 2.6 defaults to weights_only=True; retry allowing full unpickling for trusted caches
        try:
            obj = torch.load(path, map_location='cpu', weights_only=False)
        except Exception as e:
            raise e
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, dict):
        for k in ("V", "Vfine", "Vcoarse", "tokens", "feat", "features"):
            if k in obj and isinstance(obj[k], torch.Tensor):
                t = obj[k]
                if t.dim() == 3 and t.size(0) == 1:
                    t = t[0]
                return t.float()
    raise ValueError(f"Unrecognized token file: {path}")


def integrate_actions_to_waypoints(actions: np.ndarray, n_waypoints: int, dt: float = 0.2) -> np.ndarray:
    a = np.asarray(actions, dtype=np.float32)
    if a.ndim == 1: a = a[None, :]
    T, D = a.shape
    vx = a[:, 0].astype(np.float32)
    vy = a[:, 1].astype(np.float32) if D > 1 else np.zeros_like(vx)
    wz = a[:, 2].astype(np.float32) if D > 2 else np.zeros_like(vx)

    x = np.zeros(T, dtype=np.float32)
    y = np.zeros(T, dtype=np.float32)
    th = np.zeros(T, dtype=np.float32)

    for t in range(1, T):
        th[t] = th[t-1] + wz[t-1] * dt
        c, s = np.cos(th[t-1]), np.sin(th[t-1])
        x[t] = x[t-1] + (c * vx[t-1] - s * vy[t-1]) * dt
        y[t] = y[t-1] + (s * vx[t-1] + c * vy[t-1]) * dt

    traj = np.stack([x, y, th], axis=-1)
    if n_waypoints <= 1: return traj[-1:]
    idx = np.linspace(0, T-1, n_waypoints).round().astype(int)
    return traj[idx]


# Flexible loader for JSON/JSONL datasets and directories
def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    examples: List[Dict[str, Any]] = []
    with open(file_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            examples.append(json.loads(s))
    return examples


def load_examples_from_path(train_path: str) -> List[Dict[str, Any]]:
    """Load dataset examples from a JSON list, a JSONL file, or a directory (recursively).
    - If directory: recursively loads all .jsonl files in subfolders.
    - If file: supports .json (list of dicts) or .jsonl (one JSON per line).
    """
    p = Path(train_path)
    if p.is_dir():
        jsonl_files = sorted(p.rglob('*.jsonl'))
        if len(jsonl_files) == 0:
            raise FileNotFoundError(f"No .jsonl files found under directory: {train_path}")
        all_items: List[Dict[str, Any]] = []
        for fp in jsonl_files:
            all_items.extend(_read_jsonl_file(str(fp)))
        return all_items
    if p.is_file():
        if p.suffix.lower() == '.jsonl':
            return _read_jsonl_file(str(p))
        if p.suffix.lower() == '.json':
            with open(p, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(f"JSON file must contain a list at top-level: {train_path}")
            return data
        raise ValueError(f"Unsupported file type: {train_path}")
    raise FileNotFoundError(f"Path does not exist: {train_path}")


# ----------------------- Dataset -----------------------

@dataclass
class DataConfig:
    train_json: str
    n_waypoints: int = 8
    history: int = 31
    default_dt: float = 0.1
    cache_root: Optional[str] = None


class JsonTrackingDataset(Dataset):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        p = Path(cfg.train_json)
        # Determine base root to prefix relative paths in JSON examples
        # Robustly ascend until 'frames' directory is found (common layout: <root>/frames/...)
        candidate = p if p.is_dir() else p.parent
        max_up = 4
        while max_up >= 0 and not (candidate / 'frames').exists():
            if candidate.parent == candidate:
                break
            candidate = candidate.parent
            max_up -= 1
        self.base_root = candidate
        # Determine cache root for token files (defaults to <base_root>/vision_cache)
        self.cache_root = Path(cfg.cache_root) if cfg.cache_root is not None else (self.base_root / "vision_cache")
        # Lazy online encoder (created on first use)
        self._online_encoder: Optional[VisionFeatureCacher] = None
        # Dataset storage: either eager JSON list or lazy JSONL/directory index
        self._lazy = False
        self._index: Optional[List[Tuple[str, int]]] = None  # list of (file_path, byte_offset) per example
        self.examples: Optional[List[Dict[str, Any]]] = None
        if p.is_file() and p.suffix.lower() == '.json':
            data = load_examples_from_path(cfg.train_json)
            assert isinstance(data, list) and len(data) > 0, "JSON file must contain a non-empty list"
            self.examples = data
        else:
            # Build lazy index over .jsonl file or directory of .jsonl files
            files: List[Path] = []
            if p.is_file() and p.suffix.lower() == '.jsonl':
                files = [p]
            elif p.is_dir():
                files = sorted(p.rglob('*.jsonl'))
            if len(files) == 0:
                raise FileNotFoundError(f"No .jsonl files found under: {cfg.train_json}")
            self._lazy = True
            self._index = []
            for fp in files:
                try:
                    with open(fp, 'rb') as f:
                        pos = 0
                        while True:
                            line = f.readline()
                            if not line:
                                break
                            if line.strip():
                                self._index.append((str(fp), pos))
                            pos += len(line)
                except Exception as _e:
                    raise _e
            if len(self._index) == 0:
                raise RuntimeError(f"No examples indexed from .jsonl sources under: {cfg.train_json}")
        # Target history length is configured; per-sample sequences will be padded/trimmed to this length
        H_target = int(self.cfg.history)
        self.coarse_frames = H_target

    def __len__(self):
        if self.examples is not None:
            return len(self.examples)
        if self._index is not None:
            return len(self._index)
        return 0

    def _load_tokens(self, path: str) -> torch.Tensor:
        return load_tokens_file(path)

    def _get_online_encoder(self) -> VisionFeatureCacher:
        if self._online_encoder is None:
            # Use CPU when running with multiple workers to avoid GPU contention
            from torch.utils.data import get_worker_info
            worker_info = get_worker_info()
            use_cuda = torch.cuda.is_available() and (worker_info is None)
            cfg = VisionCacheConfig(image_size=384, batch_size=8, device=('cuda' if use_cuda else 'cpu'))
            self._online_encoder = VisionFeatureCacher(cfg)
            self._online_encoder.eval()
        return self._online_encoder

    @torch.inference_mode()
    def _encode_image_tokens(self, img_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        enc = self._get_online_encoder()
        pil = Image.open(str(img_path)).convert('RGB')
        Vfine, Vcoarse = enc.encode_pil_fine_coarse(pil)          # (1, 64, C), (1, 4, C)
        return Vcoarse[0].cpu().float(), Vfine[0].cpu().float()

    def _read_indexed_example(self, idx: int) -> Dict[str, Any]:
        assert self._lazy and self._index is not None
        fp, off = self._index[idx]
        with open(fp, 'rb') as f:
            f.seek(off)
            line = f.readline()
        return json.loads(line.decode('utf-8'))

    def get_example(self, idx: int) -> Dict[str, Any]:
        if self._lazy:
            return self._read_indexed_example(idx)
        assert self.examples is not None
        return self.examples[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.get_example(idx)
        H = self.coarse_frames

        # fine current (load first to determine feature dimensionality for zero padding)
        curr_path = Path(ex['current'])
        abs_curr_img = curr_path if curr_path.is_absolute() else (self.base_root / curr_path)
        try:
            rel_curr = abs_curr_img.relative_to(self.base_root)
        except ValueError:
            rel_curr = abs_curr_img
        curr_token_dir = self.cache_root / rel_curr.parent
        curr_token_name = rel_curr.stem + "_vfine.pt"
        curr_tok_path = curr_token_dir / curr_token_name
        try:    
            fine_tokens = self._load_tokens(str(curr_tok_path))  # (64, C)
        except Exception as e:            
            # Fallback: encode online and save for reuse
            curr_token_dir.mkdir(parents=True, exist_ok=True)
            vc, vf = self._encode_image_tokens(abs_curr_img)
            print (curr_tok_path)
            try:
                torch.save(vf.half(), str(curr_tok_path))
            except Exception as e:
                print (str(e))
                pass
            fine_tokens = vf
        fine_tidx = torch.full((fine_tokens.size(0),), fill_value=H, dtype=torch.long)

        # coarse history with left-padding using earliest available token (edge padding)
        imgs_src = ex.get('images', [])
        imgs_trim = imgs_src[-H:]
        missing = H - len(imgs_trim)
        coarse_list, coarse_tidx = [], []
        first_tok: Optional[torch.Tensor] = None
        current_vc: Optional[torch.Tensor] = None
        for t in range(H):
            if t < missing:
                # placeholder; will fill with first_tok after we load at least one real token
                tok = None
            else:
                img_p = imgs_trim[t - missing]
                # Map image path -> corresponding Vcoarse token path within cache_root
                rp = Path(img_p)
                abs_img = rp if rp.is_absolute() else (self.base_root / rp)
                try:
                    rel_img = abs_img.relative_to(self.base_root)
                except ValueError:
                    rel_img = abs_img
                token_dir = self.cache_root / rel_img.parent
                token_name = rel_img.stem + "_vcoarse.pt"
                tok_path = token_dir / token_name
                try:
                    tok = self._load_tokens(str(tok_path))
                except Exception as e:
                    # Fallback: encode online from image and save for reuse
                    token_dir.mkdir(parents=True, exist_ok=True)
                    vc, vf = self._encode_image_tokens(abs_img)
                    try:
                        torch.save(vc.half(), str(tok_path))
                    except Exception:
                        pass
                    tok = vc
                if first_tok is None:
                    first_tok = tok
            # Fill left padding with first available token (edge padding);
            # if none, use current frame's coarse tokens; fallback to zeros if that also fails
            if tok is None:
                if first_tok is not None:
                    tok = first_tok
                else:
                    # Try to obtain current coarse tokens
                    try:
                        if current_vc is None:
                            # Attempt to load cached current coarse tokens alongside fine
                            cur_coarse_name = rel_curr.stem + "_vcoarse.pt"
                            cur_coarse_path = curr_token_dir / cur_coarse_name
                            try:
                                current_vc = self._load_tokens(str(cur_coarse_path))
                            except Exception:
                                # Encode from current image if not cached
                                vc_tmp, _ = self._encode_image_tokens(abs_curr_img)
                                current_vc = vc_tmp
                        tok = current_vc
                    except Exception:
                        tok = torch.zeros(4, fine_tokens.size(1), dtype=torch.float32)
            coarse_list.append(tok)
            coarse_tidx.append(torch.full((tok.size(0),), fill_value=t, dtype=torch.long))
        coarse_tokens = torch.cat(coarse_list, dim=0)      # (H*4, C)
        coarse_tidx   = torch.cat(coarse_tidx, dim=0)      # (H*4,)

        # yaw (optional; used only if angle TVI is enabled)
        yaw_hist = torch.tensor(ex.get('yaw_hist', [0.0]*H), dtype=torch.float32)            # (H,)
        yaw_curr = torch.tensor(ex.get('yaw_curr', 0.0), dtype=torch.float32).view(1)        # (1,)

        # targets
        if 'waypoints' in ex:
            wp = torch.tensor(ex['waypoints'], dtype=torch.float32)
        else:
            assert 'actions' in ex, "JSON needs either 'waypoints' or 'actions'"
            dt = float(ex.get('dt', self.cfg.default_dt))
            traj = integrate_actions_to_waypoints(np.asarray(ex['actions'], dtype=np.float32), self.cfg.n_waypoints, dt)
            wp = torch.from_numpy(traj)

        # validity mask/idx (optional)
        if 'valid_mask' in ex:
            vm = torch.tensor(ex['valid_mask'], dtype=torch.bool)
        elif 'valid_idx' in ex:
            vm = torch.zeros(self.cfg.n_waypoints, dtype=torch.bool)
            vm[torch.tensor(ex['valid_idx'], dtype=torch.long)] = True
        else:
            vm = torch.ones(self.cfg.n_waypoints, dtype=torch.bool)

        item: Dict[str, Any] = {
            'coarse_tokens': coarse_tokens,
            'coarse_tidx':   coarse_tidx,
            'fine_tokens':   fine_tokens,
            'fine_tidx':     fine_tidx,
            'yaw_hist':      yaw_hist,     # (H,)
            'yaw_curr':      yaw_curr,     # (1,)
            'waypoints':     wp,           # (M, D_action)
            'valid_mask':    vm,           # (M,)
            'instruction':   ex.get('instruction', 'follow the person'),
            'current_path':  str(abs_curr_img),
        }
        return item


# ----------------------- Collate -----------------------

def collate_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    instr = [b['instruction'] for b in batch]
    return {
        'coarse_tokens': torch.stack([b['coarse_tokens'] for b in batch], dim=0),
        'coarse_tidx':   torch.stack([b['coarse_tidx']   for b in batch], dim=0),
        'fine_tokens':   torch.stack([b['fine_tokens']   for b in batch], dim=0),
        'fine_tidx':     torch.stack([b['fine_tidx']     for b in batch], dim=0),
        'yaw_hist':      torch.stack([b['yaw_hist']      for b in batch], dim=0),   # (B,H)
        'yaw_curr':      torch.stack([b['yaw_curr']      for b in batch], dim=0),   # (B,1)
        'waypoints':     torch.stack([b['waypoints']     for b in batch], dim=0),
        'valid_mask':    torch.stack([b['valid_mask']    for b in batch], dim=0),
        'instruction':   instr,
        'current_path':  [b['current_path'] for b in batch]
    }
