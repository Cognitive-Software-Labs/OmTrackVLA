#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import os, json, math, argparse, time, csv
from pathlib import Path
from contextlib import nullcontext
from PIL import Image, ImageDraw

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModel
from cache_gridpool import VisionFeatureCacher, VisionCacheConfig, grid_pool_tokens, adapt_siglip_grid


# ----------------------- utils -----------------------

# Silence tokenizers fork warnings in dataloader workers
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

def set_seed(seed: int):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _cleanup_state_dict_keys(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make checkpoints usable across single-GPU and DDP.

    - If keys are prefixed with 'module.' (DDP/DataParallel), strip that prefix.
    - Otherwise return unchanged.
    """
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict

# ----------------------- Sanity checks -----------------------

def _dataset_sanity_report(ds: 'JsonTrackingDataset', cfg: 'TrainConfig', max_items: int = 512):
    try:
        import numpy as _np
        n = min(max_items, len(ds))
        xs, ys, thetas = [], [], []
        mask_cov = []
        yaw_hist_present = 0
        yaw_curr_present = 0
        img_ok = 0
        for i in range(n):
            ex = ds.get_example(i)
            # compute target waypoints without triggering token encoding
            if 'waypoints' in ex:
                wp = _np.asarray(ex['waypoints'], dtype=_np.float32)
            elif 'actions' in ex:
                dt = float(ex.get('dt', ds.cfg.default_dt))
                wp = integrate_actions_to_waypoints(_np.asarray(ex['actions'], dtype=_np.float32), cfg.n_waypoints, dt)
            else:
                continue
            xs.append(wp[:, 0])
            if wp.shape[1] >= 2:
                ys.append(wp[:, 1])
            if wp.shape[1] >= 3:
                thetas.append(wp[:, 2])
            # valid mask coverage
            if 'valid_mask' in ex and isinstance(ex['valid_mask'], list):
                mv = _np.asarray(ex['valid_mask'], dtype=bool)
                mask_cov.append(float(mv.mean()))
            elif 'valid_idx' in ex and isinstance(ex['valid_idx'], list):
                mv = _np.zeros(cfg.n_waypoints, dtype=bool)
                idx = _np.asarray(ex['valid_idx'], dtype=int)
                mv[_np.clip(idx, 0, cfg.n_waypoints-1)] = True
                mask_cov.append(float(mv.mean()))
            # yaw fields presence
            if 'yaw_hist' in ex:
                yaw_hist_present += 1
            if 'yaw_curr' in ex:
                yaw_curr_present += 1
            # quick current image existence check (without IO)
            cur_rel = Path(ex.get('current', ''))
            if str(cur_rel):
                cur_abs = cur_rel if cur_rel.is_absolute() else (ds.base_root / cur_rel)
                if cur_abs.exists():
                    img_ok += 1
        if xs:
            x = _np.concatenate(xs)
            x_mu, x_sd = float(_np.mean(x)), float(_np.std(x))
        else:
            x_mu = x_sd = float('nan')
        if ys:
            y = _np.concatenate(ys)
            y_mu, y_sd = float(_np.mean(y)), float(_np.std(y))
        else:
            y_mu = y_sd = float('nan')
        th_sd = float(_np.std(_np.concatenate(thetas))) if thetas else float('nan')
        cov_mu = float(_np.mean(mask_cov)) if mask_cov else float('nan')
        print(f"[SANITY] GT x(mean={x_mu:.3f}, std={x_sd:.3f}) y(mean={y_mu:.3f}, std={y_sd:.3f}) theta(std={th_sd:.3f}) | mask_cov_mean={cov_mu:.3f}")
        print(f"[SANITY] yaw_hist_present={yaw_hist_present}/{n} yaw_curr_present={yaw_curr_present}/{n} | current_img_exists={img_ok}/{n}")
        # simple warnings
        if _np.isfinite(y_sd) and y_sd < 0.05:
            print("[SANITY][warn] GT lateral std is very small; model may learn straight lines.")
        if _np.isfinite(cov_mu) and cov_mu < 0.2:
            print("[SANITY][warn] Many waypoints are invalid; training signal may be sparse.")
    except Exception as _e:
        print(f"[SANITY] skipped due to error: {_e}")


# ----------------------- TVI + projector + planner + model -----------------------

# Single definition shared with inference (open_trackvla_hf loads model.OpenTrackVLA).
from model import TVIEmbedder, CrossModalityProjector, PlannerHead3L, ModelConfig, OpenTrackVLA  # noqa: E402
from tracking_dataset import (load_tokens_file, integrate_actions_to_waypoints,  # noqa: E402
                              load_examples_from_path, DataConfig, JsonTrackingDataset, collate_batch)


# ----------------------- Loss & Train -----------------------

def mse_masked(pred: torch.Tensor, target: torch.Tensor, mask_waypoints: torch.Tensor) -> torch.Tensor:
    """Mean squared error over selected waypoints (absolute units)."""
    assert pred.shape == target.shape
    B, M, D = pred.shape
    mask = mask_waypoints.view(B, M, 1).expand(B, M, D)
    se = (pred - target).pow(2)
    se = se[mask]
    return se.mean() if se.numel() > 0 else pred.new_tensor(0.0)


def _compute_total_grad_norm(parameters, norm_type: float = 2.0) -> float:
    parameters = [p for p in parameters if p.grad is not None]
    if len(parameters) == 0:
        return 0.0
    device = parameters[0].grad.device
    if norm_type == float('inf'):
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
        return float(total_norm.item())
    total = torch.zeros([], device=device)
    for p in parameters:
        param_norm = p.grad.detach().data.norm(norm_type)
        total += param_norm.pow(norm_type)
    total = total.pow(1.0 / norm_type)
    return float(total.item())


@dataclass
class TrainConfig:
    train_json: str
    out_dir: str = './ckpts_qwen4'
    n_waypoints: int = 8
    history: int = 31
    llm_name: str = "Qwen/Qwen3-0.6B"
    epochs: int = 1
    batch_size: int = 12
    lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    mixed_precision: bool = True
    vision_feat_dim: int = 1536
    seed: int = 0
    num_workers: int = 4
    # model
    freeze_llm: bool = False
    use_angle_tvi: bool = False
    beta_nav: float = 10.0
    cache_root: Optional[str] = None
    distributed: bool = False
    dist_backend: str = 'nccl'
    alpha_xy: Optional[float] = 2.0
    # logging
    log_every: int = 10
    csv_logging: bool = True
    # trajectory saving
    save_trajectories: bool = False
    traj_subdir: str = 'trajectories'
    # evaluation
    val_json: Optional[str] = None
    eval_every: int = 0
    eval_batches: int = 8
    final_wp_threshold: float = 0.2
    # single-episode evaluation
    episode_json: Optional[str] = None
    episode_eval_every: int = 0
    episode_threshold: float = 0.2
    episode_max_frames: int = 256
    # modeling options
    no_tanh_actions: bool = True
    # checkpoint retention
    max_ckpts: int = 2
    # resume
    resume: bool = False
    resume_ckpt: Optional[str] = None
    # inference
    infer_json: Optional[str] = None
    infer_ckpt: Optional[str] = None
    infer_out: str = './infer_out'
    infer_batches: int = 0
    infer_vis: bool = False
    infer_save_npz: bool = True


def train(cfg: TrainConfig):
    set_seed(cfg.seed)
    torch.backends.cudnn.benchmark = True
    # Distributed setup
    is_cuda = torch.cuda.is_available()
    use_ddp = bool(cfg.distributed) and is_cuda
    if use_ddp:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)

        dist.init_process_group(
            backend=cfg.dist_backend,
            init_method='env://',
            device_id=device,  # or device_id=local_rank
        )

        rank = dist.get_rank()
        world_size = dist.get_world_size()

    else:
        device = torch.device('cuda' if is_cuda else 'cpu')
        local_rank = 0
        rank = 0
        world_size = 1

    ds = JsonTrackingDataset(DataConfig(train_json=cfg.train_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
    if rank == 0:
        _dataset_sanity_report(ds, cfg)
    # Auto-compute scalar XY scaling (alpha_xy) from dataset if not provided
    if cfg.alpha_xy is None:
        try:
            import numpy as _np
            vals = []
            sample_n = min(4000, len(ds))
            for i in range(sample_n):
                ex = ds.get_example(i)
                arr = None
                if 'waypoints' in ex:
                    arr = _np.asarray(ex['waypoints'], dtype=_np.float32)
                elif 'actions' in ex:
                    dt = float(ex.get('dt', ds.cfg.default_dt))
                    arr = integrate_actions_to_waypoints(_np.asarray(ex['actions'], dtype=_np.float32), cfg.n_waypoints, dt)
                if arr is None:
                    continue
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[1] >= 2:
                    r = _np.linalg.norm(arr[:, :2], axis=-1)
                    vals.append(r)
            if vals:
                allr = _np.concatenate(vals)
                alpha_est = float(_np.percentile(allr, 95))
                alpha_est = max(alpha_est, 1e-3)
                cfg.alpha_xy = alpha_est
                if rank == 0:
                    print(f"[auto_alpha_xy] alpha_xy set to {alpha_est:.3f} from dataset percentiles ({cfg.train_json})")
        except Exception as _e:
            if rank == 0:
                print(f"[auto_alpha_xy] skipped due to error: {_e}")
    # Auto-detect vision_feat_dim from dataset if needed
    if rank == 0:
        try:
            sample_item = ds[0]
            detected_dim = None
            if 'fine_tokens' in sample_item:
                detected_dim = sample_item['fine_tokens'].shape[-1]
            elif 'coarse_tokens' in sample_item:
                detected_dim = sample_item['coarse_tokens'].shape[-1]
            if detected_dim is not None and detected_dim != cfg.vision_feat_dim:
                print(f"[AUTO_DIM] Detected vision_feat_dim={detected_dim} from dataset (config had {cfg.vision_feat_dim}), updating...")
                cfg.vision_feat_dim = detected_dim
        except Exception as e:
            print(f"[AUTO_DIM] Failed to auto-detect vision_feat_dim: {e}, using config value {cfg.vision_feat_dim}")
    
    # Broadcast the updated vision_feat_dim to all ranks in DDP
    if use_ddp:
        vision_feat_dim_tensor = torch.tensor([cfg.vision_feat_dim], dtype=torch.int32, device=device)
        dist.broadcast(vision_feat_dim_tensor, src=0)
        cfg.vision_feat_dim = int(vision_feat_dim_tensor.item())

    sampler = torch.utils.data.distributed.DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=True) if use_ddp else None
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=(sampler is None), num_workers=cfg.num_workers,
                    pin_memory=True, collate_fn=collate_batch, sampler=sampler)

    model = OpenTrackVLA(
        ModelConfig(
            llm_name=cfg.llm_name,
            freeze_llm=cfg.freeze_llm,
            n_waypoints=cfg.n_waypoints,
            beta_nav=cfg.beta_nav,
            use_angle_tvi=cfg.use_angle_tvi,
            use_tanh_actions=(not cfg.no_tanh_actions),
            alpha_xy=cfg.alpha_xy,
        ),
        vision_feat_dim=cfg.vision_feat_dim,
    ).to(device)
    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Log trainable vs frozen parameters (rank 0 only)
    if rank == 0:
        try:
            from collections import defaultdict
            model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            total_params = sum(p.numel() for p in model_inspect.parameters())
            trainable_params = sum(p.numel() for p in model_inspect.parameters() if p.requires_grad)
            pct = (trainable_params / max(1, total_params)) * 100.0
            print(f"[PARAMS] total={total_params:,} trainable={trainable_params:,} ({pct:.2f}%)")
            group_counts = defaultdict(lambda: [0, 0])  # [total, trainable]
            for name, p in model_inspect.named_parameters():
                head = name.split('.')[0]
                n = p.numel()
                group_counts[head][0] += n
                if p.requires_grad:
                    group_counts[head][1] += n
            summary = ' '.join([f"{k}:{v[1]}/{v[0]}" for k, v in group_counts.items()])
            print(f"[PARAMS groups] {summary}")
            tn = [n for n, p in model_inspect.named_parameters() if p.requires_grad][:16]
            print(f"[TRAINABLE names (first 16)] {tn}")
        except Exception as _e:
            print(f"[PARAMS] logging skipped due to error: {_e}")

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Mixed precision configuration
    amp_enabled = cfg.mixed_precision and is_cuda
    amp_dtype = torch.bfloat16  # switch to torch.float16 if you want fp16
    scaler = torch.amp.GradScaler('cuda', enabled=(amp_enabled and amp_dtype == torch.float16))

    # Optionally resume from checkpoint
    start_epoch = 0
    step = 0
    if cfg.resume:
        try:
            import glob as _glob
            ckpt_path = cfg.resume_ckpt
            if ckpt_path is None:
                pts = sorted(_glob.glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
                ckpt_path = pts[-1] if pts else None
            if ckpt_path and os.path.exists(ckpt_path):
                obj = torch.load(ckpt_path, map_location=device)
                msd = obj.get('model_state', None)
                if msd:
                    msd = _cleanup_state_dict_keys(msd)
                    model_to_load = model.module if isinstance(
                        model, torch.nn.parallel.DistributedDataParallel
                    ) else model
                    model_to_load.load_state_dict(msd, strict=False)
                osd = obj.get('optim_state', None)
                if osd:
                    optim.load_state_dict(osd)
                ssd = obj.get('scaler_state', None)
                if ssd and scaler.is_enabled():
                    scaler.load_state_dict(ssd)
                start_epoch = int(obj.get('epoch', 0))
                step = int(obj.get('step', 0))
                if rank == 0:
                    print(f"[RESUME] Loaded {ckpt_path} | epoch={start_epoch} step={step}")
        except Exception as _e:
            if rank == 0:
                print(f"[RESUME] Skipped due to error: {_e}")

    if rank == 0:
        Path(cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # continue step counter if resumed
    ema_loss: Optional[float] = None
    ema_nav: Optional[float] = None
    last_log_time = time.time()
    epoch_start_time = last_log_time
    for epoch in range(cfg.epochs):
        model.train()
        if use_ddp and sampler is not None:
            sampler.set_epoch(epoch)
        for batch in dl:
            coarse_tokens = batch['coarse_tokens'].to(device)
            coarse_tidx   = batch['coarse_tidx'].to(device)
            fine_tokens   = batch['fine_tokens'].to(device)
            fine_tidx     = batch['fine_tidx'].to(device)
            yaw_hist      = batch['yaw_hist'].to(device)   # (B,H)
            yaw_curr      = batch['yaw_curr'].to(device)   # (B,1)
            gt_wp         = batch['waypoints'].to(device)
            valid_mask    = batch['valid_mask'].to(device)
            instr         = batch['instruction']

            optim.zero_grad(set_to_none=True)
            amp_ctx = torch.autocast(device_type='cuda', dtype=amp_dtype, enabled=amp_enabled) if amp_enabled else nullcontext()
            with amp_ctx:
                tau_pred = model(
                    coarse_tokens, coarse_tidx,
                    fine_tokens, fine_tidx,
                    instr,
                    yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                    yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                )
                # Option A: compute loss in normalized space — divide XY by alpha, keep yaw unscaled
                # Get alpha vector from model (shape: (1,1,D_action)) and clamp to avoid div-by-zero
                model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                alpha_vec = getattr(model_inspect, 'alpha_task', None)
                if alpha_vec is None:
                    # Fallback: no scaling information; compute loss in absolute space
                    pred_norm = tau_pred
                    gt_norm = gt_wp
                else:
                    # Normalize only XY dims (0,1); leave others (e.g., yaw) unscaled
                    pred_norm = tau_pred
                    gt_norm = gt_wp
                    if pred_norm.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                        ax = alpha_vec[..., 0:2].clamp_min(1e-6)
                        pred_norm = pred_norm.clone()
                        gt_norm = gt_norm.clone()
                        pred_norm[..., 0:2] = pred_norm[..., 0:2] / ax
                        gt_norm[..., 0:2] = gt_norm[..., 0:2] / ax
                L_nav = mse_masked(pred_norm, gt_norm, valid_mask)
                L_QA = tau_pred.new_tensor(0.0)
                loss = cfg.beta_nav * L_nav + L_QA
            # Save trajectories per-step (rank 0)
            if rank == 0 and cfg.save_trajectories:
                try:
                    traj_root = os.path.join(cfg.out_dir, cfg.traj_subdir)
                    os.makedirs(traj_root, exist_ok=True)
                    with torch.no_grad():
                        pred_np = tau_pred.detach().float().cpu().numpy()
                        gt_np = gt_wp.detach().float().cpu().numpy()
                        vm_np = valid_mask.detach().cpu().numpy()
                    Bcur = pred_np.shape[0]
                    for bi in range(Bcur):
                        fpath = os.path.join(traj_root, f"ep{epoch:02d}_st{step+1:06d}_b{bi:03d}.npz")
                        # step+1 so filenames are 1-indexed and align with printed step after increment
                        np.savez_compressed(
                            fpath,
                            pred=pred_np[bi],
                            gt=gt_np[bi],
                            valid_mask=vm_np[bi],
                            instruction=instr[bi],
                            epoch=epoch,
                            step=step+1
                        )
                except Exception:
                    pass
            scaler.scale(loss).backward()
            grad_norm_before = 0.0
            if cfg.grad_clip is not None:
                scaler.unscale_(optim)
                grad_norm_before = _compute_total_grad_norm([p for p in model.parameters() if getattr(p, 'grad', None) is not None])
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optim); scaler.update()

            step += 1
            if rank == 0 and (step % cfg.log_every == 0):
                now = time.time()
                dt = now - last_log_time
                last_log_time = now
                B = coarse_tokens.size(0)
                lr = optim.param_groups[0]['lr']
                with torch.no_grad():
                    tp = tau_pred.detach().float()
                    gwp = gt_wp.detach().float()
                    vm = valid_mask.detach().float()
                    # Scale predictions to absolute units for logging if alpha is available
                    tp_abs = tp
                    try:
                        model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and alpha_vec.size(-1) >= tp.size(-1):
                            av = alpha_vec.to(tp.device, tp.dtype)
                            tp_abs = tp * av
                    except Exception:
                        pass
                    pred_mean = tp_abs.mean().item()
                    pred_std = tp_abs.std().item()
                    pred_absmax = tp_abs.abs().max().item()
                    gt_mean = gwp.mean().item()
                    gt_std = gwp.std().item()
                    mask_cov = vm.mean().item()
                    mse_total = (tp_abs - gwp).pow(2).mean().item()
                    ad = tp_abs.size(-1)
                    per_dim_mse = []
                    for d in range(min(4, ad)):
                        per_dim_mse.append(float((tp_abs[..., d] - gwp[..., d]).pow(2).mean().item()))
                loss_val = float(loss.detach().item())
                nav_val = float(L_nav.detach().item())
                ema_loss = loss_val if ema_loss is None else (0.98 * ema_loss + 0.02 * loss_val)
                ema_nav = nav_val if ema_nav is None else (0.98 * ema_nav + 0.02 * nav_val)

                mem_alloc_mb = mem_peak_mb = 0.0
                if torch.cuda.is_available():
                    try:
                        torch.cuda.synchronize()
                    except Exception:
                        pass
                    mem_alloc_mb = torch.cuda.memory_allocated(device) / (1024**2)
                    mem_peak_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

                msg_lines = [
                    f"epoch {epoch} step {step} | lr={lr:.2e} | loss={loss_val:.4f} (ema {ema_loss:.4f}) | L_nav={nav_val:.4f} (ema {ema_nav:.4f})",
                    f"  mask_cov={mask_cov:.3f} | grad_norm_preclip={grad_norm_before:.3f} | step_time={dt:.3f}s | throughput={B/dt:.2f} it/s",
                    f"  pred(mean={pred_mean:.3f}, std={pred_std:.3f}, absmax={pred_absmax:.3f}) | gt(mean={gt_mean:.3f}, std={gt_std:.3f})",
                    f"  mse_total={mse_total:.5f} | per_dim_mse={per_dim_mse} | mem_alloc={mem_alloc_mb:.1f}MB peak={mem_peak_mb:.1f}MB"
                ]
                print("\n".join(msg_lines), flush=True)

                # Debug preview: print GT and Pred waypoints (absolute), and normalized XY if alpha is set
                """
                try:
                    import numpy as _np
                    model_inspect = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    alpha_vec = getattr(model_inspect, 'alpha_task', None)
                    pred_abs_b0 = tp[0].detach().cpu().numpy()
                    gt_abs_b0 = gwp[0].detach().cpu().numpy()
                    print("[WAYPOINTS abs][b0] pred=", _np.array2string(pred_abs_b0, precision=3, floatmode='fixed'))
                    print("[WAYPOINTS abs][b0]   gt=", _np.array2string(gt_abs_b0, precision=3, floatmode='fixed'))
                    if alpha_vec is not None and pred_abs_b0.shape[1] >= 2 and alpha_vec.size(-1) >= 2:
                        ax = alpha_vec[0, 0, 0:2].clamp_min(1e-6).detach().float().cpu()
                        pred_n_xy = (tp[0, :, 0:2].detach().cpu() / ax).numpy()
                        gt_n_xy = (gwp[0, :, 0:2].detach().cpu() / ax).numpy()
                        print(f"[WAYPOINTS norm][b0] alpha_xy={ax.numpy().tolist()} pred_xy=", _np.array2string(pred_n_xy, precision=3, floatmode='fixed'))
                        print("[WAYPOINTS norm][b0]   gt_xy=", _np.array2string(gt_n_xy, precision=3, floatmode='fixed'))
                except Exception:
                    pass
                """
                if cfg.csv_logging:
                    csv_path = os.path.join(cfg.out_dir, 'train_log.csv')
                    header = [
                        'epoch','step','lr','loss','loss_ema','L_nav','L_nav_ema','mask_cov',
                        'grad_norm_preclip','step_time','throughput_it_per_s','pred_mean','pred_std','pred_absmax',
                        'gt_mean','gt_std','mse_total','mem_alloc_mb','mem_peak_mb'
                    ] + [f'mse_dim_{i}' for i in range(len(per_dim_mse))]
                    write_header = not os.path.exists(csv_path)
                    try:
                        with open(csv_path, 'a', newline='') as f:
                            w = csv.writer(f)
                            if write_header:
                                w.writerow(header)
                            row = [
                                epoch, step, lr, loss_val, ema_loss, nav_val, ema_nav, mask_cov,
                                grad_norm_before, dt, (B/dt), pred_mean, pred_std, pred_absmax,
                                gt_mean, gt_std, mse_total, mem_alloc_mb, mem_peak_mb
                            ] + per_dim_mse
                            w.writerow(row)
                    except Exception:
                        pass

                # Visualize GT vs Pred on current images 
                if rank == 0 and step % 100 == 0:
                    try:
                        vis_dir = os.path.join(cfg.out_dir, 'vis')
                        os.makedirs(vis_dir, exist_ok=True)
                        with torch.no_grad():
                            # Ensure predictions are in absolute units for visualization
                            pred_draw = tau_pred.detach().float()
                            pred_np = pred_draw.cpu().numpy()
                            gt_np = gwp.detach().float().cpu().numpy()
                            cur_paths = batch.get('current_path', [])
                        Bcur = pred_np.shape[0]
                        for bi in range(min(Bcur, 4)):
                            cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                            if cur_path is None or (not os.path.exists(cur_path)):
                                continue
                            pil_img = Image.open(cur_path).convert('RGB')
                            draw = ImageDraw.Draw(pil_img)
                            w, h = pil_img.size
                            base_x = w // 2
                            base_y = int(h * 0.86)
                            def to_pxxy(traj):
                                pts = []
                                for i in range(min(traj.shape[0], 64)):
                                    x, y = float(traj[i, 0]), float(traj[i, 1])
                                    # y is left-positive in robot frame → screen x grows to right ⇒ subtract
                                    px = base_x - int(y * 120)
                                    py = base_y - int(x * 120)
                                    pts.append((px, py))
                                return pts
                            pts_pred = to_pxxy(pred_np[bi])
                            pts_gt   = to_pxxy(gt_np[bi])
                            # outline
                            for seq, color in ((pts_gt, (0,0,0)), (pts_pred, (0,0,0))):
                                for i2 in range(1, len(seq)):
                                    draw.line([seq[i2-1], seq[i2]], fill=color, width=10)
                            # body
                            for i2 in range(1, len(pts_gt)):
                                draw.line([pts_gt[i2-1], pts_gt[i2]], fill=(255, 200, 0), width=6)
                            for i2 in range(1, len(pts_pred)):
                                draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                            # start points
                            if pts_gt:
                                r0 = 6
                                sx, sy = pts_gt[0]
                                draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(255,255,255))
                            if pts_pred:
                                r0 = 6
                                sx, sy = pts_pred[0]
                                draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                            try:
                                from inspect import isclass
                                model_cfg = (model.module.cfg if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model.cfg)
                            except Exception:
                                pass
                            out_path = os.path.join(vis_dir, f"ep{epoch:02d}_st{step:06d}_b{bi:03d}.jpg")
                            pil_img.save(out_path)
                            print(f"[VIS] saved {out_path}")
                    except Exception:
                        pass

            if step % 100 == 0 and rank == 0:
                ckpt = os.path.join(cfg.out_dir, f"model_epoch{epoch:02d}_step{step:06d}.pt")
                # Always save the *underlying* model (so no 'module.' prefix)
                model_to_save = model.module if isinstance(
                    model, torch.nn.parallel.DistributedDataParallel
                ) else model

                torch.save(
                {
                    'epoch': epoch,
                    'model_state': model_to_save.state_dict(),
                    'optim_state': optim.state_dict(),
                    'scaler_state': (scaler.state_dict() if scaler.is_enabled() else None),
                    'config': cfg.__dict__,
                    'step': step,
                },
                ckpt,
                )
                try:
                    from glob import glob
                    pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p), reverse=True)
                    if cfg.max_ckpts is not None and cfg.max_ckpts > 0 and len(pts) > cfg.max_ckpts:
                        for old in pts[cfg.max_ckpts:]:
                            try:
                                os.remove(old)
                            except Exception:
                                pass
                except Exception:
                    pass
                if torch.cuda.is_available():
                    try:
                        torch.cuda.reset_peak_memory_stats(device)
                    except Exception:
                        pass

            # Periodic evaluation on a small held-out set (rank 0 only)
            if (cfg.eval_every and (step % cfg.eval_every == 0) and rank == 0 and cfg.val_json):
                try:
                    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_eval.eval()
                    with torch.inference_mode():
                        vds = JsonTrackingDataset(DataConfig(train_json=cfg.val_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
                        vdl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(2, cfg.num_workers), pin_memory=True, collate_fn=collate_batch)
                        total_mse = 0.0
                        total_count = 0
                        final_errors: List[float] = []
                        hits = 0
                        max_batches = max(1, cfg.eval_batches)
                        bdone = 0
                        for vbatch in vdl:
                            coarse_tokens = vbatch['coarse_tokens'].to(device)
                            coarse_tidx   = vbatch['coarse_tidx'].to(device)
                            fine_tokens   = vbatch['fine_tokens'].to(device)
                            fine_tidx     = vbatch['fine_tidx'].to(device)
                            yaw_hist      = vbatch['yaw_hist'].to(device)
                            yaw_curr      = vbatch['yaw_curr'].to(device)
                            gt_wp         = vbatch['waypoints'].to(device)
                            valid_mask    = vbatch['valid_mask'].to(device)
                            instr         = vbatch['instruction']

                            pred = model_eval(
                                coarse_tokens, coarse_tidx,
                                fine_tokens, fine_tidx,
                                instr,
                                yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                                yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                            )
                            # masked MSE in normalized space (Option A): divide XY by alpha
                            model_inspect = model_eval
                            alpha_vec = getattr(model_inspect, 'alpha_task', None)
                            if alpha_vec is not None and pred.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6)
                                pred_n = pred.clone()
                                gt_n = gt_wp.clone()
                                pred_n[..., 0:2] = pred_n[..., 0:2] / ax
                                gt_n[..., 0:2] = gt_n[..., 0:2] / ax
                                mse = mse_masked(pred_n, gt_n, valid_mask).item()
                            else:
                                mse = mse_masked(pred, gt_wp, valid_mask).item()
                            total_mse += mse * pred.size(0)
                            total_count += pred.size(0)
                            # final waypoint EPE and hit rate (first two dims assumed spatial)
                            pred_xy = pred[:, -1, :2].float()
                            gt_xy = gt_wp[:, -1, :2].float()
                            epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (B,)
                            final_errors.extend(epe.cpu().tolist())
                            hits += (epe <= cfg.final_wp_threshold).sum().item()

                            bdone += 1
                            if bdone >= max_batches:
                                break
                    mean_mse = total_mse / max(1, total_count)
                    if len(final_errors) > 0:
                        import numpy as _np
                        epe_mean = float(_np.mean(final_errors))
                        epe_median = float(_np.median(final_errors))
                        hit_rate = float(hits / len(final_errors))
                    else:
                        epe_mean = epe_median = hit_rate = float('nan')
                    print(f"[VAL] step {step} | masked_MSE={mean_mse:.5f} | final_EPE_mean={epe_mean:.4f} | final_EPE_median={epe_median:.4f} | hit@{cfg.final_wp_threshold}={hit_rate:.3f}", flush=True)
                except Exception as _e:
                    print(f"[VAL] evaluation skipped due to error: {_e}")
                finally:
                    model.train()

            # Single-episode evaluation (rank 0 only)
            if (cfg.episode_eval_every and (step % cfg.episode_eval_every == 0) and rank == 0 and cfg.episode_json):
                try:
                    model_eval = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    model_eval.eval()
                    with torch.inference_mode():
                        ds_tmp = JsonTrackingDataset(DataConfig(train_json=cfg.episode_json, n_waypoints=cfg.n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
                        if len(ds_tmp) == 0:
                            raise RuntimeError('episode_json produced no examples')
                        max_frames = min(cfg.episode_max_frames, len(ds_tmp))
                        epe_list: List[float] = []
                        hits = 0
                        for i in range(max_frames):
                            item = ds_tmp[i]
                            coarse_tokens = item['coarse_tokens'].unsqueeze(0).to(device)
                            coarse_tidx   = item['coarse_tidx'].unsqueeze(0).to(device)
                            fine_tokens   = item['fine_tokens'].unsqueeze(0).to(device)
                            fine_tidx     = item['fine_tidx'].unsqueeze(0).to(device)
                            yaw_hist      = item['yaw_hist'].unsqueeze(0).to(device)
                            yaw_curr      = item['yaw_curr'].unsqueeze(0).to(device)
                            gt_wp         = item['waypoints'].unsqueeze(0).to(device)
                            instr         = [item['instruction']]

                            pred = model_eval(
                                coarse_tokens, coarse_tidx,
                                fine_tokens, fine_tidx,
                                instr,
                                yaw_hist=yaw_hist if cfg.use_angle_tvi else None,
                                yaw_curr=yaw_curr if cfg.use_angle_tvi else None
                            )
                            pred_xy = pred[:, -1, :2].float()
                            gt_xy = gt_wp[:, -1, :2].float()
                            epe = torch.linalg.norm(pred_xy - gt_xy, dim=-1)  # (1,)
                            e = float(epe.item())
                            epe_list.append(e)
                            if e <= cfg.episode_threshold:
                                hits += 1
                        if len(epe_list) > 0:
                            import numpy as _np
                            epe_mean = float(_np.mean(epe_list))
                            epe_median = float(_np.median(epe_list))
                            follow_rate = float(hits / len(epe_list))
                        else:
                            epe_mean = epe_median = follow_rate = float('nan')
                    print(f"[EPISODE] step {step} | frames={len(epe_list)} | EPE_mean={epe_mean:.4f} | EPE_median={epe_median:.4f} | follow@{cfg.episode_threshold}={follow_rate:.3f}", flush=True)
                except Exception as _e:
                    print(f"[EPISODE] evaluation skipped due to error: {_e}")
                finally:
                    model.train()

    # Optional inference after training when requested (rank 0 only)
    if rank == 0 and cfg.infer_json:
        try:
            _run_inference(cfg)
        except Exception as _e:
            print(f"[INFER] failed: {_e}")

    if use_ddp:
        dist.barrier()
        dist.destroy_process_group()

    if rank == 0:
        print(f"[TRAIN] Finished all epochs. last_step={step}")


# ----------------------- Inference -----------------------

@torch.inference_mode()
def _run_inference(cfg: TrainConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Resolve checkpoint
    ckpt_path = cfg.infer_ckpt
    try:
        if ckpt_path is None:
            from glob import glob
            pts = sorted(glob(os.path.join(cfg.out_dir, 'model_epoch*.pt')), key=lambda p: os.path.getmtime(p))
            ckpt_path = pts[-1] if pts else None
    except Exception:
        ckpt_path = None
    if ckpt_path is None or (not os.path.exists(ckpt_path)):
        raise FileNotFoundError(f"No checkpoint found for inference (looked at --infer_ckpt and {cfg.out_dir})")

    # Load checkpoint config
    obj = torch.load(ckpt_path, map_location=device)
    ck = obj if isinstance(obj, dict) else {}
    ck_cfg = ck.get('config', {})

    # Build model config mirroring training
    n_waypoints = int(ck_cfg.get('n_waypoints', cfg.n_waypoints))
    use_angle_tvi = bool(ck_cfg.get('use_angle_tvi', cfg.use_angle_tvi))
    no_tanh_actions = bool(ck_cfg.get('no_tanh_actions', cfg.no_tanh_actions))
    vision_feat_dim = int(ck_cfg.get('vision_feat_dim', cfg.vision_feat_dim))
    alpha_xy        = ck_cfg.get('alpha_xy', getattr(cfg, 'alpha_xy', None))
    llm_name        = str(ck_cfg.get('llm_name', getattr(cfg, 'llm_name', "Qwen/Qwen3-8B")))
    print (llm_name)

    model = OpenTrackVLA(
        ModelConfig(
            llm_name=llm_name,
            n_waypoints=n_waypoints,
            beta_nav=float(ck_cfg.get('beta_nav', cfg.beta_nav)),
            use_angle_tvi=use_angle_tvi,
            use_tanh_actions=(not no_tanh_actions),
            alpha_xy=alpha_xy,
        ),
        vision_feat_dim=vision_feat_dim,
    ).to(device).eval()
    msd = ck.get('model_state', None)
    if msd:
        model.load_state_dict(msd, strict=False)

    # Data
    if cfg.infer_json is None:
        raise ValueError('--infer_json is required for inference')
    vds = JsonTrackingDataset(DataConfig(train_json=cfg.infer_json, n_waypoints=n_waypoints, history=cfg.history, cache_root=cfg.cache_root))
    vdl = DataLoader(vds, batch_size=cfg.batch_size, shuffle=False, num_workers=min(2, cfg.num_workers), pin_memory=True, collate_fn=collate_batch)

    # Output dirs
    os.makedirs(cfg.infer_out, exist_ok=True)
    vis_dir = os.path.join(cfg.infer_out, 'vis')
    npz_dir = os.path.join(cfg.infer_out, 'npz')
    if cfg.infer_vis:
        os.makedirs(vis_dir, exist_ok=True)
    if cfg.infer_save_npz:
        os.makedirs(npz_dir, exist_ok=True)

    batches_limit = max(0, int(cfg.infer_batches))
    bdone = 0
    for bidx, batch in enumerate(vdl):
        coarse_tokens = batch['coarse_tokens'].to(device)
        coarse_tidx   = batch['coarse_tidx'].to(device)
        fine_tokens   = batch['fine_tokens'].to(device)
        fine_tidx     = batch['fine_tidx'].to(device)
        yaw_hist      = batch['yaw_hist'].to(device)
        yaw_curr      = batch['yaw_curr'].to(device)
        instr         = batch['instruction']

        pred = model(
            coarse_tokens, coarse_tidx,
            fine_tokens, fine_tidx,
            instr,
            yaw_hist=yaw_hist if use_angle_tvi else None,
            yaw_curr=yaw_curr if use_angle_tvi else None
        )  # absolute space (alpha applied inside model)

        # Save NPZ
        if cfg.infer_save_npz:
            try:
                with torch.no_grad():
                    pred_np = pred.detach().float().cpu().numpy()
                Bcur = pred_np.shape[0]
                for bi in range(Bcur):
                    fpath = os.path.join(npz_dir, f"b{bidx:06d}_i{bi:03d}.npz")
                    np.savez_compressed(
                        fpath,
                        pred=pred_np[bi],
                        instruction=instr[bi],
                        current_path=(batch.get('current_path', [''])[bi] if isinstance(batch.get('current_path', []), list) else '')
                    )
            except Exception:
                pass

        # Visualization in absolute space (consistent with training vis and deployment)
        if cfg.infer_vis:
            try:
                with torch.no_grad():
                    # Ensure predictions are in absolute units for visualization
                    pred_draw = pred.detach().float()
                    try:
                        model_inspect = model
                        alpha_vec = getattr(model_inspect, 'alpha_task', None)
                        if alpha_vec is not None and pred_draw.size(-1) >= 2 and alpha_vec.size(-1) >= 2:
                            max_xy = pred_draw[..., 0:2].abs().max().item()
                            if max_xy <= 1.5:
                                ax = alpha_vec[..., 0:2].clamp_min(1e-6).to(pred_draw.device, pred_draw.dtype)
                                pred_draw = pred_draw.clone()
                                pred_draw[..., 0:2] = pred_draw[..., 0:2] * ax
                    except Exception:
                        pass
                    pred_np = pred_draw.cpu().numpy()
                cur_paths = batch.get('current_path', [])
                Bcur = pred_np.shape[0]
                for bi in range(min(Bcur, 4)):
                    cur_path = cur_paths[bi] if isinstance(cur_paths, list) and bi < len(cur_paths) else None
                    if cur_path is None or (not os.path.exists(cur_path)):
                        continue
                    pil_img = Image.open(cur_path).convert('RGB')
                    draw = ImageDraw.Draw(pil_img)
                    w, h = pil_img.size
                    base_x = w // 2
                    base_y = int(h * 0.86)
                    def to_pxxy(traj):
                        pts = []
                        for i in range(min(traj.shape[0], 64)):
                            x, y = float(traj[i, 0]), float(traj[i, 1])
                            px = base_x - int(y * 120)
                            py = base_y - int(x * 120)
                            pts.append((px, py))
                        return pts
                    pts_pred = to_pxxy(pred_np[bi])
                    for i2 in range(1, len(pts_pred)):
                        draw.line([pts_pred[i2-1], pts_pred[i2]], fill=(0, 255, 200), width=6)
                    if pts_pred:
                        r0 = 6
                        sx, sy = pts_pred[0]
                        draw.ellipse([sx-r0, sy-r0, sx+r0, sy+r0], fill=(0,255,0))
                    out_path = os.path.join(vis_dir, f"b{bidx:06d}_i{bi:03d}.jpg")
                    pil_img.save(out_path)
            except Exception:
                pass

        bdone += 1
        if batches_limit and bdone >= batches_limit:
            break

    print(f"[INFER] Done. Outputs under {cfg.infer_out}")


# ----------------------- CLI -----------------------

def parse_args() -> TrainConfig:
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_json', type=str, required=True)
    ap.add_argument('--out_dir', type=str, default='./ckpts')
    ap.add_argument('--n_waypoints', type=int, default=8)
    ap.add_argument('--history', type=int, default=31)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch_size', type=int, default=2)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--grad_clip', type=float, default=1.0)
    ap.add_argument('--mixed_precision', action='store_true')
    ap.add_argument('--vision_feat_dim', type=int, default=1536)
    ap.add_argument('--freeze_llm', action='store_true',
                    help='Freeze the pretrained language model; projector and waypoint head remain trainable')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--cache_root', type=str, default=None, help='Root folder containing vision_cache; defaults to <train_json>/vision_cache')
    ap.add_argument('--distributed', action='store_true')
    ap.add_argument('--dist_backend', type=str, default='nccl')
    ap.add_argument('--use_angle_tvi', action='store_true')  # default False
    ap.add_argument('--alpha_xy', type=float, default=2.0, help='Scalar to scale only XY targets; yaw unscaled')
    ap.add_argument('--beta_nav',      type=float, default=10.0)
    # logging / saving
    ap.add_argument('--log_every', type=int, default=10)
    ap.add_argument('--csv_logging', action='store_true')
    ap.add_argument('--save_trajectories', action='store_true')
    ap.add_argument('--traj_subdir', type=str, default='trajectories')
    # evaluation
    ap.add_argument('--eval_every', type=int, default=0, help='Evaluate every N steps (0 = disabled)')
    ap.add_argument('--eval_batches', type=int, default=8, help='Number of validation batches per eval')
    ap.add_argument('--final_wp_threshold', type=float, default=0.2, help='Hit threshold (units of target XY)')
    # modeling
    ap.add_argument('--no_tanh_actions', action=argparse.BooleanOptionalAction, default=True, help='Remove tanh cap in action head (unbounded outputs)')
    # checkpoint retention
    ap.add_argument('--max_ckpts', type=int, default=3, help='Keep at most this many checkpoints')
    # resume
    ap.add_argument('--resume', action='store_true', help='Resume from the latest checkpoint in out_dir (or --resume_ckpt)')
    ap.add_argument('--resume_ckpt', type=str, default=None, help='Path to a specific checkpoint to resume from')
    # inference
    ap.add_argument('--infer_json', type=str, default=None, help='Run inference on this dataset (json/jsonl/dir)')
    ap.add_argument('--infer_ckpt', type=str, default=None, help='Checkpoint to load for inference (defaults to latest in out_dir)')
    ap.add_argument('--infer_out', type=str, default='./infer_out', help='Output directory for inference results')
    ap.add_argument('--infer_batches', type=int, default=0, help='Limit number of batches to run at inference (0 = all)')
    ap.add_argument('--infer_vis', action='store_true', help='Save visualization images during inference')
    ap.add_argument('--infer_save_npz', action='store_true', help='Save npz predictions during inference')
    # single-episode evaluation
    ap.add_argument('--episode_json', type=str, default=None, help='JSON/JSONL path for a single episode sequence to evaluate')
    ap.add_argument('--episode_eval_every', type=int, default=0, help='Evaluate the episode every N steps (0 = disabled)')
    ap.add_argument('--episode_threshold', type=float, default=0.2, help='Following threshold radius for episode eval')
    ap.add_argument('--episode_max_frames', type=int, default=256, help='Maximum frames to evaluate in the episode')
    ap.add_argument('--llm_name', type=str, default='Qwen/Qwen3-0.6B', help='HuggingFace model id for the LLM backbone')

    args = ap.parse_args()
    return TrainConfig(**vars(args))


if __name__ == '__main__':
    cfg = parse_args()
    # Inference-only mode: if infer_json is provided and epochs==0, run inference with a loaded model
    if cfg.infer_json and cfg.epochs == 0:
        _run_inference(cfg)
    else:
        train(cfg)
