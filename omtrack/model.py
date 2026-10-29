#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""OmTrackVLA network: projected vision tokens + instruction -> 8 robot-frame waypoints.

Upstream code called this OpenTrackVLA; the released checkpoint's config.json still
lists "OpenTrackVLAForWaypoint" and model_type "navfom". Loading goes through
omtrack.hf with explicit classes, so only model_type has to keep matching.

Single definition used by inference (omtrack.hf), training (training/train.py) and
the simulator agent. Data loading lives in omtrack/dataset.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


# ----------------------- TVI + projector + planner -----------------------

class TVIEmbedder(nn.Module):
    """Temporal-Viewpoint Indicator with token insertion.
    - make_time_token(t, kind_id, view_id)
    - make_angle_token(theta, kind_id, view_id) -> uses [sinθ, cosθ] projection
    kind_id: 0 = coarse/history, 1 = fine/current.
    """
    def __init__(self, d_model: int, max_time: int = 4096, max_views: int = 1):
        super().__init__()
        self.time_emb   = nn.Embedding(max_time, d_model)
        self.view_emb   = nn.Embedding(max_views, d_model)
        self.kind_emb   = nn.Embedding(2, d_model)
        self.angle_proj = nn.Linear(2, d_model)
        # Retain the checkpoint's bbox branch for strict state loading. The
        # current trainer forward path does not consume bbox features.
        # Unused by forward; kept so the released checkpoint loads with strict=True.
        self.bbox_proj  = nn.Linear(4, d_model)

    def make_time_token(self, t_scalar: int, kind_id: int, view_id: int = 0,
                        device: Optional[torch.device] = None) -> torch.Tensor:
        tok = self.time_emb.weight[t_scalar] + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok.to(device) if device is not None else tok

    def make_angle_token(self, theta: float, kind_id: int, view_id: int = 0,
                         device: Optional[torch.device] = None) -> torch.Tensor:
        # project [sinθ, cosθ] into d_model
        theta = (theta + math.pi) % (2*math.pi) - math.pi
        sincos = torch.tensor([math.sin(theta), math.cos(theta)],
                              dtype=self.angle_proj.weight.dtype,
                              device=device)
        ang = F.linear(sincos, self.angle_proj.weight, self.angle_proj.bias)
        tok = ang + self.view_emb.weight[view_id] + self.kind_emb.weight[kind_id]
        return tok


class CrossModalityProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim), nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, x): return self.net(x)


class PlannerHead3L(nn.Module):
    """Three-layer MLP A_θ mapping E_A^T → normalized waypoints â ∈ [-1,1]."""
    def __init__(self, d_model: int, n_waypoints: int, action_dims: int, use_tanh: bool = True):
        super().__init__()
        hid = d_model * 2
        out_dim = n_waypoints * action_dims
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, out_dim)
        )
        self.nw = n_waypoints
        self.ad = action_dims
        self.use_tanh = use_tanh
    def forward(self, act_h: torch.Tensor) -> torch.Tensor:
        y = self.mlp(act_h)
        if self.use_tanh:
            y = torch.tanh(y)                 # bound to [-1,1]
        return y.view(-1, self.nw, self.ad)   # (B, M, D_action)


# ----------------------- Model -----------------------

@dataclass
class ModelConfig:
    llm_name: str = "Qwen/Qwen3-0.6B"
    # Defaults suit inference. Training passes freeze_llm/alpha_xy explicitly from
    # TrainConfig, and the released checkpoint's config sets alpha_xy=2.0.
    freeze_llm: bool = True
    n_waypoints: int = 8
    max_time: int = 4096
    beta_nav: float = 10.0
    use_angle_tvi: bool = False     # single-cam default: off
    # Action/target configuration
    use_tanh_actions: bool = True   # allow removing tanh cap via flag
    alpha_xy: Optional[float] = None  # Optional scalar to scale XY only; yaw stays unscaled


class OmTrackVLA(nn.Module):
    def __init__(self, cfg: ModelConfig, vision_feat_dim: int):
        super().__init__()
        self.cfg = cfg
        self.llm = AutoModel.from_pretrained(cfg.llm_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else None)
        self.llm.requires_grad_(not cfg.freeze_llm)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)
        self.D = self.llm.config.hidden_size
        self.proj = CrossModalityProjector(vision_feat_dim, self.D)
        # Always keep projector trainable regardless of LLM freeze
        self.proj.requires_grad_(True)
        self.tvi = TVIEmbedder(self.D, max_time=cfg.max_time)
        self.act_token = nn.Parameter(torch.zeros(1, 1, self.D))
        nn.init.normal_(self.act_token, std=0.02)
        # Determine target/action dimensionality
        action_dims = 3
        self.action_dims = action_dims
        self.planner = PlannerHead3L(self.D, cfg.n_waypoints, action_dims, use_tanh=cfg.use_tanh_actions)
        # Always keep planner trainable regardless of LLM freeze
        self.planner.requires_grad_(True)
        # Freeze unused TVI heads so DDP doesn't expect grads for them
        if not cfg.use_angle_tvi:
            for p in self.tvi.angle_proj.parameters():
                p.requires_grad = False
        for p in self.tvi.bbox_proj.parameters():
            p.requires_grad = False
        if cfg.alpha_xy is not None:
            vec = [1.0] * action_dims
            if action_dims >= 2:
                vec[0] = cfg.alpha_xy
                vec[1] = cfg.alpha_xy
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        else:
            vec = [1.0] * action_dims
            alpha = torch.tensor(vec, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("alpha_task", alpha)

    def _embed_text(self, instructions: List[str], device):
        tok = self.tokenizer(instructions, return_tensors='pt', padding=True, truncation=True, max_length=128)
        tok = {k: v.to(device) for k, v in tok.items()}
        emb = self.llm.get_input_embeddings()(tok['input_ids'])
        return emb, tok['attention_mask']

    def _interleave_tvi(self, tokens: torch.Tensor, t_idx: torch.Tensor, kind_id: int,
                        yaw_per_frame: Optional[torch.Tensor] = None, use_angle: bool = False) -> torch.Tensor:
        """Insert TVI time (and optional angle) token before each frame's token block.
        tokens: (B, N, D_llm); t_idx: (B, N) frame ids; yaw_per_frame: (B, F) or None.
        Returns: (B, N + (1 or 2)*F, D_llm)
        """
        B, N, D = tokens.shape
        out_list = []
        for b in range(B):
            tb = t_idx[b]
            xb = tokens[b]
            items = []
            i = 0
            fcount = 0
            while i < N:
                tcur = int(tb[i].item())
                j = i + 1
                while j < N and int(tb[j].item()) == tcur:
                    j += 1
                time_tok = self.tvi.make_time_token(tcur, kind_id, device=xb.device).unsqueeze(0)
                items.append(time_tok)
                if use_angle:
                    theta = 0.0
                    if yaw_per_frame is not None and fcount < yaw_per_frame.size(1):
                        theta = float(yaw_per_frame[b, fcount].item())
                    angle_tok = self.tvi.make_angle_token(theta, kind_id, device=xb.device).unsqueeze(0)
                    items.append(angle_tok)
                items.append(xb[i:j])
                i = j
                fcount += 1
            out_list.append(torch.cat(items, dim=0))
        return torch.stack(out_list, dim=0)

    def forward(self,
                coarse_tokens, coarse_tidx,
                fine_tokens, fine_tidx,
                instructions,
                yaw_hist: Optional[torch.Tensor] = None,
                yaw_curr: Optional[torch.Tensor] = None):
        device = next(self.parameters()).device
        B = coarse_tokens.size(0)
        # project to LLM space
        vis_c = self.proj(coarse_tokens.to(device))   # (B, Nc, D)
        vis_f = self.proj(fine_tokens.to(device))     # (B, Nf, D)
        # insert TVI tokens per frame
        vis_c = self._interleave_tvi(
            vis_c, coarse_tidx.to(device), kind_id=0,
            yaw_per_frame=yaw_hist, use_angle=self.cfg.use_angle_tvi
        )
        vis_f = self._interleave_tvi(
            vis_f, fine_tidx.to(device), kind_id=1,
            yaw_per_frame=yaw_curr, use_angle=self.cfg.use_angle_tvi
        )
        txt_emb, txt_mask = self._embed_text(instructions, device)  # (B, Ltxt, D), (B, Ltxt)
        extra = []
        act = self.act_token.expand(B, 1, -1)
        pieces = [txt_emb] + ([extra[0]] if extra else []) + [vis_c, vis_f, act]
        seq = torch.cat(pieces, dim=1).to(self.llm.dtype)
        extra_len = (extra[0].size(1) if extra else 0)
        attn = torch.cat([
            txt_mask,
            torch.ones(B, extra_len + vis_c.size(1) + vis_f.size(1) + 1, dtype=torch.long, device=device)  # +1 for ACT
        ], dim=1)
        out = self.llm(inputs_embeds=seq, attention_mask=attn, output_hidden_states=True, use_cache=False)
        h_act = out.last_hidden_state[:, -1, :]        # E_A^T (ACT is last)
        # Cast to float32 to match planner LayerNorm/Linear parameter dtype
        h_act = h_act.float()
        a_hat = self.planner(h_act)                # normalized [-1,1]
        tau_pred = a_hat * self.alpha_task             # absolute units
        return tau_pred
