#!/usr/bin/env python3

import os
import sys
import bpy
import math
import json
import torch
import utils3d
import argparse
import numpy as np
from typing import *
import open3d as o3d
from tqdm import tqdm
from PIL import Image
from queue import Queue
from pathlib import Path
from mathutils import Vector
from easydict import EasyDict as edict
from types import MethodType
from typing import Dict, Tuple, Optional
import torch.nn.functional as F
from torchvision import transforms
from concurrent.futures import ThreadPoolExecutor

import trellis.modules.sparse as sp
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils

from scipy.ndimage import label, generate_binary_structure
from plyfile import PlyData, PlyElement

# Import voxel encoding function
from inference.voxel_encoding import voxel_grid_to_ply

torch.set_grad_enabled(False)


def sample_sparse_structure(
    self,
    cond: dict,
    num_samples: int = 1,
    sampler_params: dict = {},
) -> torch.Tensor:

    flow_model = self.models['sparse_structure_flow_model']
    reso = flow_model.resolution
    noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)
    sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}
    z_s = self.sparse_structure_sampler.sample(
        flow_model,
        noise,
        **cond,
        **sampler_params,
        verbose=True
    )
    decoder       = self.models['sparse_structure_decoder']
    decoded_voxel = decoder(z_s)
    output_ply    = os.path.join(sampler_params["output_path"], "edit_voxel.ply")
    voxel_grid_to_ply(
            decoded_voxel, 
            output_ply, 
            threshold=0.5
        )

    coords = torch.argwhere(decoded_voxel>0)[:, [0, 2, 3, 4]].int()
    return coords

# ============================================================================
# 从 flow_euler.py 中提取的 sample 函数
# ============================================================================
@torch.no_grad()
def sample(
    self,
    model,
    noise,
    cond: Optional[Any] = None,
    steps: int = 50,
    rescale_t: float = 1.0,
    verbose: bool = True,
    **kwargs
):
    if "tar_cond" in kwargs:
        n_avg        = kwargs["n_avg"]   if "n_avg"   in kwargs else 5
        st_step      = kwargs["st_step"] if "st_step" in kwargs else 12
        print(f"Sampling with n_avg={n_avg}, st_step={st_step}")
        src_cfg      = kwargs["src_cfg"] if "src_cfg" in kwargs else 1.5
        tar_cfg      = kwargs["tar_cfg"] if "tar_cfg" in kwargs else 5.5
        x_src_packed = kwargs["source_voxel_latent"].to("cuda")
        src_cond     = kwargs["src_cond"]["cond"]
        tar_cond     = kwargs["tar_cond"]["cond"]
        # x_src_packed:  torch.Size([1, 8, 16, 16, 16])
        # cond        :  torch.Size([1, 1370, 1024])

        # 部件1+2 参数
        use_agve     = kwargs.get("use_agve", False)
        warmup_steps = kwargs.get("warmup_steps", 5)    # warmup 阶段 gate=1，不剪枝
        alpha        = kwargs.get("alpha", 6.0)          # 速度差信号权重
        beta         = kwargs.get("beta", 3.0)           # 注意力信号权重
        delta        = kwargs.get("delta", 0.5)          # sigmoid 偏置
        gamma        = kwargs.get("gamma", 0.8)          # EMA 衰减系数
        topk_k       = kwargs.get("topk_k", 50)          # 取变化最大的 DINO patch 数
        vital_layers = kwargs.get("vital_layers", [-1, -2, -3])  # 提取 attention 的 block 索引

        t_seq        = np.linspace(1, 0, steps + 1)
        t_seq        = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs      = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        zt_edit      = x_src_packed.clone()

        src_kwargs   = {
                "neg_cond":     kwargs["neg_cond"],
                "cfg_interval": kwargs["cfg_interval"],
                "cfg_strength": src_cfg
            }

        tar_kwargs   = {
                "neg_cond":     kwargs["neg_cond"],
                "cfg_interval": kwargs["cfg_interval"],
                "cfg_strength": tar_cfg
            }

        # --- 部件1：在循环前预计算 E（DINO patch 差异 top-k 索引）---
        if use_agve:
            c_diff = (tar_cond - src_cond).norm(dim=-1)  # [1, N_dino]
            E      = c_diff[0].topk(topk_k).indices      # [k]，最显著变化的 patch 索引

            num_blocks = len(model.blocks)
            vital_block_indices = [
                li if li >= 0 else num_blocks + li
                for li in vital_layers
            ]

        M        = None   # 运行掩码，部件1+2 共同维护
        edit_idx = 0      # 实际参与编辑的步数

        def _make_attn_hook(buf, idx):
            """注册到 cross_attn 的 forward hook，提取 target 前向的 attention weights。"""
            def _hook(module, args, output):
                if len(args) >= 2:
                    attn = module.get_attn_weights(
                        args[0].detach(), args[1].detach()
                    )  # [B, H, N_3d, N_dino]
                    # CFG 时 batch=2（neg+cond），只取 cond 半边
                    if attn.shape[0] > 1:
                        attn = attn[attn.shape[0] // 2:]
                    buf[idx] = attn
            return _hook

        num_st = -1
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            num_st += 1
            if num_st < st_step:
                continue

            V_delta_avg = torch.zeros_like(x_src_packed)
            attn_accum  = {}  # 累积 vital layers 的 attention（仅 use_agve 时使用）

            for k_avg in range(n_avg):
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                zt_src    = (1 - t) * x_src_packed + t * fwd_noise
                zt_tar    = zt_edit + zt_src - x_src_packed

                _, _, Vt_src = self._get_model_prediction(model, zt_src, t, src_cond, **src_kwargs)

                # target 前向前注册 hook，结束后立即移除
                if use_agve:
                    attn_k = {}
                    hooks  = [
                        model.blocks[bi].cross_attn.register_forward_hook(
                            _make_attn_hook(attn_k, bi)
                        )
                        for bi in vital_block_indices
                    ]

                _, _, Vt_tar = self._get_model_prediction(model, zt_tar, t, tar_cond, **tar_kwargs)

                if use_agve:
                    for h in hooks:
                        h.remove()
                    for idx, attn in attn_k.items():
                        if idx not in attn_accum:
                            attn_accum[idx] = attn.float() / n_avg
                        else:
                            attn_accum[idx] += attn.float() / n_avg

                V_delta_avg += (1 / n_avg) * (Vt_tar - Vt_src)

            if use_agve:
                # 部件1：速度信号 [1,1,16,16,16]
                s_vel = V_delta_avg.norm(dim=1, keepdim=True)
                s_vel = (s_vel - s_vel.min()) / (s_vel.max() - s_vel.min() + 1e-8)

                # 部件1：注意力信号 [1,1,16,16,16]
                if attn_accum:
                    # 均值融合 vital layers → [1, H, N_3d, N_dino]
                    A_tgt = torch.stack(list(attn_accum.values()), dim=0).mean(0)
                    A_tgt = A_tgt.mean(1)          # 均值 over heads → [1, N_3d, N_dino]
                    A_E   = A_tgt[:, :, E]          # 选 E 列 → [1, N_3d, k]
                    s_attn_flat = A_E.mean((0, 2))  # 均值 over batch 和 k → [N_3d]

                    # 从 patch token 空间 [pg^3] 还原到 latent voxel 空间 [res^3]
                    pg     = model.resolution // model.patch_size  # 8
                    s_attn = s_attn_flat.reshape(1, 1, pg, pg, pg)
                    s_attn = F.interpolate(
                        s_attn.float(),
                        size=(model.resolution,) * 3,
                        mode='trilinear',
                        align_corners=False,
                    )  # [1,1,16,16,16]
                    s_attn = (s_attn - s_attn.min()) / (s_attn.max() - s_attn.min() + 1e-8)
                else:
                    s_attn = torch.zeros_like(s_vel)

                # 双信号合并 → sigmoid mask
                m = torch.sigmoid(alpha * s_vel + beta * s_attn - delta)

                # 部件2：EMA 维护运行掩码
                if M is None or edit_idx < warmup_steps:
                    M = m
                else:
                    M = gamma * M + (1 - gamma) * m

                gate = torch.ones_like(M) if edit_idx < warmup_steps else M
                edit_idx += 1

            zt_edit = zt_edit.to(torch.float32)
            if use_agve:
                zt_edit = zt_edit + (t_prev - t) * gate * V_delta_avg
            else:
                zt_edit = zt_edit + (t_prev - t) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        # 部件4：把最终 mask 挂到 sampler 上，供 run() 做软混合
        self._agve_mask = M.detach().cpu() if (use_agve and M is not None) else None
        return zt_edit
    else:
        kwargs.pop("output_path")
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret
