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

        # Optional: round-0 target cond for geometric consistency guidance.
        # When provided, all three velocities are queried at the same zt_tar,
        # and the round-0 delta is added with a small weight to anchor the
        # trajectory without dominating the new editing signal.
        tar_cond_r0 = kwargs["tar_cond_r0"]["cond"] if "tar_cond_r0" in kwargs else None
        geo_alpha   = kwargs.get("geo_alpha", 0.1)   # intentionally small
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

        vdelta_save_dir = os.path.join(kwargs.get("output_path", ""), "vdelta")
        os.makedirs(vdelta_save_dir, exist_ok=True)
        vdelta_records = []   # list of {"step": int, "t": float, "v_delta": np.ndarray}

        # Save source latent for later cosine-sim analysis
        np.save(os.path.join(vdelta_save_dir, "x_src_packed.npy"),
                x_src_packed.float().cpu().numpy())

        num_st = -1
        for t, t_prev in tqdm(t_pairs, desc="Sampling", disable=not verbose):
            num_st += 1
            if num_st < st_step:
                continue
            V_delta_avg  = torch.zeros_like(x_src_packed)
            for k in range(n_avg):
                fwd_noise = torch.randn_like(x_src_packed).to(x_src_packed.device)
                zt_src    = (1-t)*x_src_packed + (t)*fwd_noise
                zt_tar    = zt_edit + zt_src - x_src_packed

                if tar_cond_r0 is not None:
                    # All three predictions at the same zt_tar so the
                    # differences are pure conditioning deltas with no
                    # cross-state noise.
                    #   V = (V_tgt2 - V_src)           ← new editing signal
                    #     + geo_alpha*(V_tgt1 - V_src)  ← soft anchor to r0
                    _, _, Vt_src  = self._get_model_prediction(model, zt_tar, t, src_cond,    **src_kwargs)
                    _, _, Vt_tar1 = self._get_model_prediction(model, zt_tar, t, tar_cond_r0, **tar_kwargs)
                    _, _, Vt_tar2 = self._get_model_prediction(model, zt_tar, t, tar_cond,    **tar_kwargs)
                    V_delta = (Vt_tar2 - Vt_src) + geo_alpha * (Vt_tar1 - Vt_src)
                else:
                    # Original algorithm — untouched
                    _, _, Vt_src = self._get_model_prediction(model, zt_src, t, src_cond, **src_kwargs)
                    _, _, Vt_tar = self._get_model_prediction(model, zt_tar, t, tar_cond, **tar_kwargs)
                    V_delta = Vt_tar - Vt_src

                V_delta_avg += (1/n_avg) * V_delta

            # Save V_delta for this step: shape [1, 8, 16, 16, 16]
            v_delta_np = V_delta_avg.float().cpu().numpy()
            np.save(os.path.join(vdelta_save_dir, f"step_{num_st:03d}_t{t:.4f}.npy"), v_delta_np)
            vdelta_records.append({"step": num_st, "t": float(t), "v_delta": v_delta_np})

            zt_edit = zt_edit.to(torch.float32)
            zt_edit = zt_edit + (t_prev - t) * V_delta_avg
            zt_edit = zt_edit.to(V_delta_avg.dtype)

        # Save final edited latent for voxel-diff cosine-sim analysis
        np.save(os.path.join(vdelta_save_dir, "zt_edit_final.npy"),
                zt_edit.float().cpu().numpy())

        # Save metadata: step index and t value for each record
        meta = [{"step": r["step"], "t": r["t"]} for r in vdelta_records]
        np.save(os.path.join(vdelta_save_dir, "meta.npy"), meta)
        print(f"[vdelta] Saved {len(vdelta_records)} V_delta tensors to: {vdelta_save_dir}")
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
