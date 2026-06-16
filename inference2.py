#!/usr/bin/env python3

import os
import re
import sys
import bpy
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

torch.set_grad_enabled(False)

# ============================================================================
# PASTE YOUR API KEY HERE
# ============================================================================
DEEPSEEK_API_KEY = "sk-b9faa6503e554495ad41d8d815a022ff"   # ← 把 key 粘贴到这里的引号里
# ============================================================================

# Import all utility functions from submodules
from inference.image_processing import bg_to_white, resize_to_512
from inference.rendering import render_front_view, render_3d_model
from inference.model_utils import load_sparse_structure_encoder, inject_methods
from inference.sampling import sample
from inference.qwen_image_edit import qwen_image_edit_main, load_qwen_image

# ============================================================================
# STEP-0: Load pipeline and model
# ============================================================================
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
pipeline = load_sparse_structure_encoder(pipeline)
# Save a reference to the original TRELLIS run() before injection (for direct mode)
_trellis_run_original = TrellisImageTo3DPipeline.run
pipeline = inject_methods(pipeline)
print(f"\nLoading TRELLIS pipeline Done")

dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
dinov2_model.eval().cuda()
print(f"\nLoading DINOv2 model Done")


# ============================================================================
# DeepSeek part decomposition
# ============================================================================
def decompose_parts_via_deepseek(
    object_description: str,
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> list:
    """Call DeepSeek API with a text description to get an ordered parts list."""
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = (
        "You are a 3D asset decomposition expert. "
        "When given a text description of a 3D object, decompose it into its structural parts "
        "and return them as an ordered JSON array of short edit instructions, "
        "from the most fundamental base part to the finest detail."
    )

    user_prompt = (
        f'Decompose this 3D object into its distinct structural parts: "{object_description}"\n\n'
        "Rules:\n"
        "- Order from most fundamental (e.g. body/torso) to most detailed (e.g. accessories)\n"
        "- Each item must be a short edit instruction in English (under 10 words)\n"
        "- Format: 'add <part description>'\n"
        "- Return ONLY a valid JSON array of strings, nothing else\n\n"
        'Example for "a cartoon bear with a red hat and backpack":\n'
        '["add bear body", "add bear legs", "add bear arms", "add bear head", "add red hat", "add backpack"]'
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",  "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.2,
    )

    content = response.choices[0].message.content.strip()
    print(f"[DeepSeek] Raw response: {content}")

    json_match = re.search(r"\[.*?\]", content, re.DOTALL)
    if json_match:
        parts = json.loads(json_match.group())
        assert isinstance(parts, list), "Expected a list"
        return [str(p) for p in parts]

    raise ValueError(f"Could not parse parts list from DeepSeek response: {content}")


# ============================================================================
# Qwen prompt template (same as app.py)
# ============================================================================
QWEN_TEMPLATE = (
    'Carefully edit the image to only perform the following change: [Edit content: "{}"], '
    "while strictly keeping the rest of the original image unchanged. "
    "Do not alter the shape, proportions, colors, textures, pose/structure, composition, "
    "or lighting of the original subjects. "
    "Do not change any elements that are not directly related to [Edit content]. "
    "The overall style, sharpness, and level of detail must remain perfectly consistent "
    "with the original image. keep white background."
)


def wrapped_edit_instruction(instruction: str) -> str:
    return QWEN_TEMPLATE.format(instruction)


# ============================================================================
# Path A: iterative part-by-part generation
# ============================================================================
def run_decompose(
    src_input_image_path: str,
    object_description: str,
    output_dir: str,
    editing_mode: str,
    qwen_image_pipeline,
    deepseek_api_key: str,
    deepseek_model: str,
    lora_path: str,
):
    decompose_dir = os.path.join(output_dir, "decompose")
    os.makedirs(decompose_dir, exist_ok=True)

    # ---- iter_00: generate base mesh from source image ----
    iter_dir = os.path.join(decompose_dir, "iter_00")
    os.makedirs(os.path.join(iter_dir, "image"), exist_ok=True)

    print("\n[Decompose] Step 0: Generating base mesh from source image...")
    result = pipeline.run_custom(
        src_input_image_path,
        seed        = 1,
        output_path = iter_dir,
    )
    with torch.enable_grad():
        src_glb = postprocessing_utils.to_glb(
            result["src_mesh"]["gaussian"][0],
            result["src_mesh"]["mesh"][0],
            simplify     = 0.95,
            texture_size = 1024,
        )
    current_mesh_path = os.path.join(iter_dir, "mesh.glb")
    src_glb.export(current_mesh_path)

    # ---- Call DeepSeek to get parts list ----
    print("\n[Decompose] Calling DeepSeek to decompose object into parts...")
    parts = decompose_parts_via_deepseek(
        object_description = object_description,
        api_key            = deepseek_api_key,
        model              = deepseek_model,
    )
    print(f"[Decompose] Parts ({len(parts)}): {parts}")
    with open(os.path.join(decompose_dir, "parts.json"), "w") as f:
        json.dump(parts, f, indent=2, ensure_ascii=False)

    # ---- Iterative editing: one part per iteration ----
    current_slat = result["src_slat"]

    for idx, part_instruction in enumerate(parts):
        prev_iter_dir = os.path.join(decompose_dir, f"iter_{idx:02d}")
        curr_iter_dir = os.path.join(decompose_dir, f"iter_{idx + 1:02d}")
        os.makedirs(os.path.join(curr_iter_dir, "image"), exist_ok=True)

        print(f"\n[Decompose] Iteration {idx + 1}/{len(parts)}: '{part_instruction}'")

        # Re-derive voxels/latent for the current mesh via run_custom on its render
        render_front_view(
            file_path   = current_mesh_path,
            output_dir  = os.path.join(curr_iter_dir, "image"),
            output_name = "front.png",
        )
        src_image_path = bg_to_white(os.path.join(curr_iter_dir, "image", "front.png"))

        # run_custom saves voxels.ply + latent.pt to curr_iter_dir
        iter_result = pipeline.run_custom(
            src_image_path,
            seed        = 1,
            output_path = curr_iter_dir,
        )
        current_slat = iter_result["src_slat"]

        # Qwen-Image edit for this part
        tar_image_path = os.path.join(curr_iter_dir, "image", "edited.png")
        qwen_image_edit_main(
            pipe                = qwen_image_pipeline,
            model_name          = "Qwen/Qwen-Image-Edit-2509",
            image_path          = src_image_path,
            edit_instruction    = wrapped_edit_instruction(part_instruction),
            save_path           = tar_image_path,
            base_seed           = 42,
            num_inference_steps = 8,
            true_cfg_scale      = 1.0,
        )
        tar_image_path = resize_to_512(tar_image_path, os.path.join(curr_iter_dir, "image"))

        # Nano3D editing
        outputs = pipeline.run(
            src_image_path,
            tar_image_path,
            source_ply_path          = os.path.join(curr_iter_dir, "voxels.ply"),
            source_voxel_latent_path = os.path.join(curr_iter_dir, "latent.pt"),
            source_slat              = current_slat,
            editing_mode             = editing_mode,
            seed                     = 1,
            output_path              = curr_iter_dir,
        )

        with torch.enable_grad():
            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify     = 0.95,
                texture_size = 1024,
            )
        current_mesh_path = os.path.join(curr_iter_dir, "mesh.glb")
        glb.export(current_mesh_path)
        print(f"[Decompose] Part {idx + 1} done → {current_mesh_path}")

    # Copy final mesh to decompose root
    import shutil
    final_path = os.path.join(decompose_dir, "final_mesh.glb")
    shutil.copy(current_mesh_path, final_path)
    print(f"\n[Decompose] Final mesh saved to {final_path}")


# ============================================================================
# Path B: direct TRELLIS generation (baseline, no editing)
# ============================================================================
def run_direct(
    src_input_image_path: str,
    output_dir: str,
):
    direct_dir = os.path.join(output_dir, "direct")
    os.makedirs(direct_dir, exist_ok=True)

    print("\n[Direct] Running original TRELLIS image-to-3D (no editing)...")
    image = Image.open(src_input_image_path)

    # Use the original uninjected TRELLIS run() via the class-level unbound method
    outputs = _trellis_run_original(pipeline, image, preprocess_image=True, seed=1)

    with torch.enable_grad():
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify     = 0.95,
            texture_size = 1024,
        )
    out_path = os.path.join(direct_dir, "mesh.glb")
    glb.export(out_path)
    print(f"[Direct] Mesh saved to {out_path}")


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nano3D — decompose or direct generation")

    parser.add_argument("--src_input_image_path", type=str, required=True,
                        help="Path to source input image")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["decompose", "direct"],
                        help="'decompose': LLM part-by-part editing; 'direct': TRELLIS one-shot generation")

    # decompose-mode args
    parser.add_argument("--editing_mode", type=str, default="add",
                        choices=["add", "remove", "replace"],
                        help="Nano3D editing mode (decompose only)")
    parser.add_argument("--lora_path", type=str, default="",
                        help="Path to Qwen-Image LoRA weights (decompose only)")
    parser.add_argument("--deepseek_api_key", type=str, default=os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY),
                        help="DeepSeek API key (decompose only)")
    parser.add_argument("--deepseek_model", type=str, default="deepseek-chat",
                        help="DeepSeek model name (default: deepseek-chat)")
    parser.add_argument("--object_description", type=str, default="",
                        help="Text description of the object to decompose into parts (decompose only)")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.mode == "decompose":
        assert args.deepseek_api_key, "--deepseek_api_key is required for decompose mode"
        assert args.lora_path, "--lora_path is required for decompose mode"
        assert args.object_description, "--object_description is required for decompose mode"

        print("Loading Qwen-Image model...")
        qwen_image_pipeline = load_qwen_image(
            model_name = "Qwen/Qwen-Image-Edit-2509",
            lora_path  = args.lora_path,
        )
        print("Qwen-Image loaded\n")

        run_decompose(
            src_input_image_path = args.src_input_image_path,
            object_description   = args.object_description,
            output_dir           = args.output_dir,
            editing_mode         = args.editing_mode,
            qwen_image_pipeline  = qwen_image_pipeline,
            deepseek_api_key     = args.deepseek_api_key,
            deepseek_model       = args.deepseek_model,
            lora_path            = args.lora_path,
        )

    elif args.mode == "direct":
        run_direct(
            src_input_image_path = args.src_input_image_path,
            output_dir           = args.output_dir,
        )
