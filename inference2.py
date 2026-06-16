#!/usr/bin/env python3

import os
import re
import sys
import bpy
import json
import torch
import utils3d
import requests
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
# PASTE YOUR API KEYS HERE
# ============================================================================
DEEPSEEK_API_KEY = "sk-b9faa6503e554495ad41d8d815a022ff"  # ← DeepSeek key
TEXT2IMG_API_KEY = ""                                       # ← 文生图 key（SiliconFlow 等）
# ============================================================================

# Import all utility functions from submodules
from inference.image_processing import bg_to_white, resize_to_512
from inference.rendering import render_front_view, render_3d_model
from inference.model_utils import load_sparse_structure_encoder, inject_methods
from inference.sampling import sample
from inference.qwen_image_edit import qwen_image_edit_main, load_qwen_image

# ============================================================================
# Load pipeline and model
# ============================================================================
pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
pipeline.cuda()
pipeline = load_sparse_structure_encoder(pipeline)
_trellis_run_original = TrellisImageTo3DPipeline.run   # save before injection
pipeline = inject_methods(pipeline)
print(f"\nLoading TRELLIS pipeline Done")

dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14_reg", pretrained=True)
dinov2_model.eval().cuda()
print(f"\nLoading DINOv2 model Done")


# ============================================================================
# Text-to-image (SiliconFlow / any OpenAI-compatible image API)
# ============================================================================
def text_to_image(
    prompt: str,
    save_path: str,
    api_key: str,
    base_url: str = "https://api.siliconflow.cn/v1",
    model: str = "black-forest-labs/FLUX.1-schnell",
    image_size: str = "1024x1024",
) -> str:
    resp = requests.post(
        f"{base_url}/images/generations",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "prompt": prompt, "n": 1, "image_size": image_size},
        timeout=120,
    )
    resp.raise_for_status()
    url = resp.json()["data"][0]["url"]
    img_bytes = requests.get(url, timeout=60).content
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    with open(save_path, "wb") as f:
        f.write(img_bytes)
    print(f"[text2img] Saved → {save_path}")
    return save_path


# ============================================================================
# DeepSeek part decomposition (text-only)
# ============================================================================
def decompose_parts_via_deepseek(
    text_prompt: str,
    api_key: str,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
) -> dict:
    """
    Returns:
        {
            "first_part": str,       # text2img prompt for the base part
            "edits":      list[str], # delta edit instructions for Qwen
        }
    """
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url=base_url)

    system_prompt = (
        "You are a 3D asset decomposition expert. "
        "Given a text description of a 3D object, you decompose it into structural parts "
        "and return a JSON object with two fields:\n"
        "  'first_part': a text-to-image prompt describing ONLY the most fundamental base part\n"
        "  'edits': an ordered list of short delta edit instructions for adding remaining parts\n"
        "Return ONLY valid JSON, nothing else."
    )
    user_prompt = (
        f'Decompose this 3D object: "{text_prompt}"\n\n'
        "Rules:\n"
        "- 'first_part': text-to-image prompt for the LARGEST, most fundamental base part only "
        "(include 'white background, 3D render, front view')\n"
        "- 'edits': ordered list of delta instructions, strictly from LARGE/COARSE to SMALL/FINE — "
        "large structural parts first (torso, limbs), small surface details last (hat, buttons, patterns). "
        "Each instruction starts with 'add', under 8 words.\n\n"
        'Example for "a cartoon bear with a red hat and backpack":\n'
        '{\n'
        '  "first_part": "cartoon bear body torso, white background, 3D render, front view",\n'
        '  "edits": ["add bear legs", "add bear arms", "add bear head", "add backpack", "add red hat"]\n'
        '}'
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        max_tokens=512,
        temperature=0.2,
    )
    content = response.choices[0].message.content.strip()
    print(f"[DeepSeek] {content}")

    match = re.search(r"\{.*\}", content, re.DOTALL)
    if match:
        result = json.loads(match.group())
        assert "first_part" in result and "edits" in result
        return {"first_part": str(result["first_part"]), "edits": [str(e) for e in result["edits"]]}
    raise ValueError(f"Cannot parse decomposition from DeepSeek response: {content}")


# ============================================================================
# Qwen prompt wrapper (same as app.py)
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


# ============================================================================
# Path A: decompose → part-by-part iterative generation
# ============================================================================
def run_decompose(
    text_prompt: str,
    output_dir: str,
    editing_mode: str,
    qwen_image_pipeline,
    deepseek_api_key: str,
    deepseek_model: str,
    text2img_api_key: str,
    text2img_base_url: str,
    text2img_model: str,
):
    decompose_dir = os.path.join(output_dir, "decompose")
    os.makedirs(decompose_dir, exist_ok=True)

    # ── Step 1: DeepSeek decomposes text prompt ─────────────────────────────
    print("\n[Decompose] Calling DeepSeek to decompose prompt...")
    decomp = decompose_parts_via_deepseek(
        text_prompt = text_prompt,
        api_key     = deepseek_api_key,
        model       = deepseek_model,
    )
    first_part = decomp["first_part"]   # text2img prompt for the base
    edits      = decomp["edits"]        # delta instructions for Qwen
    print(f"[Decompose] first_part: {first_part}")
    print(f"[Decompose] edits ({len(edits)}): {edits}")
    with open(os.path.join(decompose_dir, "parts.json"), "w") as f:
        json.dump({"text_prompt": text_prompt, **decomp}, f, indent=2, ensure_ascii=False)

    # ── Step 2: Generate image of FIRST part → base mesh ────────────────────
    iter_dir = os.path.join(decompose_dir, "iter_00")
    os.makedirs(os.path.join(iter_dir, "image"), exist_ok=True)

    print(f"\n[Decompose] Generating image for first part: '{first_part}'")
    first_part_image = text_to_image(
        prompt    = first_part,
        save_path = os.path.join(iter_dir, "image", "generated.png"),
        api_key   = text2img_api_key,
        base_url  = text2img_base_url,
        model     = text2img_model,
    )
    first_part_image = resize_to_512(first_part_image, os.path.join(iter_dir, "image"))

    print(f"\n[Decompose] Generating base mesh from first part image...")
    base_result = pipeline.run_custom(
        first_part_image,
        seed        = 1,
        output_path = iter_dir,   # saves voxels.ply + latent.pt here
    )
    with torch.enable_grad():
        base_glb = postprocessing_utils.to_glb(
            base_result["src_mesh"]["gaussian"][0],
            base_result["src_mesh"]["mesh"][0],
            simplify     = 0.95,
            texture_size = 1024,
        )
    current_mesh_path = os.path.join(iter_dir, "mesh.glb")
    base_glb.export(current_mesh_path)
    print(f"[Decompose] Base mesh → {current_mesh_path}")

    # ── Step 3: Iteratively add remaining parts via Nano3D edit ─────────────
    for idx, part_prompt in enumerate(edits, start=1):
        curr_iter_dir = os.path.join(decompose_dir, f"iter_{idx:02d}")
        os.makedirs(os.path.join(curr_iter_dir, "image"), exist_ok=True)

        print(f"\n[Decompose] Part {idx}/{len(edits)}: '{part_prompt}'")

        # Render current mesh → source image for this step
        render_front_view(
            file_path   = current_mesh_path,
            output_dir  = os.path.join(curr_iter_dir, "image"),
            output_name = "front.png",
        )
        src_image_path = bg_to_white(os.path.join(curr_iter_dir, "image", "front.png"))

        # Re-derive voxels/latent/slat for the current mesh state
        iter_result = pipeline.run_custom(
            src_image_path,
            seed        = 1,
            output_path = curr_iter_dir,
        )
        current_slat = iter_result["src_slat"]

        # Qwen-Image: edit rendered image to add this part
        # edit_instruction is the diff between parts[idx-1] and parts[idx]
        edit_instruction = QWEN_TEMPLATE.format(f"add {part_prompt}")
        tar_image_path = os.path.join(curr_iter_dir, "image", "edited.png")
        qwen_image_edit_main(
            pipe                = qwen_image_pipeline,
            model_name          = "Qwen/Qwen-Image-Edit-2509",
            image_path          = src_image_path,
            edit_instruction    = edit_instruction,
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
        print(f"[Decompose] Part {idx} done → {current_mesh_path}")

    import shutil
    final_path = os.path.join(decompose_dir, "final_mesh.glb")
    shutil.copy(current_mesh_path, final_path)
    print(f"\n[Decompose] Final mesh → {final_path}")


# ============================================================================
# Path B: direct — text → image → TRELLIS one-shot (baseline)
# ============================================================================
def run_direct(
    text_prompt: str,
    output_dir: str,
    text2img_api_key: str,
    text2img_base_url: str,
    text2img_model: str,
):
    direct_dir = os.path.join(output_dir, "direct")
    os.makedirs(direct_dir, exist_ok=True)

    print(f"\n[Direct] Generating full-object image from prompt: '{text_prompt}'")
    full_image_path = text_to_image(
        prompt    = text_prompt,
        save_path = os.path.join(direct_dir, "input.png"),
        api_key   = text2img_api_key,
        base_url  = text2img_base_url,
        model     = text2img_model,
    )

    print("[Direct] Running original TRELLIS image-to-3D...")
    image   = Image.open(full_image_path)
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
    print(f"[Direct] Mesh → {out_path}")


# ============================================================================
# Entry point
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nano3D — decompose or direct generation")

    parser.add_argument("--text_prompt", type=str, required=True,
                        help="Text description of the target 3D object")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Root output directory")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["decompose", "direct"],
                        help="'decompose': part-by-part; 'direct': one-shot baseline")

    # decompose-only args
    parser.add_argument("--editing_mode", type=str, default="add",
                        choices=["add", "remove", "replace"])
    parser.add_argument("--lora_path", type=str, default="")
    parser.add_argument("--deepseek_api_key", type=str,
                        default=os.environ.get("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY))
    parser.add_argument("--deepseek_model", type=str, default="deepseek-chat")

    # text-to-image args (both modes)
    parser.add_argument("--text2img_api_key", type=str,
                        default=os.environ.get("TEXT2IMG_API_KEY", TEXT2IMG_API_KEY))
    parser.add_argument("--text2img_base_url", type=str,
                        default="https://api.siliconflow.cn/v1")
    parser.add_argument("--text2img_model", type=str,
                        default="black-forest-labs/FLUX.1-schnell")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    assert args.text2img_api_key, "--text2img_api_key is required"

    if args.mode == "decompose":
        assert args.deepseek_api_key, "--deepseek_api_key is required for decompose mode"
        assert args.lora_path, "--lora_path is required for decompose mode"

        print("Loading Qwen-Image model...")
        qwen_image_pipeline = load_qwen_image(
            model_name = "Qwen/Qwen-Image-Edit-2509",
            lora_path  = args.lora_path,
        )
        print("Qwen-Image loaded\n")

        run_decompose(
            text_prompt      = args.text_prompt,
            output_dir       = args.output_dir,
            editing_mode     = args.editing_mode,
            qwen_image_pipeline = qwen_image_pipeline,
            deepseek_api_key = args.deepseek_api_key,
            deepseek_model   = args.deepseek_model,
            text2img_api_key = args.text2img_api_key,
            text2img_base_url= args.text2img_base_url,
            text2img_model   = args.text2img_model,
        )

    elif args.mode == "direct":
        run_direct(
            text_prompt      = args.text_prompt,
            output_dir       = args.output_dir,
            text2img_api_key = args.text2img_api_key,
            text2img_base_url= args.text2img_base_url,
            text2img_model   = args.text2img_model,
        )
