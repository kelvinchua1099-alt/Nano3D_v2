#!/usr/bin/env python3
"""
Multi-round refinement for Nano3D.

After round-0 editing produces edit_mesh.glb, this module:
  1. Renders 4 views (front/right/back/left) of the edited mesh.
  2. Calls Qwen on each view with the round-0 front edit as a visual reference.
  3. Runs pipeline.run() with:
       - SS flow  : front refined view as tar_cond,
                    round-0 tar image as geo-anchor (tar_cond_r0, alpha=0.1)
       - SLAT flow: all 4 refined views via inject_sampler_multi_image (stochastic)
  4. Exports the refined mesh and returns the new slat for the next round.
"""

import os
import shutil
import torch
from PIL import Image
from trellis.utils import postprocessing_utils

from inference.rendering import render_n_views, FOUR_VIEW_CONFIG
from inference.image_processing import bg_to_white, resize_to_512
from inference.qwen_image_edit import qwen_image_edit_main
from inference.voxel_encoding import encode_voxel_grid


# ─── View-specific prompts ────────────────────────────────────────────────────
# The composite mechanism in qwen_image_edit places the reference on the LEFT
# half of the image, so prompts refer to "the left image" as reference.
VIEW_PROMPTS = {
    "front": (
        "{instruction}. "
        "The left image shows the desired editing result from the front. "
        "Apply the same edit to this front view and fix any remaining issues."
    ),
    "right": (
        "{instruction}. "
        "The left image shows the desired editing result seen from the front. "
        "Now viewing from the right side: ensure '{instruction}' looks correct "
        "and complete from this angle. Fix any missing or broken parts visible "
        "from the right that are inconsistent with the front reference."
    ),
    "back": (
        "{instruction}. "
        "The left image shows the desired editing result seen from the front. "
        "Now viewing from the back: ensure '{instruction}' looks correct "
        "and complete from the back. Fix any missing or broken parts visible "
        "from behind that are inconsistent with the front reference."
    ),
    "left": (
        "{instruction}. "
        "The left image shows the desired editing result seen from the front. "
        "Now viewing from the left side: ensure '{instruction}' looks correct "
        "and complete from this angle. Fix any missing or broken parts visible "
        "from the left that are inconsistent with the front reference."
    ),
}


class MultiRoundRefiner:
    """
    Wraps the multi-round refinement loop.

    Args:
        pipeline:       Injected TrellisImageTo3DPipeline.
        qwen_pipeline:  Loaded Qwen image-edit pipeline (or None to skip Qwen).
        lora_path:      LoRA weights path (passed through to qwen_image_edit_main).
    """

    def __init__(self, pipeline, qwen_pipeline=None, lora_path=""):
        self.pipeline      = pipeline
        self.qwen_pipeline = qwen_pipeline
        self.lora_path     = lora_path

    # ── helpers ───────────────────────────────────────────────────────────────

    def _encode_ply_to_latent(self, ply_path: str, work_dir: str) -> str:
        """
        Copy a voxel PLY to work_dir/voxels.ply, run the sparse-structure
        encoder, save latent.pt, and return the path.
        """
        voxel_dir = os.path.join(work_dir, "voxel_src")
        os.makedirs(voxel_dir, exist_ok=True)
        shutil.copy(ply_path, os.path.join(voxel_dir, "voxels.ply"))
        latent = encode_voxel_grid(self.pipeline, voxel_dir)
        latent_path = os.path.join(work_dir, "latent.pt")
        torch.save(latent, latent_path)
        return latent_path

    def _render_and_edit_views(
        self,
        mesh_path: str,
        ref_image_path: str,
        edit_instruction: str,
        out_dir: str,
        seed: int = 42,
        model_name: str = "Qwen/Qwen-Image-Edit",
        num_inference_steps: int = 8,
        true_cfg_scale: float = 1.0,
    ):
        """
        Render 4 views, white-background them, Qwen-edit each with the
        round-0 front image as a composite reference, resize to 512.

        Returns:
            List of dicts: {name, path}  (path = final 512×512 edited image)
        """
        views_dir   = os.path.join(out_dir, "rendered_views")
        refined_dir = os.path.join(out_dir, "refined_views")
        os.makedirs(refined_dir, exist_ok=True)

        # 1. Render
        rendered = render_n_views(mesh_path, views_dir)   # [{name, yaw, image_path}]

        results = []
        for view in rendered:
            name     = view["name"]
            raw_path = view["image_path"]

            # 2. White background
            white_path = bg_to_white(raw_path)

            # 3. Build view-specific prompt
            prompt = VIEW_PROMPTS[name].format(instruction=edit_instruction)

            save_path = os.path.join(refined_dir, f"{name}_qwen.png")

            if self.qwen_pipeline is not None:
                # 4. Qwen edit (reference composite is handled inside)
                qwen_image_edit_main(
                    pipe                 = self.qwen_pipeline,
                    model_name           = model_name,
                    image_path           = white_path,
                    edit_instruction     = prompt,
                    save_path            = save_path,
                    base_seed            = seed,
                    num_inference_steps  = num_inference_steps,
                    true_cfg_scale       = true_cfg_scale,
                    reference_image_path = ref_image_path,
                )
            else:
                # No Qwen: fall back to using the raw white view directly
                shutil.copy(white_path, save_path)
                print(f"[refine] Qwen not loaded, using raw view for {name}")

            # 5. Resize to 512×512
            resized = resize_to_512(save_path, refined_dir)
            results.append({"name": name, "path": resized})
            print(f"[refine] {name} view refined → {resized}")

        return results

    # ── single round ──────────────────────────────────────────────────────────

    def refine_one_round(
        self,
        src_image_path: str,        # original source front image (for src_cond)
        src_ply_path: str,          # previous round's edit_voxel_post.ply
        current_mesh_path: str,     # previous round's mesh to render views from
        current_slat,               # previous round's tar_slat (SparseTensor)
        ref_image_path: str,        # round-0 front edited image (geo-anchor)
        edit_instruction: str,
        editing_mode: str,
        out_dir: str,
        round_idx: int,
        seed: int = 1,
        geo_alpha: float = 0.1,
        formats=('mesh', 'gaussian', 'radiance_field'),
    ) -> dict:
        """
        One refinement round.  Returns dict with mesh_path, ply_path, slat.
        """
        round_dir = os.path.join(out_dir, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"REFINEMENT ROUND {round_idx}  →  {round_dir}")
        print(f"{'='*60}")

        # ── 1. Render 4 views + Qwen edit ────────────────────────────────────
        refined_views = self._render_and_edit_views(
            mesh_path        = current_mesh_path,
            ref_image_path   = ref_image_path,
            edit_instruction = edit_instruction,
            out_dir          = round_dir,
            seed             = seed,
        )

        # ── 2. Re-encode previous round's voxel structure ────────────────────
        latent_path = self._encode_ply_to_latent(src_ply_path, round_dir)

        # ── 3. Front refined view drives SS flow (single tar_cond) ───────────
        front_refined = next(v["path"] for v in refined_views if v["name"] == "front")
        all_refined   = [v["path"] for v in refined_views]   # all 4 for SLAT

        # ── 4. Run editing pipeline ───────────────────────────────────────────
        outputs = self.pipeline.run(
            source_image_path        = src_image_path,
            target_image_path        = front_refined,
            source_ply_path          = src_ply_path,
            source_voxel_latent_path = latent_path,
            source_slat              = current_slat,
            editing_mode             = editing_mode,
            seed                     = seed,
            output_path              = round_dir,
            # geo-anchor: round-0 front edit keeps trajectory from drifting
            ref_image_path           = ref_image_path,
            geo_alpha                = geo_alpha,
            # SLAT: stochastic alternation across all 4 refined views
            slat_tar_image_paths     = all_refined,
            formats                  = list(formats),
        )

        # ── 5. Export GLB ─────────────────────────────────────────────────────
        with torch.enable_grad():
            glb = postprocessing_utils.to_glb(
                outputs['gaussian'][0],
                outputs['mesh'][0],
                simplify     = 0.95,
                texture_size = 1024,
            )
        mesh_path = os.path.join(round_dir, "edit_mesh.glb")
        glb.export(mesh_path)
        print(f"[refine] Round {round_idx} mesh → {mesh_path}")

        return {
            "mesh_path": mesh_path,
            "ply_path":  os.path.join(round_dir, "edit_voxel_post.ply"),
            "slat":      outputs["slat"],
        }

    # ── multi-round loop ──────────────────────────────────────────────────────

    def run(
        self,
        src_image_path: str,
        first_edit_mesh_path: str,
        first_edit_ply_path: str,
        first_edit_slat,
        first_edit_image_path: str,   # round-0 tar_image, used as geo-anchor ref
        edit_instruction: str,
        editing_mode: str,
        out_dir: str,
        n_rounds: int = 1,
        seed: int = 1,
        geo_alpha: float = 0.1,
    ) -> dict:
        """
        Run n_rounds of refinement starting from round-0 outputs.

        The geo-anchor reference (ref_image_path) is always the round-0 front
        edited image — it stays fixed across rounds to prevent drift.

        Returns:
            dict with final mesh_path and slat.
        """
        current_mesh = first_edit_mesh_path
        current_ply  = first_edit_ply_path
        current_slat = first_edit_slat

        for round_idx in range(1, n_rounds + 1):
            result = self.refine_one_round(
                src_image_path   = src_image_path,
                src_ply_path     = current_ply,
                current_mesh_path= current_mesh,
                current_slat     = current_slat,
                ref_image_path   = first_edit_image_path,
                edit_instruction = edit_instruction,
                editing_mode     = editing_mode,
                out_dir          = out_dir,
                round_idx        = round_idx,
                seed             = seed,
                geo_alpha        = geo_alpha,
            )
            current_mesh = result["mesh_path"]
            current_ply  = result["ply_path"]
            current_slat = result["slat"]

        return {"mesh_path": current_mesh, "slat": current_slat}
