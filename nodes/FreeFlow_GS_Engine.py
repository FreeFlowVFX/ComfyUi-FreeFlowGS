"""
FreeFlow_GS_Engine - 4D Quality Training Node
Extends AdaptiveEngine with Rig-Guided Warm Start functionality.

Supports multiple backends:
- Brush (Fast): ArthurBrussee/brush - Single binary, fast training
- Splatfacto (Pro): Nerfstudio Splatfacto - Production quality, rich features
"""

import shutil
import time
import subprocess
import os
import sys
import re
import json
import threading
import numpy as np
import scipy.spatial  # For KDTree
import scipy.signal  # For Savitzky-Golay temporal smoothing
from scipy.spatial import cKDTree  # For topology realignment
import struct

from pathlib import Path
import folder_paths
from ..utils import FreeFlowUtils

# Import engine backends from engines package
from .modules.engines import (
    IGSEngine, 
    BrushEngine, 
    SPLATFACTO_AVAILABLE, 
    OPENSPLAT_AVAILABLE,
    CUDA_SUPPORTED,
    CUDA_UNSUPPORTED_REASON,
)

# Only import SplatfactoEngine if CUDA is supported on this platform
if SPLATFACTO_AVAILABLE:
    from .modules.engines import SplatfactoEngine
else:
    SplatfactoEngine = None

if OPENSPLAT_AVAILABLE:
    from .modules.engines import OpenSplatEngine
else:
    OpenSplatEngine = None  # Placeholder when not available

try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


# ==================================================================================
#   MAIN NODE: FREEFLOW GS ENGINE
# ==================================================================================

class FreeFlow_GS_Engine:
    """
    "Pro" version of the engine.
    Supports Multi-Engine Backends and Mesh-Guided Initialization.
    """
    
    def __init__(self):
        self._last_error = None
        self._cached_seconds_per_step = None
        self._preview_step_counter = {}

    @classmethod
    def INPUT_TYPES(s):
        # Build engine options based on platform capabilities
        # Only show engines that can actually work on the current system
        engine_options = ["Brush (Fast)"]  # Brush is always available (bundled binary for all platforms)
        
        # Splatfacto requires CUDA - only show if platform supports it
        if SPLATFACTO_AVAILABLE:
            engine_options.append("Splatfacto (Pro)")
        
        # OpenSplat works on all platforms (OpenCL/CPU fallback)
        if OPENSPLAT_AVAILABLE and OpenSplatEngine is not None:
            try:
                _test_engine = OpenSplatEngine()
                if _test_engine.is_available():
                    engine_options.append("OpenSplat (Mac/CPU)")
            except Exception:
                pass  # OpenSplat not ready
        
        # Build tooltip based on available engines
        tooltip_parts = ["Brush: Fast single-binary trainer (all platforms)."]
        if SPLATFACTO_AVAILABLE:
            tooltip_parts.append("Splatfacto: Production quality (CUDA required).")
        else:
            tooltip_parts.append(f"Splatfacto: Not available ({CUDA_UNSUPPORTED_REASON}).")
        tooltip_parts.append("OpenSplat: Mac/CPU alternative.")
        
        return {
            "required": {
                # ═══════════════════════════════════════════════════════════════
                # DATA INPUTS
                # ═══════════════════════════════════════════════════════════════
                "multicam_feed": ("MULTICAM_DICT", {
                    "tooltip": "Multi-camera input sequence dictionary. Provide synchronized frame lists for each camera."
                }),
                "colmap_anchor": ("COLMAP_DATA", {
                    "tooltip": "COLMAP anchor reconstruction used for camera calibration and initial sparse geometry."
                }),
            },
            "optional": {
                # ═══════════════════════════════════════════════════════════════
                # ENGINE SELECTION
                # ═══════════════════════════════════════════════════════════════
                "engine_backend": (engine_options, {
                    "default": "Brush (Fast)",
                    "tooltip": " ".join(tooltip_parts)
                }),

                # --- Splatfacto quick selectors (shown only for Splatfacto backend) ---
                "splatfacto_variant": (["splatfacto", "splatfacto-big", "splatfacto-w"], {
                    "default": "splatfacto",
                    "tooltip": "Splatfacto model type. splatfacto is balanced, big is higher quality/cost, w adds appearance embedding for lighting variation."
                }),
                "splatfacto_viewer": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Open Splatfacto's live viewer in browser during training. Useful for inspection, adds overhead."
                }),

                # ═══════════════════════════════════════════════════════════════
                # CINEMA GUIDANCE (Optional mesh-based initialization)
                # ═══════════════════════════════════════════════════════════════
                "guidance_mesh": ("GUIDANCE_MESH_SEQUENCE", {
                    "tooltip": "Optional per-frame mesh guidance for warp-based initialization. Use only if you have a matching mesh sequence."
                }),
                
                # ═══════════════════════════════════════════════════════════════
                # VISUALIZATION & MONITORING [SHARED]
                # ═══════════════════════════════════════════════════════════════
                "visualize_training": (["Off", "Save Preview Images", "Spawn Native GUI"], {
                    "default": "Off",
                    "tooltip": "Training visualization mode. Off is fastest. Save Preview Images writes periodic snapshots. Spawn Native GUI opens Brush's live app window."
                }),
                "preview_interval": ("INT", {
                    "default": 500, "min": 100, "max": 10000,
                    "tooltip": "Steps between preview renders. Lower values give more frequent previews but add overhead."
                }),
                
                # --- Brush-only preview controls ---
                "preview_camera_filter": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Brush-only preview filter. Show only camera names containing this text. Leave empty to include all cameras."
                }),
                "eval_camera_index": ("INT", {
                    "default": 10, "min": 1, "max": 100,
                    "tooltip": "Brush-only preview stride. Render every Nth camera during preview. Higher is faster, lower is more complete."
                }),

                # ═══════════════════════════════════════════════════════════════
                # TOPOLOGY CONTROL [SHARED]
                # ═══════════════════════════════════════════════════════════════
                "topology_mode": (["Dynamic (Default-Flicker)", "Fixed (Stable)"], {
                    "default": "Dynamic (Default-Flicker)",
                    "tooltip": "Topology behavior across frames. Dynamic allows point count changes. Fixed locks topology after anchor for smoother temporal post-processing."
                }),
                "realign_topology": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Reorder splats consistently between frames before smoothing. Recommended when using Fixed mode and temporal filtering."
                }),
                "apply_smoothing": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Run temporal smoothing on the final sequence. Works best with Fixed topology and at least 4 frames."
                }),
                "initial_iterations_override": ("INT", {
                    "default": 12000, "min": 0, "max": 100000,
                    "tooltip": "[Fixed mode] Initial-frame iterations override. 0 disables override. Applies only to the first training frame in the run (or anchor generation frame in distributed producer mode)."
                }),
                "initial_quality_preset": (["Off (Use Base Params)", "High (Recommended)", "Extreme (Slow)"], {
                    "default": "High (Recommended)",
                    "tooltip": "[Fixed + Splatfacto] First-frame quality preset. Off: use base params. High: variant=splatfacto-big, refine_every=100, warmup_length=700, densify_grad_thresh=0.0012, num_downscales=max(base-1,0), cull_alpha_thresh=0.22. Extreme: variant=splatfacto-big, refine_every=80, warmup_length=500, densify_grad_thresh=0.0010, num_downscales=0, cull_alpha_thresh=0.20."
                }),
                
                # ═══════════════════════════════════════════════════════════════
                # CORE TRAINING PARAMETERS [SHARED]
                # ═══════════════════════════════════════════════════════════════
                "iterations": ("INT", {
                    "default": 6000, "min": 100, "max": 100000,
                    "tooltip": "Training steps per frame after initialization. Higher improves quality but increases time. Production baseline: Brush 6k-10k, Splatfacto 10k-25k."
                }),
                "sh_degree": ("INT", {
                    "default": 3, "min": 0, "max": 3,
                    "tooltip": "View-dependent color complexity. 3 gives best realism; 0 is flat color and faster."
                }),
                
                # ═══════════════════════════════════════════════════════════════
                # BRUSH-SPECIFIC PARAMETERS
                # ═══════════════════════════════════════════════════════════════
                "splat_count": ("INT", {
                    "default": 500000, "min": 1000, "max": 10000000,
                    "tooltip": "Brush max splat budget. Higher can improve detail but increases memory and export size."
                }),
                "learning_rate": ("FLOAT", {
                    "default": 0.00002, "min": 0.000001, "max": 0.001, "step": 0.000001,
                    "tooltip": "Brush position learning rate. Lower is more stable, higher adapts faster but may jitter."
                }),
                "densification_interval": ("INT", {
                    "default": 200, "min": 10, "max": 1000,
                    "tooltip": "Brush densification interval. Lower refines more often, higher is more conservative and stable."
                }),
                "densify_grad_threshold": ("FLOAT", {
                    "default": 0.00004, "min": 0.000001, "max": 0.01, "step": 0.000001,
                    "tooltip": "Brush growth threshold. Lower creates more splats, higher reduces noise and growth."
                }),
                "growth_select_fraction": ("FLOAT", {
                    "default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Brush growth fraction per refine step. Lower is safer and less drift-prone; higher grows faster."
                }),
                "feature_lr": ("FLOAT", {
                    "default": 0.0025, "min": 0.0001, "max": 0.05, "step": 0.0001,
                    "tooltip": "Brush color/feature learning rate. Higher adapts color quickly but can increase flicker or saturation."
                }),
                "gaussian_lr": ("FLOAT", {
                    "default": 0.00016, "min": 0.00001, "max": 0.1, "step": 0.00001,
                    "tooltip": "Brush scale/shape learning rate. Keep low for stable geometry; high values can cause stretching artifacts."
                }),
                "opacity_lr": ("FLOAT", {
                    "default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.0001,
                    "tooltip": "Brush opacity learning rate. Higher prunes/reveals faster but may destabilize thin details."
                }),
                "scale_loss_weight": ("FLOAT", {
                    "default": 1e-8, "min": 1e-10, "max": 1e-5, "step": 1e-9,
                    "tooltip": "Brush scale regularization strength. Increase to suppress exploding or elongated splats."
                }),
                "masking_method": (["None (No Masking)", "Optical Flow (Robust)", "Simple Diff (Fast)"], {
                    "default": "None (No Masking)",
                    "tooltip": "Mask generation method for moving regions. None disables masks. Optical Flow is robust, Simple Diff is faster."
                }),
                "motion_sensitivity": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1,
                    "tooltip": "Mask sensitivity for motion detection. Lower catches subtle motion; higher keeps only strong motion."
                }),
                
                # ═══════════════════════════════════════════════════════════════
                # SPLATFACTO-SPECIFIC PARAMETERS
                # ═══════════════════════════════════════════════════════════════
                "cull_alpha_thresh": ("FLOAT", {
                    "default": 0.22, "min": 0.001, "max": 0.5, "step": 0.001,
                    "tooltip": "Splatfacto opacity culling threshold. Higher removes more dark/transparent junk splats. Lower keeps more faint detail and can improve edge coverage."
                }),
                "splatfacto_densify_grad_thresh": ("FLOAT", {
                    "default": 0.0012, "min": 0.0001, "max": 0.01, "step": 0.0001,
                    "tooltip": "Splatfacto growth threshold. Lower grows denser detail (and can add noise); higher is cleaner but less dense."
                }),
                "use_scale_regularization": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Splatfacto anti-stretch regularization. Keep enabled to reduce elongated, noisy, and unstable splats."
                }),

                # --- Advanced Splatfacto Parameters ---
                "max_gs_num": ("INT", {
                    "default": 1000000, "min": 1000, "max": 10000000,
                    "tooltip": "Maximum Gaussian cap (advanced). Only relevant on newer/MCMC strategies; ignored by most default runs."
                }),
                "refine_every": ("INT", {
                    "default": 120, "min": 10, "max": 1000,
                    "tooltip": "Splatfacto refine interval. Lower increases split/prune frequency and density; higher is more conservative."
                }),
                "warmup_length": ("INT", {
                    "default": 800, "min": 0, "max": 5000,
                    "tooltip": "Initial steps before densify/prune starts. Shorter warmup increases early growth; longer warmup is more stable."
                }),
                "num_downscales": ("INT", {
                    "default": 1, "min": 0, "max": 5,
                    "tooltip": "Initial image downscale level. Lower values preserve fine detail (better quality) but increase memory/time."
                }),
                "cull_screen_size": ("FLOAT", {
                    "default": 0.15, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Cull oversized splats by screen footprint (advanced). Increase to remove large floaters more aggressively."
                }),
                "split_screen_size": ("FLOAT", {
                    "default": 0.05, "min": 0.01, "max": 1.0, "step": 0.01,
                    "tooltip": "Split threshold by screen footprint (advanced). Lower splits more often; higher is more conservative."
                }),
                "sh_degree_interval": ("INT", {
                    "default": 1000, "min": 100, "max": 10000,
                    "tooltip": "How often SH complexity increases (advanced). Lower learns view-dependent color sooner."
                }),
                "background_color": (["random", "black", "white"], {
                    "default": "black",
                    "tooltip": "Training background color. Black is typically most stable for reducing bright halos and overexposure artifacts."
                }),
                
                # ═══════════════════════════════════════════════════════════════
                # FRAME SELECTION & INITIALIZATION [SHARED]
                # ═══════════════════════════════════════════════════════════════
                "frame_selection": ("STRING", {
                    "default": "all", "multiline": False,
                    "tooltip": "Frames to process. Examples: all, 0-50, 0,5,10, 100-200."
                }),
                "init_from_sparse": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Start from COLMAP sparse points. Keep enabled for correct scene scale and faster convergence."
                }),
                
                # ═══════════════════════════════════════════════════════════════
                # OUTPUT SETTINGS [SHARED]
                # ═══════════════════════════════════════════════════════════════
                "filename_prefix": ("STRING", {
                    "default": "FreeFlow_GS", "multiline": False,
                    "tooltip": "Output file prefix. Final files are named like {prefix}_frame_0001.ply."
                }),
                "custom_output_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Custom output folder path. Leave empty to use ComfyUI default output directory."
                }),
                "use_symlinks": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Use symlinks for dataset images instead of copying. Faster and smaller disk usage."
                }),
                "cleanup_work_dirs": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Delete temporary per-frame work folders after completion to save disk space."
                }),
                
                # ═══════════════════════════════════════════════════════════════
                # DISTRIBUTED TRAINING [SHARED]
                # ═══════════════════════════════════════════════════════════════
                "distributed_anchor": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable distributed multi-machine workflow. One machine can produce the anchor, others consume it for consistent sequence initialization."
                }),
                "distributed_anchor_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Optional explicit anchor source path. Splatfacto accepts checkpoint dir/.ckpt/anchor metadata .json. Brush/OpenSplat accept anchor PLY path. Leave empty to auto-resolve from Distributed_Anchor."
                }),
                "distributed_anchor_frame": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Real frame ID to use as distributed anchor. If missing from selected range, it is inserted first in producer mode."
                }),
                "warmup_frames": ("INT", {
                    "default": 0, "min": 0, "max": 1000,
                    "tooltip": "Number of initial processed frames to keep as warmup-only outputs. Anchor frame is always preserved and never treated as warmup."
                }),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("ply_sequence_output",)
    FUNCTION = "execute_training_pipeline"
    CATEGORY = "FreeFlow"
    OUTPUT_NODE = True

    # ==================================================================================
    #   HELPER: PLY I/O 
    # ==================================================================================
    
    def _read_ply_positions(self, ply_path):
        """ Reads just the XYZ positions from a Binary PLY string. """
        try:
            with open(ply_path, "rb") as f:
                header_end = False
                num_verts = 0
                props = []
                while not header_end:
                    line = f.readline().decode('utf-8').strip()
                    if line == "end_header": header_end = True
                    elif line.startswith("element vertex"): num_verts = int(line.split()[-1])
                    elif line.startswith("property float"): props.append(line.split()[-1])
                
                if 'x' not in props or 'y' not in props or 'z' not in props: return None
                dtype = np.dtype([ (p, 'f4') for p in props ])
                data = np.fromfile(f, dtype=dtype, count=num_verts)
                # Combine XYZ
                return np.stack([data['x'], data['y'], data['z']], axis=1)
        except Exception as e:
            print(f"Error reading PLY: {e}")
            return None

    def _write_ply(self, output_path, xyz, original_ply_path):
        """ Writes a New PLY by copying header but replacing XYZ. """
        try:
            with open(original_ply_path, "rb") as f_src:
                header_lines = []
                num_verts = 0
                props = []
                while True:
                    line = f_src.readline()
                    header_lines.append(line)
                    s_line = line.decode('utf-8').strip()
                    if s_line == "end_header": break
                    elif s_line.startswith("element vertex"): num_verts = int(s_line.split()[-1])
                    elif s_line.startswith("property float"): props.append(s_line.split()[-1])
                
                dtype = np.dtype([ (p, 'f4') for p in props ])
                data = np.fromfile(f_src, dtype=dtype, count=num_verts)
            
            if len(xyz) != num_verts:
                if len(xyz) < num_verts:
                     print(f"   ⚠️ Warning: Truncating PLY from {num_verts} to {len(xyz)}")
                     data = data[:len(xyz)]
                     for i, l in enumerate(header_lines):
                         if l.decode().startswith("element vertex"):
                             header_lines[i] = f"element vertex {len(xyz)}\n".encode()
                             break
                else:
                     print(f"   ⚠️ Warning: Size Mismatch {len(xyz)} vs {num_verts}. Aborting Warp Write.")
                     return False
    
            data['x'] = xyz[:, 0]; data['y'] = xyz[:, 1]; data['z'] = xyz[:, 2]
            with open(output_path, "wb") as f_dst:
                f_dst.writelines(header_lines)
                data.tofile(f_dst)
            return True
        except Exception as e:
            print(f"Error writing warped PLY: {e}")
            return False

    # ==================================================================================
    #   CORE ALGORITHM: BINDING & WARPING
    # ==================================================================================

    def _bind_to_mesh(self, ply_path, mesh_verts):
        """ Frame 0: Binds every Gaussian in 'ply_path' to the nearest vertex in 'mesh_verts'. """
        print("   🔗 Binding Gaussians to Mesh...")
        gs_pos = self._read_ply_positions(ply_path)
        if gs_pos is None: return None
        try:
            tree = scipy.spatial.cKDTree(mesh_verts)
            dists, indices = tree.query(gs_pos, k=1, workers=-1)
            offsets = gs_pos - mesh_verts[indices]
            print(f"   ✅ Bound {len(gs_pos)} gaussians to nearest of {len(mesh_verts)} mesh vertices.")
            return { 'indices': indices, 'offsets': offsets, 'original_count': len(gs_pos) }
        except Exception as e:
            print(f"   ❌ Binding Failed: {e}"); return None

    def _warp_ply(self, source_ply, binding_data, current_mesh_verts, output_path):
        """ Frame N: Warps the source_ply to new positions based on current_mesh_verts. """
        indices = binding_data['indices']
        offsets = binding_data['offsets']
        if len(current_mesh_verts) == 0: return False
        try:
            new_xyz = current_mesh_verts[indices] + offsets
            return self._write_ply(output_path, new_xyz, source_ply)
        except Exception as e:
            print(f"   ❌ Warp Error: {e}"); return False

    # ==================================================================================
    #   DATASET PREP (Common)
    # ==================================================================================
    # ... Helper utils can stay ...
    
    def _extract_frame_numbers(self, multicam_feed):
        # ... same as before
        if not multicam_feed: return []
        best_cam = max(multicam_feed, key=lambda k: len(multicam_feed[k]))
        file_paths = multicam_feed[best_cam]
        frame_nums = []
        for fp in file_paths:
            fname = Path(fp).stem
            matches = re.findall(r'\d+', fname)
            if matches: frame_nums.append(int(matches[-1]))
            else: frame_nums.append(len(frame_nums))
        return frame_nums

    def _parse_frames(self, frame_str, available_frames):
        # ... same as before
        frame_str = str(frame_str).lower().strip()
        max_idx = len(available_frames)
        if frame_str == "all" or frame_str == "*": return list(range(max_idx))
        indices = set()
        parts = frame_str.split(',')
        desired_frames = set()
        ranges = []
        for part in parts:
            part = part.strip(); 
            if not part: continue
            if '-' in part:
                 try: s, e = map(int, part.split('-')); ranges.append((s, e))
                 except: pass
            else:
                 try: desired_frames.add(int(part))
                 except: pass
        final_indices = []
        for idx, frame_num in enumerate(available_frames):
            match = False
            if frame_num in desired_frames: match = True
            else:
                for (s, e) in ranges:
                    if s <= frame_num <= e: match = True; break
            if match: final_indices.append(idx)
        return sorted(final_indices)

    def _prepare_dataset(self, frame_work_dir, multicam_feed, frame_idx, anchor_sparse, filename_map, use_symlinks=True, mask_engine=None, prev_images_paths=None, masking_method=None):
        """
        Prepares standard COLMAP format:
        /images/ (symlinked from input)
        /sparse/0/ (copied from anchor)
        /masks/ (optional)
        """
        img_dir = frame_work_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        cameras = list(multicam_feed.keys())
        active_cams = 0
        
        def safe_link(src, dst, link=True):
            if link:
                try: 
                    if dst.exists(): dst.unlink()
                    os.symlink(src, dst); return True
                except: pass 
            shutil.copy2(src, dst); return False

        for cam in cameras:
            frames = multicam_feed[cam]
            if frame_idx < len(frames):
                src_img = Path(frames[frame_idx])
                dst_name = filename_map.get(cam, f"{cam}{src_img.suffix}")
                dst_img = img_dir / dst_name
                
                # Diagnostic: Verify linking for first few cameras
                if active_cams < 3:
                     print(f"[FreeFlow] Data Prep: Linked {src_img.name} -> {dst_img.name}")
                
                dst_img = img_dir / dst_name
                
                mask_applied = False
                if mask_engine and prev_images_paths and prev_images_paths.get(cam):
                    # Compute Opical Flow Mask
                    prev_img = prev_images_paths[cam]
                    mask_path = frame_work_dir / f"mask_{cam}.png"
                    generated_mask = mask_engine.compute_mask(src_img, prev_img, mask_path, method=masking_method)
                    if generated_mask:
                        mask_dir = frame_work_dir / "masks"
                        mask_dir.mkdir(exist_ok=True)
                        dst_mask = mask_dir / f"{Path(dst_name).stem}.png"
                        shutil.move(str(generated_mask), dst_mask)
                        mask_applied = True
                
                # If mask applied, we might need copy not link? No, input img is same.
                safe_link(src_img, dst_img, use_symlinks)
                
                if prev_images_paths is not None: prev_images_paths[cam] = src_img
                active_cams += 1
                
        sparse_dest = frame_work_dir / "sparse"
        sparse_dest.mkdir(exist_ok=True)
        shutil.copytree(anchor_sparse, sparse_dest / "0", dirs_exist_ok=True)
        return active_cams

    def _get_anchor_filename_map(self, colmap_anchor, cameras):
        anchor_images = Path(colmap_anchor) / "images"
        if not anchor_images.exists(): return {}
        mapping = {}
        files = [f.name for f in anchor_images.iterdir() if f.is_file()]
        for cam in cameras:
            matches = [f for f in files if f.startswith(f"{cam}_")]
            if matches: mapping[cam] = matches[0]; continue
            matches = [f for f in files if cam in f]
            if matches: mapping[cam] = matches[0]; continue
        return mapping

    def _pbar_simulator(self, pbar, iterations, seconds_per_step, start_step, process, is_first_frame=False, visualize_training="Off", frame_name=""):
        """
        Helper to simulate progress since Brush doesn't give stdout progress easily.
        On first frame, performs calibration to measure actual GPU speed.
        In "Save Preview Images" mode, uses file-based progress from preview image count.
        """
        start_time = time.time()
        last_pct = 0
        calibration_done = False
        calibration_sample_time = 30  # Seconds to sample for calibration
        use_file_based_progress = (visualize_training == "Save Preview Images")
        
        # Initial estimate print
        if use_file_based_progress:
            print(f"   📊 Using file-based progress (tracking preview images)...")
        else:
            est_total = iterations * seconds_per_step
            print(f"   ⏱️ Estimated training time: ~{est_total:.0f}s for {iterations} steps")

        # Loop until process finishes
        while process.poll() is None:
            elapsed = time.time() - start_time
            
            # --- PROGRESS CALCULATION ---
            if use_file_based_progress and frame_name:
                # Use REAL step count from preview file watcher
                actual_step = self._preview_step_counter.get(frame_name, 0)
                estimated_step = actual_step
            else:
                # --- CALIBRATION PHASE (First frame only) ---
                if is_first_frame and not calibration_done and elapsed >= calibration_sample_time:
                    calibration_done = True
                
                # Calculate estimated current step based on elapsed time
                estimated_step = min(int(elapsed / seconds_per_step), iterations)
            
            pct = (estimated_step / iterations) * 100
            
            # Logging update every 10%
            if pct - last_pct >= 10:
                print(f"   └─ 🔄 Step {estimated_step}/{iterations} ({pct:.0f}%) [elapsed: {elapsed:.0f}s]")
                last_pct = pct
                
                # Sync ComfyUI ProgressBar
                if pbar: 
                    pass  # Already updating via pbar.update below

            # Regular update loop
            time.sleep(1)  # Check every second
            if pbar:
                pbar.update(1)
        
        # --- POST-TRAINING: Calculate ACTUAL speed for calibration ---
        total_elapsed = time.time() - start_time
        
        if is_first_frame and total_elapsed > 5:  # Only calibrate if ran for reasonable time
            actual_seconds_per_step = total_elapsed / iterations
            self._cached_seconds_per_step = actual_seconds_per_step
            print(f"   📊 Calibrated: {actual_seconds_per_step:.4f}s/step (Total: {total_elapsed:.1f}s for {iterations} steps)")
        elif total_elapsed > 5:
            # Update calibration on subsequent frames too (rolling average)
            new_rate = total_elapsed / iterations
            if self._cached_seconds_per_step:
                # Weighted average: 70% old, 30% new (smooth out variance)
                self._cached_seconds_per_step = 0.7 * self._cached_seconds_per_step + 0.3 * new_rate
            else:
                self._cached_seconds_per_step = new_rate

    def _apply_temporal_smoothing(self, output_dir, prefix, indices, multicam_feed, cameras,
                                  realign_topology=True, do_smoothing=True, source_dir=None, out_subdir=None):
        """
        Applies topology realignment and optional Savitzky-Golay smoothing.

        Writes post-processed outputs into dedicated subfolders to preserve raw training outputs:
        - realigned/: topology realignment only
        - smoothed/: temporal smoothing (optionally from realigned source)

        Args:
            realign_topology: Whether to run KD-tree point-ID realignment.
            do_smoothing: Whether to apply Savitzky-Golay smoothing.
            source_dir: Folder containing source PLYs (defaults to output_dir root).
            out_subdir: Destination subfolder name (defaults by mode).

        Returns:
            Path to output folder on success, None if skipped/failed.
        """
        if not indices:
            return None

        source_base = Path(source_dir) if source_dir else output_dir
        if out_subdir is None:
            out_subdir = "smoothed" if do_smoothing else "realigned"

        out_dir = output_dir / out_subdir
        out_dir.mkdir(exist_ok=True)
        FreeFlowUtils.log(f"   📁 {out_subdir.capitalize()} output will be saved to: {out_dir}")

        # Extract REAL frame IDs from source image filenames
        first_cam = cameras[0]
        real_frame_ids = []
        for i in indices:
            if i < len(multicam_feed[first_cam]):
                real_id = self._extract_frame_number(multicam_feed[first_cam][i])
            else:
                real_id = i
            real_frame_ids.append(real_id)

        source_files = [source_base / f"{prefix}_frame_{fid:04d}.ply" for fid in real_frame_ids]
        output_files = [out_dir / f"{prefix}_frame_{fid:04d}.ply" for fid in real_frame_ids]

        # Pair files so missing frames never shift output indexing
        file_pairs = []
        for fid, src, dst in zip(real_frame_ids, source_files, output_files):
            if not src.exists():
                FreeFlowUtils.log(f"Skipping missing source frame {fid:04d}: {src}", "WARN")
                continue
            file_pairs.append((fid, src, dst))

        if not file_pairs:
            FreeFlowUtils.log("No valid source frames found for post-processing.", "WARN")
            return None

        # Parse PLY layout helper
        def get_ply_meta(path):
            with open(path, "rb") as f:
                header = b""
                while True:
                    line = f.readline()
                    if not line:
                        break
                    header += line
                    if line.strip() == b"end_header":
                        break
                header_str = header.decode("utf-8", errors="ignore")
                m = re.search(r"element vertex (\d+)", header_str)
                if not m:
                    return None, None
                return len(header), int(m.group(1))

        # Load/validate all frame buffers with a consistent layout
        all_data = []
        valid_pairs = []
        header_len = None
        vertex_count = None
        stride = None

        for fid, src, dst in file_pairs:
            local_header_len, local_vertex_count = get_ply_meta(src)
            if local_header_len is None or local_vertex_count is None or local_vertex_count <= 0:
                FreeFlowUtils.log(f"Invalid PLY header for frame {fid:04d}: {src}", "WARN")
                continue

            with open(src, "rb") as f:
                data = bytearray(f.read())

            payload_len = len(data) - local_header_len
            if payload_len <= 0 or payload_len % local_vertex_count != 0:
                FreeFlowUtils.log(f"Invalid PLY payload for frame {fid:04d}: {src}", "WARN")
                continue

            local_stride = payload_len // local_vertex_count

            if header_len is None:
                header_len = local_header_len
                vertex_count = local_vertex_count
                stride = local_stride
            elif local_header_len != header_len or local_vertex_count != vertex_count or local_stride != stride:
                FreeFlowUtils.log(
                    f"Layout mismatch for frame {fid:04d}; expected (hdr={header_len}, verts={vertex_count}, stride={stride}) "
                    f"got (hdr={local_header_len}, verts={local_vertex_count}, stride={local_stride}). Skipping.",
                    "WARN",
                )
                continue

            valid_pairs.append((fid, src, dst))
            all_data.append(data)

        if not all_data:
            FreeFlowUtils.log("No compatible frames available for post-processing.", "WARN")
            return None

        if header_len is None or vertex_count is None or stride is None:
            FreeFlowUtils.log("Post-processing metadata could not be resolved.", "WARN")
            return None

        header_len = int(header_len)
        vertex_count = int(vertex_count)
        stride = int(stride)

        T = len(all_data)
        if do_smoothing and T < 4:
            FreeFlowUtils.log("Smoothing skipped: need at least 4 valid frames.", "WARN")
            return None

        points = np.zeros((T, vertex_count, 3), dtype=np.float32)

        for t, data in enumerate(all_data):
            raw_view = np.frombuffer(data, dtype=np.uint8, offset=header_len)
            reshaped = raw_view.reshape(vertex_count, stride)
            xyz_bytes = reshaped[:, :12]
            xyz_floats = xyz_bytes.view(dtype=np.float32).reshape(vertex_count, 3)
            points[t] = xyz_floats

        if realign_topology:
            FreeFlowUtils.log("   🔧 Realigning topology using KD-Tree matching...")
            reorder_maps = self._compute_topology_realignment(points)

            for t in range(T):
                indices_map = reorder_maps[t]
                points[t] = points[t][indices_map]
                all_data[t] = self._reorder_ply_binary(all_data[t], indices_map, header_len, stride, vertex_count)

            FreeFlowUtils.log(f"   ✅ Realigned {T} frames")

        points_to_write = points
        if do_smoothing:
            window_length = min(7, T if T % 2 == 1 else T - 1)
            if window_length < 3:
                window_length = 3
            points_to_write = scipy.signal.savgol_filter(points, window_length=window_length, polyorder=2, axis=0)

        for t, data in enumerate(all_data):
            raw_view = np.frombuffer(data, dtype=np.uint8, offset=header_len)
            reshaped = raw_view.reshape(vertex_count, stride)
            target_bytes = reshaped[:, :12]

            new_xyz = points_to_write[t].astype(np.float32)
            new_uint8 = new_xyz.view(np.uint8).reshape(vertex_count, 12)
            np.copyto(target_bytes, new_uint8)

            output_path = valid_pairs[t][2]
            with open(output_path, "wb") as f:
                f.write(data)

        return out_dir

    def _compute_topology_realignment(self, all_positions):
        """
        Computes reorder indices for each frame using incremental KD-Tree matching.
        
        Uses "chained" matching where each frame is matched to the previous frame,
        not to Frame 0. This prevents drift over long sequences with large motion.
        
        Args:
            all_positions: np.ndarray of shape (T, N, 3) where T=frames, N=points
            
        Returns:
            List of T index arrays. Each array maps original indices to reordered indices.
        """
        T, N, _ = all_positions.shape
        reorder_maps = [np.arange(N)]  # Frame 0: Identity mapping (reference)
        
        ref_positions = all_positions[0].copy()  # Start with Frame 0 as reference
        
        for t in range(1, T):
            current_positions = all_positions[t]
            
            # Build tree of CURRENT frame's positions
            tree = cKDTree(current_positions)
            
            # Query: For each point in REFERENCE, find closest point in CURRENT
            # This gives us: "Where in current frame is the point that was at ref[i]?"
            distances, indices = tree.query(ref_positions, k=1)
            
            # Store the mapping
            reorder_maps.append(indices)
            
            # Update reference for next iteration (Incremental Chaining)
            # The new reference is the CURRENT positions reordered to match the old reference
            ref_positions = current_positions[indices]
        
        return reorder_maps

    def _reorder_ply_binary(self, data, indices, header_len, stride, vertex_count):
        """
        Reorders all vertex data in a PLY binary buffer according to the given index mapping.
        Preserves the header and reorders all vertex properties (XYZ, colors, SH, etc.).
        
        Args:
            data: bytearray containing the full PLY file (header + binary data)
            indices: np.ndarray of integers, new order for vertices
            header_len: int, length of the PLY header in bytes
            stride: int, bytes per vertex
            vertex_count: int, number of vertices
            
        Returns:
            New bytearray with reordered vertex data
        """
        # Extract header (unchanged)
        header = bytes(data[:header_len])
        
        # Extract vertex data as numpy array for efficient reordering
        vertex_data = np.frombuffer(data, dtype=np.uint8, offset=header_len)
        vertex_data = vertex_data.reshape(vertex_count, stride)
        
        # Reorder using fancy indexing
        reordered_data = vertex_data[indices].copy()
        
        # Reconstruct the full buffer
        result = bytearray(header)
        result.extend(reordered_data.tobytes())
        
        return result

    def _convert_ply_to_points3d(self, ply_path, out_path):
        """
        Reads a Gaussian PLY (binary) and writes a simplified COLMAP points3D.txt (text).
        Handling both 'float' and 'uchar' properties for robust stride calculation.
        """
        try:
            with open(ply_path, "rb") as f:
                header = b""
                # Use readline to strictly consume complete lines including newlines
                while True:
                    line = f.readline()
                    header += line
                    if line.strip() == b"end_header":
                        break
                    if not line:  # EOF safety
                        break
                
                header_str = header.decode("utf-8", errors="ignore")
                
                # 1. Parse Vertex Count
                m = re.search(r"element vertex (\d+)", header_str)
                if not m: return False
                vertex_count = int(m.group(1))
                
                # 2. Parse Properties to calculate Stride and Offsets
                properties = []  # List of (name, type, size)
                
                x_offset = -1; y_offset = -1; z_offset = -1
                r_offset = -1; g_offset = -1; b_offset = -1
                
                r_type = 'f'; g_type = 'f'; b_type = 'f'  # Default to float
                
                current_offset = 0
                
                for line in header_str.split('\n'):
                    if line.startswith("property"):
                        parts = line.split()
                        if len(parts) < 3: continue
                        ptype = parts[1]
                        pname = parts[2]
                        
                        size = 4  # default float
                        fmt = 'f'
                        if ptype in ['uchar', 'uint8']: 
                            size = 1
                            fmt = 'B'
                        elif ptype in ['float', 'float32']: 
                            size = 4
                            fmt = 'f'
                        elif ptype in ['double', 'float64']:
                            size = 8
                            fmt = 'd'
                            
                        properties.append((pname, fmt, size))
                        
                        # Check if this property is one we need
                        if pname == 'x': x_offset = current_offset
                        elif pname == 'y': y_offset = current_offset
                        elif pname == 'z': z_offset = current_offset
                        elif pname in ['red', 'f_dc_0']: 
                            r_offset = current_offset; r_type = fmt
                        elif pname in ['green', 'f_dc_1']: 
                            g_offset = current_offset; g_type = fmt
                        elif pname in ['blue', 'f_dc_2']: 
                            b_offset = current_offset; b_type = fmt
                            
                        current_offset += size
                
                stride = current_offset
                
                # 3. Read Data
                points = []
                SH_C0 = 0.28209479177387814
                
                # Buffer read entire body for speed
                body = f.read()
                
                batch_limit = 50000 
                step = max(1, vertex_count // batch_limit)
                
                for i in range(0, vertex_count, step):
                    base = i * stride
                    if base + stride > len(body): break
                    
                    try:
                        x = struct.unpack_from('<f', body, base + x_offset)[0]
                        y = struct.unpack_from('<f', body, base + y_offset)[0]
                        z = struct.unpack_from('<f', body, base + z_offset)[0]
                        
                        if r_offset >= 0:
                            if r_type == 'B':
                                r = struct.unpack_from('<B', body, base + r_offset)[0]
                                g = struct.unpack_from('<B', body, base + g_offset)[0]
                                b = struct.unpack_from('<B', body, base + b_offset)[0]
                            else:
                                # Float colors (likely SH or 0-1)
                                rf = struct.unpack_from('<f', body, base + r_offset)[0]
                                gf = struct.unpack_from('<f', body, base + g_offset)[0]
                                bf = struct.unpack_from('<f', body, base + b_offset)[0]
                                
                                # Simple heuristic for SH vs 0-1
                                if abs(rf) > 1.0 or abs(rf) < -1.0:
                                    rf = rf * SH_C0 + 0.5
                                    gf = gf * SH_C0 + 0.5
                                    bf = bf * SH_C0 + 0.5
                                 
                                r = min(255, max(0, int(rf * 255)))
                                g = min(255, max(0, int(gf * 255)))
                                b = min(255, max(0, int(bf * 255)))
                        else:
                            r, g, b = 127, 127, 127
                        
                        # Check for NaN to prevent crash
                        if x != x or y != y or z != z: continue 
                        
                        points.append((i+1, x, y, z, r, g, b))
                        
                    except Exception:
                        continue
            
            FreeFlowUtils.log(f"Extracted {len(points)}/{vertex_count} points for Warm Start", "INFO")
            if len(points) == 0:
                FreeFlowUtils.log("CRITICAL: No points extracted! Check stride/format.", "ERROR")
            
            # 4. Write COLMAP points3D.txt
            with open(out_path.with_suffix(".txt"), "w") as f_out: 
                f_out.write("# 3D point list with one line of data per point:\n")
                f_out.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                f_out.write(f"# Number of points: {len(points)}, mean track length: 0\n")
                
                for p in points:
                    f_out.write(f"{p[0]} {p[1]:.6f} {p[2]:.6f} {p[3]:.6f} {p[4]} {p[5]} {p[6]} 0.0\n")
            
            # Delete points3D.bin if it exists — nerfstudio checks .bin FIRST
            # and would read the COLMAP sparse instead of our warm start positions
            bin_path = out_path.with_suffix(".bin")
            if bin_path.exists():
                bin_path.unlink()
                FreeFlowUtils.log("Deleted points3D.bin (nerfstudio will read our .txt)", "INFO")
                    
            return True

        except Exception as e:
            FreeFlowUtils.log(f"Partial Warm Start failed: {e}", "WARN")
            return False
    
    def _monitor_previews_task(self, process, output_dir, unique_id, preview_camera_filter, frame_name):
        """
        Helper task to monitor and organize previews for a specific training process.
        Running in a thread, so it must be robust.
        """
        from server import PromptServer
        
        last_img_time = 0
        
        # 1. Setup Organized Folder for User (Ensure it exists for this process)
        previews_dir = output_dir / "TrainingPreviews" / frame_name
        previews_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Setup Temp Folder for Web Serving
        temp_serve_dir = Path(folder_paths.get_temp_directory()) / "FreeFlow_Previews"
        temp_serve_dir.mkdir(exist_ok=True)
        
        while process.poll() is None:
            # Monitor output_dir RECURSIVELY for new PNGs
            all_candidates = list(output_dir.rglob("*.png"))
            
            # Filter out images already in TrainingPreviews
            # AND Filter out images in working directories (masks, images, sparse)
            candidates = []
            for p in all_candidates:
                if "TrainingPreviews" in p.parts:
                    continue
                if not p.parent.name.startswith("eval_"):
                    continue
                candidates.append(p)
            
            if candidates:
                for cand in candidates:
                    try:
                        # Wait for write stability
                        mtime = cand.stat().st_mtime
                        if time.time() - mtime < 1.0:
                            continue
                            
                        parent_name = cand.parent.name  # eval_500
                        
                        # --- EXTRACT STEP NUMBER for progress tracking ---
                        step_match = re.search(r'eval_(\d+)', parent_name)
                        if step_match:
                            step_num = int(step_match.group(1))
                            # Update progress counter (keep highest step seen)
                            if frame_name in self._preview_step_counter:
                                if step_num > self._preview_step_counter[frame_name]:
                                    self._preview_step_counter[frame_name] = step_num
                            else:
                                self._preview_step_counter[frame_name] = step_num
                        
                        # Construct new name: step_500_cam01.png
                        new_name = f"{parent_name}_{cand.name}"
                        dest = previews_dir / new_name
                        
                        # Move it
                        if not dest.exists():
                            shutil.move(str(cand), str(dest))
                            
                            # Use this for preview IF it's likely the main camera
                            if mtime > last_img_time:
                                show_it = True
                                if preview_camera_filter and preview_camera_filter.strip():
                                    if preview_camera_filter.strip() not in new_name:
                                        show_it = False
                                
                                if show_it:
                                    last_img_time = mtime
                                    
                                    # Serve this one
                                    serve_name = f"preview_{unique_id}_{int(mtime)}.png"
                                    serve_path = temp_serve_dir / serve_name
                                    shutil.copy2(dest, serve_path)
                                    
                                    PromptServer.instance.send_sync("freeflow_preview", {
                                        "node": unique_id,
                                        "filename": serve_name,
                                        "subfolder": "FreeFlow_Previews",
                                        "type": "temp"
                                    })
                        
                        # Clean up empty parent folder
                        try:
                            cand.parent.rmdir() 
                        except:
                            pass 
                            
                    except Exception as e:
                        print(f"   [Preview Error] Processing {cand}: {e}")

            time.sleep(1)

    def _export_sparse_to_ply(self, sparse_dir, output_ply):
        """
        Export COLMAP sparse points to PLY for initialization.
        This provides initial Gaussian positions from tie-points.
        """
        points_file = sparse_dir / "points3D.bin"
        if not points_file.exists():
            return None
        
        try:
            with open(points_file, "rb") as f:
                num_points = struct.unpack("<Q", f.read(8))[0]
                
                vertices = []
                for _ in range(num_points):
                    point_id = struct.unpack("<Q", f.read(8))[0]
                    x, y, z = struct.unpack("<ddd", f.read(24))
                    r, g, b = struct.unpack("<BBB", f.read(3))
                    error = struct.unpack("<d", f.read(8))[0]
                    track_length = struct.unpack("<Q", f.read(8))[0]
                    f.read(track_length * 8)  # Skip track data
                    
                    vertices.append((x, y, z, r, g, b))
            
            # Write PLY (Binary Little Endian)
            with open(output_ply, "wb") as f:
                header = (
                    "ply\n"
                    "format binary_little_endian 1.0\n"
                    f"element vertex {len(vertices)}\n"
                    "property float x\n"
                    "property float y\n"
                    "property float z\n"
                    "property uchar red\n"
                    "property uchar green\n"
                    "property uchar blue\n"
                    "end_header\n"
                )
                f.write(header.encode("utf-8"))
                
                for v in vertices:
                    f.write(struct.pack("<fffBBB", float(v[0]), float(v[1]), float(v[2]), v[3], v[4], v[5]))
            
            FreeFlowUtils.log(f"Exported {len(vertices)} sparse points to {output_ply.name} (Binary)")
            return output_ply
            
        except Exception as e:
            FreeFlowUtils.log(f"Failed to export sparse PLY: {e}", "WARN")
            return None

    def _extract_frame_number(self, image_path):
        """
        Extract the real frame number from an image filename.
        Examples: 
            'cam01_frame_0001.jpg' -> 1
            'frame_0042.png' -> 42
            '00005.exr' -> 5
        Returns the last number found in the filename (common convention).
        """
        filename = Path(image_path).stem
        nums = re.findall(r'\d+', filename)
        if nums:
            return int(nums[-1])  # Last number is typically the frame number
        return 0  # Fallback

    def _is_checkpoint_dir(self, path: Path):
        """Return True if path looks like a nerfstudio checkpoint directory."""
        try:
            p = Path(path)
            return p.is_dir() and any(p.glob("step-*.ckpt"))
        except Exception:
            return False

    def _read_anchor_checkpoint_metadata(self, metadata_path: Path, output_dir: Path):
        """Read distributed anchor checkpoint metadata and resolve checkpoint directory."""
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            # Prefer relative path for cross-machine shared roots
            rel = meta.get("checkpoint_dir_rel")
            if rel:
                rel_candidate = output_dir / rel
                if self._is_checkpoint_dir(rel_candidate):
                    return rel_candidate

            abs_path = meta.get("checkpoint_dir")
            if abs_path:
                abs_candidate = Path(os.path.expanduser(str(abs_path)))
                if self._is_checkpoint_dir(abs_candidate):
                    return abs_candidate
        except Exception as e:
            FreeFlowUtils.log(f"Failed reading anchor metadata {metadata_path.name}: {e}", "WARN")

        return None

    def _resolve_distributed_checkpoint_dir(self, output_dir: Path, target_anchor_id: int, distributed_anchor_path: str):
        """
        Resolve distributed Splatfacto anchor checkpoint directory.

        Priority:
        1) explicit distributed_anchor_path (checkpoint dir / .ckpt / metadata json)
        2) output_dir/Distributed_Anchor/anchor_frame_XXXX.json metadata
        """
        explicit = str(distributed_anchor_path).strip()
        if explicit:
            p = Path(os.path.expanduser(explicit))

            # direct checkpoint dir
            if self._is_checkpoint_dir(p):
                return p

            # direct checkpoint file
            if p.is_file() and p.suffix.lower() == ".ckpt" and p.parent.exists():
                if self._is_checkpoint_dir(p.parent):
                    return p.parent

            # metadata json
            if p.is_file() and p.suffix.lower() == ".json":
                resolved = self._read_anchor_checkpoint_metadata(p, output_dir)
                if resolved:
                    return resolved

            # directory containing metadata json
            if p.is_dir():
                metadata_file = p / f"anchor_frame_{target_anchor_id:04d}.json"
                if metadata_file.exists():
                    resolved = self._read_anchor_checkpoint_metadata(metadata_file, output_dir)
                    if resolved:
                        return resolved

            FreeFlowUtils.log(
                f"Distributed anchor path provided but not a valid checkpoint source for Splatfacto: {p}",
                "WARN",
            )

        # default shared metadata location
        default_meta = output_dir / "Distributed_Anchor" / f"anchor_frame_{target_anchor_id:04d}.json"
        if default_meta.exists():
            resolved = self._read_anchor_checkpoint_metadata(default_meta, output_dir)
            if resolved:
                return resolved

        return None

    def _save_anchor_checkpoint_metadata(self, distributed_dir: Path, output_dir: Path, frame_id: int, checkpoint_dir: Path):
        """Write distributed anchor metadata containing checkpoint location."""
        try:
            distributed_dir.mkdir(parents=True, exist_ok=True)

            ckpt_dir = Path(checkpoint_dir).resolve()
            try:
                ckpt_rel = os.path.relpath(str(ckpt_dir), str(output_dir.resolve()))
            except Exception:
                ckpt_rel = None

            meta = {
                "frame_id": int(frame_id),
                "checkpoint_dir": str(ckpt_dir),
                "checkpoint_dir_rel": ckpt_rel,
                "created_at": int(time.time()),
                "format_version": 1,
            }

            meta_path = distributed_dir / f"anchor_frame_{int(frame_id):04d}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            return meta_path
        except Exception as e:
            FreeFlowUtils.log(f"Failed to save distributed anchor metadata: {e}", "WARN")
            return None

    # ==================================================================================
    #   MAIN EXECUTION
    # ==================================================================================

    def execute_training_pipeline(self, multicam_feed, colmap_anchor, 
                               engine_backend="Brush (Fast)",
                               guidance_mesh=None, topology_mode="Dynamic (Default-Flicker)",
                               apply_smoothing=False, realign_topology=True,
                               visualize_training="Off", preview_interval=500, eval_camera_index=10,
                               iterations=6000, 
                               **kwargs):
        
        # Safety: handle if a saved workflow somehow passes a boolean
        if isinstance(visualize_training, bool):
            visualize_training = "Spawn Native GUI" if visualize_training else "Off"
        # 1. Select Engine Strategy
        engine = None
        is_splatfacto = False
        is_opensplat = False
        
        if "Brush" in engine_backend:
            engine = BrushEngine()
        elif "Splatfacto" in engine_backend or "Nerfstudio" in engine_backend:
            engine = SplatfactoEngine()
            
            # Override visualizer based on the Splatfacto-specific boolean toggle
            # (visualize_training dropdown is hidden for Splatfacto in UI)
            if kwargs.get("splatfacto_viewer", False):
                visualize_training = "Spawn Native GUI"
            else:
                visualize_training = "Off"
            
            # Auto-install nerfstudio if not available
            if not engine.is_available():
                FreeFlowUtils.log("=" * 50)
                FreeFlowUtils.log("🔧 Splatfacto: First-time setup - installing nerfstudio...")
                FreeFlowUtils.log("   This may take 5-15 minutes. Please wait...")
                FreeFlowUtils.log("=" * 50)
                
                def install_progress(msg, pct):
                    FreeFlowUtils.log(f"   [{pct*100:.0f}%] {msg}")
                
                success = engine.install(progress_callback=install_progress)
                
                if not success or not engine.is_available():
                    status = engine.get_status()
                    raise RuntimeError(
                        f"Failed to install Splatfacto/nerfstudio. Status: {status}\n"
                        "Try manual install: from nodes.modules.nerfstudio_env import NerfstudioEnvironment; "
                        "NerfstudioEnvironment.create_venv()"
                    )
                FreeFlowUtils.log("✅ Splatfacto installation complete!")
            
            is_splatfacto = True
            FreeFlowUtils.log(f"   🚀 Using Splatfacto Engine v{engine.get_version()}")
        elif "OpenSplat" in engine_backend:
            if not OPENSPLAT_AVAILABLE:
                raise RuntimeError(
                    "OpenSplat backend module not available. Check engines/__init__.py"
                )
            engine = OpenSplatEngine()
            if not engine.is_available():
                raise RuntimeError(
                    f"OpenSplat binary not found. {OpenSplatEngine.get_install_instructions()}"
                )
            is_opensplat = True
            FreeFlowUtils.log(f"   🚀 Using OpenSplat Engine (GPU: {engine.get_status().get('gpu_backend', 'unknown')})")
        elif "GSplat" in engine_backend:
            raise NotImplementedError("GSplat backend coming soon (Phase 3).")
        else:
            raise ValueError(f"Unknown engine: {engine_backend}")
             
        # 2. Setup Output
        filename_prefix = kwargs.get("filename_prefix", "FreeFlow_4D")
        custom_out = kwargs.get("custom_output_path", "")
        if custom_out.strip(): output_dir = Path(os.path.expanduser(custom_out.strip()))
        else: output_dir = Path(folder_paths.get_output_directory()) / f"{filename_prefix}_{int(time.time())}"
        output_dir.mkdir(parents=True, exist_ok=True)
        FreeFlowUtils.log(f"✅ Output Directory set to: {output_dir}")
        
        # 3. Helpers
        masking_method = kwargs.get("masking_method", "None (No Masking)")
        if masking_method == "None (No Masking)":
            mask_engine = None
            FreeFlowUtils.log("Masking: Disabled (No Masking)", "INFO")
        else:
            from .modules.motion_masking import MotionMasking
            mask_engine = MotionMasking(sensitivity=kwargs.get("motion_sensitivity", 0.3))
            FreeFlowUtils.log(f"Masking: {masking_method} (sensitivity={kwargs.get('motion_sensitivity', 0.3)})", "INFO")
        
        # 4. Frame Management
        cameras = list(multicam_feed.keys())
        total_frames = max(len(multicam_feed[cam]) for cam in cameras)
        all_frame_numbers = self._extract_frame_numbers(multicam_feed)
        if len(all_frame_numbers) != total_frames: all_frame_numbers = list(range(total_frames))
        
        indices_to_process = self._parse_frames(kwargs.get("frame_selection", "all"), all_frame_numbers)
        if not indices_to_process:
            raise ValueError("No valid frames selected for training.")
        
        # 5. State
        binding_data = None
        prev_ply_path = None
        anchor_sparse = Path(colmap_anchor) / "sparse" / "0"
        filename_map = self._get_anchor_filename_map(colmap_anchor, cameras)
        prev_images_paths = {k: None for k in cameras}
        
        # Distributed anchor settings
        distributed_anchor = kwargs.get("distributed_anchor", False)
        distributed_anchor_path = kwargs.get("distributed_anchor_path", "")
        distributed_anchor_frame = kwargs.get("distributed_anchor_frame", "")
        warmup_frames = kwargs.get("warmup_frames", 0)
        unique_id = kwargs.get("unique_id", None)
        
        # Init from sparse settings
        init_from_sparse = kwargs.get("init_from_sparse", True)
        
        FreeFlowUtils.log("=" * 50)
        FreeFlowUtils.log(f"🎬 FreeFlow GS Engine (Mode: {engine_backend}) | Frames: {len(indices_to_process)}")
        if guidance_mesh: FreeFlowUtils.log(f"   🧬 Mesh Guidance: ENABLED ({len(guidance_mesh)} meshes loaded)")
        
        # Export sparse cloud for initialization (if enabled)
        sparse_init_ply = None
        if init_from_sparse:
            sparse_init_ply = output_dir / "sparse_init.ply"
            sparse_init_ply = self._export_sparse_to_ply(anchor_sparse, sparse_init_ply)
        
        # Default first-frame initialization source
        prev_ply = sparse_init_ply

        # Distributed anchor resolution
        distributed_mode = "off"  # off | producer | consumer
        distributed_anchor_source = None
        distributed_anchor_checkpoint_dir = None
        target_anchor_id = all_frame_numbers[indices_to_process[0]]

        if distributed_anchor:
            frame_str = str(distributed_anchor_frame).strip()
            if frame_str:
                if frame_str.isdigit():
                    target_anchor_id = int(frame_str)
                else:
                    FreeFlowUtils.log(
                        f"⚠️ Invalid distributed_anchor_frame='{distributed_anchor_frame}'. Using first selected frame as anchor.",
                        "WARN",
                    )

            # Map real frame ID -> first list index in full sequence
            real_to_list_index = {}
            for list_idx, fid in enumerate(all_frame_numbers):
                if fid not in real_to_list_index:
                    real_to_list_index[fid] = list_idx
            anchor_list_index = real_to_list_index.get(target_anchor_id)

            if is_splatfacto:
                # Splatfacto distributed mode uses checkpoint handoff (not PLY handoff)
                distributed_anchor_checkpoint_dir = self._resolve_distributed_checkpoint_dir(
                    output_dir, target_anchor_id, distributed_anchor_path
                )

                if distributed_anchor_checkpoint_dir:
                    distributed_mode = "consumer"
                    FreeFlowUtils.log(
                        f"⚓ Distributed CONSUMER mode (Splatfacto checkpoint): {distributed_anchor_checkpoint_dir}",
                        "INFO",
                    )
                    if hasattr(engine, "last_checkpoint_dir"):
                        engine.last_checkpoint_dir = distributed_anchor_checkpoint_dir

                    # If anchor frame is in selected range, process it first
                    if anchor_list_index is not None and anchor_list_index in indices_to_process:
                        if indices_to_process[0] != anchor_list_index:
                            indices_to_process.remove(anchor_list_index)
                            indices_to_process.insert(0, anchor_list_index)
                            print(
                                f"   ⚓ Distributed Priority: Moved Real Frame {target_anchor_id} (index {anchor_list_index}) to start of queue."
                            )
                else:
                    distributed_mode = "producer"
                    FreeFlowUtils.log(
                        f"⚓ Distributed PRODUCER mode: will generate anchor frame {target_anchor_id:04d}",
                        "INFO",
                    )

                    # Ensure requested anchor frame is generated first
                    if anchor_list_index is None:
                        raise RuntimeError(
                            f"Distributed anchor frame {target_anchor_id} not found in input sequence. "
                            "Please choose an existing frame ID."
                        )

                    if anchor_list_index not in indices_to_process:
                        indices_to_process.insert(0, anchor_list_index)
                        print(
                            f"   ⚓ Distributed Priority: Added Real Frame {target_anchor_id} (index {anchor_list_index}) as anchor at start."
                        )
                    elif indices_to_process[0] != anchor_list_index:
                        indices_to_process.remove(anchor_list_index)
                        indices_to_process.insert(0, anchor_list_index)
                        print(
                            f"   ⚓ Distributed Priority: Moved Real Frame {target_anchor_id} (index {anchor_list_index}) to start of queue."
                        )
            else:
                # Brush/OpenSplat distributed mode uses PLY handoff
                candidate_anchor_paths = []
                if str(distributed_anchor_path).strip():
                    candidate_anchor_paths.append(Path(os.path.expanduser(str(distributed_anchor_path).strip())))

                default_anchor_path = output_dir / "Distributed_Anchor" / f"anchor_frame_{target_anchor_id:04d}.ply"
                candidate_anchor_paths.append(default_anchor_path)

                for candidate in candidate_anchor_paths:
                    if candidate and candidate.exists():
                        distributed_anchor_source = candidate
                        break

                if distributed_anchor_source:
                    distributed_mode = "consumer"
                    prev_ply = distributed_anchor_source
                    FreeFlowUtils.log(
                        f"⚓ Distributed CONSUMER mode: using existing anchor {distributed_anchor_source}",
                        "INFO",
                    )

                    # If anchor frame is in selected range, process it first
                    if anchor_list_index is not None and anchor_list_index in indices_to_process:
                        if indices_to_process[0] != anchor_list_index:
                            indices_to_process.remove(anchor_list_index)
                            indices_to_process.insert(0, anchor_list_index)
                            print(
                                f"   ⚓ Distributed Priority: Moved Real Frame {target_anchor_id} (index {anchor_list_index}) to start of queue."
                            )
                else:
                    distributed_mode = "producer"
                    FreeFlowUtils.log(
                        f"⚓ Distributed PRODUCER mode: will generate anchor frame {target_anchor_id:04d}",
                        "INFO",
                    )

                    # Ensure requested anchor frame is generated first
                    if anchor_list_index is None:
                        raise RuntimeError(
                            f"Distributed anchor frame {target_anchor_id} not found in input sequence. "
                            "Please choose an existing frame ID."
                        )

                    if anchor_list_index not in indices_to_process:
                        indices_to_process.insert(0, anchor_list_index)
                        print(
                            f"   ⚓ Distributed Priority: Added Real Frame {target_anchor_id} (index {anchor_list_index}) as anchor at start."
                        )
                    elif indices_to_process[0] != anchor_list_index:
                        indices_to_process.remove(anchor_list_index)
                        indices_to_process.insert(0, anchor_list_index)
                        print(
                            f"   ⚓ Distributed Priority: Moved Real Frame {target_anchor_id} (index {anchor_list_index}) to start of queue."
                        )
        
        # Topology mode flag
        is_fixed_topology = "Fixed" in topology_mode

        # Initial-frame quality overrides (Fixed mode only)
        initial_iterations_override = int(kwargs.get("initial_iterations_override", 12000))
        initial_quality_preset = kwargs.get("initial_quality_preset", "High (Recommended)")

        # Consumer machines should continue from anchor checkpoint as-is (no first-frame densify boost)
        initial_override_allowed = not (distributed_anchor and distributed_mode == "consumer")

        # Progress Bar Setup (account for initial-frame iteration override)
        total_steps_global = len(indices_to_process) * iterations
        if is_fixed_topology and initial_override_allowed and initial_iterations_override > 0 and len(indices_to_process) > 0:
            total_steps_global += (initial_iterations_override - iterations)
        pbar = ProgressBar(total_steps_global) if ProgressBar else None
        current_step_global = 0
        
        # Rolling cleanup for Splatfacto: keep only latest frame work dir
        last_splatfacto_work_dir = None
        
        # 6. Loop
        for idx_seq, i in enumerate(indices_to_process):
            real_id = all_frame_numbers[i]
            frame_work_dir = output_dir / f"frame_{real_id:04d}_work"
            frame_work_dir.mkdir(parents=True, exist_ok=True)
            
            ply_out = output_dir / f"{filename_prefix}_frame_{real_id:04d}.ply"
            
            # A. Prepare Dataset (Images, Masks, Sparse)
            self._prepare_dataset(
                frame_work_dir, multicam_feed, i, anchor_sparse, filename_map, 
                kwargs.get("use_symlinks", True), mask_engine, prev_images_paths, kwargs.get("masking_method")
            )
            
            # B. Guidance / Warm Start Logic
            warped_ply_path = frame_work_dir / "warped_init.ply"
            init_ply = None
            
            # -- INITIALIZATION STRATEGY --
            # Priority 1: Warm Start (Previous Frame in this batch) -- Sequential Continuity
            # Priority 2: Distributed Anchor (If specified/found & No Prev Frame) -- Distributed Continuity
            # Priority 3: Sparse Cloud (Default) -- Single Machine Start
            
            sparse_target = frame_work_dir / "sparse" / "0"
            init_source_ply = None
            init_mode = "Sparse"

            # 1. Check Warm Start
            if prev_ply and prev_ply.exists():
                init_source_ply = prev_ply
                init_mode = "Warm Start"
            
            # 2. Check resolved Distributed Anchor source (Only if NOT warm starting)
            elif distributed_anchor_source and distributed_anchor_source.exists():
                init_source_ply = distributed_anchor_source
                init_mode = f"Distributed Anchor ({distributed_anchor_source.name})"
            
            # Perform Initialization Override
            if init_source_ply:
                converted_ok = self._convert_ply_to_points3d(init_source_ply, sparse_target / "points3D.txt")
                if converted_ok:
                    print(f"   🚀 Initializing Frame {i} from {init_mode}")
            
            # Frame 0 (Anchor) - Standard Init
            if idx_seq == 0:
                pass 
            
            # Subsequent Frames
            else:
                # 1. Try Mesh Warp (If Guidance + Binding exists)
                if guidance_mesh and prev_ply_path and binding_data:
                    if i < len(guidance_mesh):
                        current_mesh = guidance_mesh[i]
                        success = self._warp_ply(prev_ply_path, binding_data, current_mesh['vertices'], warped_ply_path)
                        if success:
                            init_ply = warped_ply_path
                            print(f"   🧬 Warped previous frame to match Mesh {real_id}")
                
                # 2. Fallback to Standard Warm Start
                if not init_ply and prev_ply_path:
                    init_ply = prev_ply_path
            
            # C. Execute Engine
            # Define frame_name early (used in params and callbacks)
            frame_name = f"{filename_prefix}_frame_{real_id:04d}"

            is_initial_quality_frame = (
                is_fixed_topology
                and idx_seq == 0
                and initial_override_allowed
            )

            frame_iterations = iterations
            if is_initial_quality_frame and initial_iterations_override > 0:
                frame_iterations = int(initial_iterations_override)
                print(f"   ⭐ [Fixed Init Override] Frame {real_id}: iterations {iterations} -> {frame_iterations}")
            
            # Param dict construction with Fixed Topology CLI flags
            params = {
                'iterations': frame_iterations,
                'splat_count': kwargs.get('splat_count', 500000),
                'learning_rate': kwargs.get('learning_rate', 0.00002),
                'sh_degree': kwargs.get('sh_degree', 3),
            }
            
            # Add Splatfacto-specific params if using that engine
            if is_splatfacto:
                # Setup preview directory only in Save Preview mode
                previews_dir = output_dir / "TrainingPreviews" / frame_name
                if visualize_training == "Save Preview Images":
                    previews_dir.mkdir(parents=True, exist_ok=True)
                
                # Create preview callback for websocket events
                def make_splatfacto_preview_callback(uid, frame_nm, prev_dir):
                    """Create a closure for the preview callback."""
                    last_preview_time = [0]  # Mutable to track last sent preview
                    
                    def preview_callback(image_path, step):
                        """Called when SplatfactoEngine finds a new preview image."""
                        try:
                            from server import PromptServer
                            
                            # Avoid sending too many previews (throttle to 1 per 3 seconds)
                            current_time = time.time()
                            if current_time - last_preview_time[0] < 3.0:
                                return
                            last_preview_time[0] = current_time
                            
                            # Copy to temp dir for web serving
                            temp_serve_dir = Path(folder_paths.get_temp_directory()) / "FreeFlow_Previews"
                            temp_serve_dir.mkdir(exist_ok=True)
                            
                            serve_name = f"preview_{uid}_{int(current_time)}.png"
                            serve_path = temp_serve_dir / serve_name
                            
                            # Copy the latest preview image
                            shutil.copy2(image_path, serve_path)
                            
                            # Send websocket event
                            PromptServer.instance.send_sync("freeflow_preview", {
                                "node": uid,
                                "filename": serve_name,
                                "subfolder": "FreeFlow_Previews",
                                "type": "temp"
                            })
                            print(f"   📸 [Splatfacto Preview] Step {step}: {serve_name}")
                        except Exception as e:
                            print(f"   ⚠️ [Splatfacto Preview] Error: {e}")
                    
                    return preview_callback
                
                splatfacto_preview_callback = None
                if visualize_training == "Save Preview Images" and unique_id:
                    splatfacto_preview_callback = make_splatfacto_preview_callback(
                        unique_id, frame_name, previews_dir
                    )
                
                params.update({
                    'splatfacto_variant': kwargs.get('splatfacto_variant', 'splatfacto'),
                    'cull_alpha_thresh': kwargs.get('cull_alpha_thresh', 0.22),
                    'densify_grad_thresh': kwargs.get('splatfacto_densify_grad_thresh', 0.0012),
                    'use_scale_regularization': kwargs.get('use_scale_regularization', True),
                    'refine_every': kwargs.get('refine_every', 120),
                    'warmup_length': kwargs.get('warmup_length', 800),
                    'num_downscales': kwargs.get('num_downscales', 1),
                    'cull_screen_size': kwargs.get('cull_screen_size', 0.15),
                    'split_screen_size': kwargs.get('split_screen_size', 0.05),
                    'sh_degree_interval': kwargs.get('sh_degree_interval', 1000),
                    'background_color': kwargs.get('background_color', 'black'),
                    'frame_idx': idx_seq,
                    'output_dir': output_dir / "nerfstudio_outputs",
                    'experiment_name': frame_name,
                    # Visualization settings - now properly wired to the dropdown
                    'visualize_training': visualize_training,
                    'preview_interval': preview_interval,
                    'viewer_port': 7007,  # Could be exposed as param later
                    # Preview monitoring (for "Save Preview Images" mode)
                    'previews_dir': str(previews_dir),
                    'preview_callback': splatfacto_preview_callback,
                })

                # Fixed-mode initial quality preset (first frame only)
                if is_initial_quality_frame:
                    preset = str(initial_quality_preset)
                    base_downscale = int(params.get('num_downscales', 1))
                    if preset == "High (Recommended)":
                        params['splatfacto_variant'] = 'splatfacto-big'
                        params['refine_every'] = 100
                        params['warmup_length'] = 700
                        params['densify_grad_thresh'] = 0.0012
                        params['num_downscales'] = max(base_downscale - 1, 0)
                        params['cull_alpha_thresh'] = 0.22
                        print("   ⭐ [Fixed Init Preset] High applied (variant=big, refine=100, warmup=700, densify=0.0012)")
                    elif preset == "Extreme (Slow)":
                        params['splatfacto_variant'] = 'splatfacto-big'
                        params['refine_every'] = 80
                        params['warmup_length'] = 500
                        params['densify_grad_thresh'] = 0.0010
                        params['num_downscales'] = 0
                        params['cull_alpha_thresh'] = 0.20
                        print("   ⭐ [Fixed Init Preset] Extreme applied (variant=big, refine=80, warmup=500, densify=0.0010)")

                # Fixed topology for Splatfacto: lock immediately after anchor frame
                # Frame 0: Establish topology (allow densification)
                # Frame 1+: Lock topology (no further densification)
                if is_fixed_topology and idx_seq > 0:
                    params['continue_cull_post_densification'] = False
                    print(f"   🔒 [Fixed Topology] Frame {real_id}: Splatfacto topology locked (no densification)")
            elif is_opensplat:
                # OpenSplat params - uses similar COLMAP format, simpler config
                params.update({
                    'output_dir': output_dir / "opensplat_outputs",
                    'downscale_factor': 1,  # Could be exposed as param later
                })
                # Fixed topology for OpenSplat
                if is_fixed_topology and idx_seq > 0:
                    # OpenSplat may have different flags for topology locking
                    print(f"   🔒 [Fixed Topology] Frame {i}: OpenSplat warm start (topology via resume)")
            else:
                # --- BRUSH ENGINE PARAMS ---
                # Add visualization params for Brush
                params.update({
                    'visualize_training': visualize_training,
                    'preview_interval': preview_interval,
                    'eval_camera_index': kwargs.get('eval_camera_index', 10),
                })
                
                # --- DENSIFICATION PARAMS (Dynamic mode only) ---
                # In Fixed mode, Frame 0 uses Brush defaults for maximum growth
                # Frame 1+ has densification disabled by Fixed Topology settings
                if not is_fixed_topology:
                    params.update({
                        'densification_interval': kwargs.get('densification_interval', 200),
                        'densify_grad_threshold': kwargs.get('densify_grad_threshold', 0.00004),
                        'growth_select_fraction': kwargs.get('growth_select_fraction', 0.1),
                    })
                
                # --- LEARNING RATE PARAMS ---
                params.update({
                    'feature_lr': kwargs.get('feature_lr', 0.0025),
                    'gaussian_lr': kwargs.get('gaussian_lr', 0.00016),
                    'opacity_lr': kwargs.get('opacity_lr', 0.01),
                })
                
                # --- REGULARIZATION PARAMS (help prevent camera drift) ---
                params.update({
                    'scale_loss_weight': kwargs.get('scale_loss_weight', 1e-8),
                })
                
                # Add Fixed Topology CLI flags for Brush if enabled AND not first frame
                # Frame 0: Establish topology (allow densification with defaults for max growth)
                # Frame 1+: Fixed topology (lock to preserve established structure)
                if is_fixed_topology and idx_seq > 0:
                    params['growth_stop_iter'] = 0  # Stop adding points immediately (no growth)
                    params['densification_interval'] = 999999  # Disable refinement (no split/clone/replace)
                    print(f"   🔒 [Fixed Topology] Frame {real_id}: Topology Locked (Strict Frozen Mode)")
            
            # Setup progress bar thread with adaptive timing
            if self._cached_seconds_per_step is not None:
                seconds_per_step = self._cached_seconds_per_step
                print(f"   ⏱️ Using calibrated speed: {seconds_per_step:.4f}s/step")
            else:
                # Default estimate (will be recalibrated)
                seconds_per_step = 0.035  # Conservative default
                print(f"   ⏱️ First frame - calibrating speed...")
            
            def pbar_update(proc):
                self._pbar_simulator(pbar, frame_iterations, seconds_per_step, current_step_global, proc, 
                                    idx_seq == 0, visualize_training, frame_name)
            
            callbacks = { 'pbar_func': pbar_update }
            
            # --- EXECUTE TRAINING ---
            # Brush returns (success, process) tuple for external monitoring
            # Splatfacto/OpenSplat return bool directly (they handle monitoring internally)
            if not is_splatfacto and not is_opensplat:
                # BRUSH ENGINE - need to handle process monitoring ourselves
                result = engine.train(
                    dataset_path=frame_work_dir, 
                    output_path=ply_out, 
                    params=params, 
                    prev_ply_path=init_ply, 
                    callback_data=callbacks
                )
                
                # Unpack result - Brush returns (success, process)
                if isinstance(result, tuple):
                    started_ok, process = result
                else:
                    # Fallback for old API
                    started_ok = result
                    process = None
                
                if not started_ok or process is None:
                    raise RuntimeError(f"Failed to start training for frame {real_id}")
                
                # --- PREVIEW MONITOR (Threaded) for Brush ---
                if visualize_training == "Save Preview Images" and unique_id:
                    self._preview_step_counter[frame_name] = 0
                    monitor_thread = threading.Thread(
                        target=self._monitor_previews_task,
                        args=(process, output_dir, unique_id, kwargs.get('preview_camera_filter', ''), frame_name)
                    )
                    monitor_thread.daemon = True
                    monitor_thread.start()
                
                # --- GUI AUTO-CLOSE MONITOR for "Spawn Native GUI" mode ---
                if visualize_training == "Spawn Native GUI":
                    def monitor_completion():
                        while process.poll() is None:
                            if ply_out.exists():
                                time.sleep(5)  # Wait for file to finish writing
                                if process.poll() is None:
                                    print("   ✅ Output found. Auto-Closing Brush GUI...")
                                    process.terminate()
                                    return
                            time.sleep(2)
                    
                    gui_monitor_thread = threading.Thread(target=monitor_completion)
                    gui_monitor_thread.daemon = True
                    gui_monitor_thread.start()
                
                # --- INTERRUPT-AWARE WAIT LOOP (matches AdaptiveEngine) ---
                import comfy.model_management
                try:
                    while process.poll() is None:
                        if comfy.model_management.processing_interrupted():
                            process.terminate()
                            print("🚫 Processing Interrupted by User. Terminating Brush...")
                            comfy.model_management.throw_exception_if_processing_interrupted()
                        time.sleep(0.5)
                    
                    # Check return code
                    if process.returncode != 0:
                        # For GUI mode, ignore error if output exists (user may have closed manually)
                        if visualize_training == "Spawn Native GUI" and ply_out.exists():
                            success = True
                        else:
                            raise RuntimeError(f"Brush process failed with return code {process.returncode}")
                    else:
                        success = ply_out.exists()
                        
                except KeyboardInterrupt:
                    process.terminate()
                    raise
                except Exception:
                    if process.poll() is None:
                        process.terminate()
                    raise
                
            else:
                # SPLATFACTO / OPENSPLAT - they handle monitoring internally
                # Initialize preview counter for Splatfacto
                if visualize_training == "Save Preview Images" and unique_id:
                    self._preview_step_counter[frame_name] = 0
                
                # Pass nerfstudio checkpoint from previous frame for warm start
                # (preserves all Gaussian attributes: positions, scales, rotations,
                # opacities, SH coefficients — by seeding model with checkpoint points
                # so Gaussian count matches, then loading weights)
                if is_splatfacto and hasattr(engine, 'last_checkpoint_dir') and engine.last_checkpoint_dir:
                    params['nerfstudio_checkpoint_dir'] = str(engine.last_checkpoint_dir)
                
                success = engine.train(
                    dataset_path=frame_work_dir, 
                    output_path=ply_out, 
                    params=params, 
                    prev_ply_path=init_ply, 
                    callback_data=callbacks
                )
            
            if not success:
                raise RuntimeError(f"Training failed for frame {real_id}")
            
            current_step_global += frame_iterations
            
            # --- DISTRIBUTED: Save Anchor ---
            if distributed_anchor and distributed_mode == "producer" and real_id == target_anchor_id and ply_out.exists():
                distributed_dir = output_dir / "Distributed_Anchor"
                distributed_dir.mkdir(exist_ok=True)
                anchor_filename = f"anchor_frame_{real_id:04d}.ply"
                anchor_dst = distributed_dir / anchor_filename
                shutil.copy(str(ply_out), str(anchor_dst))
                print(f"   ⚓ Saved Distributed Anchor to: {anchor_dst}")

                # For Splatfacto distributed workflows, also save checkpoint metadata.
                if is_splatfacto and hasattr(engine, "last_checkpoint_dir") and engine.last_checkpoint_dir:
                    meta_path = self._save_anchor_checkpoint_metadata(
                        distributed_dir,
                        output_dir,
                        real_id,
                        Path(engine.last_checkpoint_dir),
                    )
                    if meta_path:
                        print(f"   ⚓ Saved Distributed Checkpoint Metadata to: {meta_path}")
                elif is_splatfacto:
                    FreeFlowUtils.log(
                        "⚠️ Distributed anchor checkpoint metadata not saved: checkpoint dir unavailable.",
                        "WARN",
                    )

            # --- WARMUP LOGIC ---
            is_warmup = (idx_seq < warmup_frames)
            # Force keep if it's the anchor
            if distributed_anchor and real_id == target_anchor_id:
                is_warmup = False
            
            if is_warmup:
                warmup_dir = output_dir / "_warmup_temp"
                warmup_dir.mkdir(exist_ok=True)
                warmup_path = warmup_dir / ply_out.name
                shutil.move(str(ply_out), str(warmup_path))
                prev_ply = warmup_path
                print(f"   🔥 Warmup: Moved Frame {real_id} to temp storage.")
            else:
                prev_ply = ply_out
            
            prev_ply_path = prev_ply
            
            # D. Post-Train Binding (Frame 0)
            if idx_seq == 0 and guidance_mesh and i < len(guidance_mesh):
                 frame0_mesh = guidance_mesh[i]
                 binding_data = self._bind_to_mesh(ply_out, frame0_mesh['vertices'])

            # Cleanup
            if kwargs.get("cleanup_work_dirs", True):
                if is_splatfacto:
                    # Keep only one Splatfacto work dir at a time (latest frame).
                    # Safe because training/export for current frame already finished.
                    if last_splatfacto_work_dir and last_splatfacto_work_dir != frame_work_dir:
                        try:
                            shutil.rmtree(last_splatfacto_work_dir, ignore_errors=True)
                            print(f"   🗑️  Cleaned: {last_splatfacto_work_dir.name}")
                        except Exception as e:
                            print(f"   ⚠️  Failed to clean {last_splatfacto_work_dir.name}: {e}")
                    last_splatfacto_work_dir = frame_work_dir
                else:
                    # Brush/OpenSplat: cleanup immediately as usual
                    shutil.rmtree(frame_work_dir, ignore_errors=True)

        # --- POST-PROCESS: REALIGNMENT & SMOOTHING (Stable) ---
        realigned_dir = None
        smoothed_dir = None

        if is_fixed_topology:
            if realign_topology:
                FreeFlowUtils.log("🔗 Running Topology Realignment...")
                try:
                    realigned_dir = self._apply_temporal_smoothing(
                        output_dir,
                        filename_prefix,
                        indices_to_process,
                        multicam_feed,
                        cameras,
                        realign_topology=True,
                        do_smoothing=False,
                        source_dir=output_dir,
                        out_subdir="realigned",
                    )
                    if realigned_dir:
                        FreeFlowUtils.log(f"✅ Realignment Complete! Saved to: {realigned_dir}")
                except Exception as e:
                    FreeFlowUtils.log(f"Realignment failed: {e}", "WARN")

            if apply_smoothing:
                if len(indices_to_process) > 3:
                    FreeFlowUtils.log("🌊 Running Temporal Smoothing (Savitzky-Golay)...")
                    try:
                        smoothing_source = realigned_dir if realigned_dir else output_dir
                        smoothed_dir = self._apply_temporal_smoothing(
                            output_dir,
                            filename_prefix,
                            indices_to_process,
                            multicam_feed,
                            cameras,
                            realign_topology=False,  # Avoid double-realignment when smoothing from realigned source
                            do_smoothing=True,
                            source_dir=smoothing_source,
                            out_subdir="smoothed",
                        )
                        if smoothed_dir:
                            FreeFlowUtils.log(f"✅ Smoothing Complete! Saved to: {smoothed_dir}")
                    except Exception as e:
                        FreeFlowUtils.log(f"Smoothing failed: {e}", "WARN")
                else:
                    FreeFlowUtils.log("⚠️ Smoothing skipped: Sequence too short (<4 frames).", "WARN")
        else:
            if realign_topology:
                FreeFlowUtils.log("⚠️ Realignment skipped: Requires 'Fixed (Stable)' topology.", "WARN")
            if apply_smoothing:
                FreeFlowUtils.log("⚠️ Smoothing skipped: Requires 'Fixed (Stable)' topology.", "WARN")

        FreeFlowUtils.log("=" * 50)
        FreeFlowUtils.log(f"✅ Training Complete! Output: {output_dir}")
        FreeFlowUtils.log("=" * 50)

        return (str(output_dir),)
