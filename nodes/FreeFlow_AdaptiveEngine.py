"""
FreeFlow_AdaptiveEngine - Training Node with Sparse Cloud Initialization
Manages per-frame training with warm-start from COLMAP sparse reconstruction.
"""

import shutil
import time
import subprocess
import os
import sys
import re
import struct
import threading
import numpy as np
import scipy.signal

from pathlib import Path
import folder_paths
from ..utils import FreeFlowUtils

# Import ComfyUI Progress Bar
try:
    from comfy.utils import ProgressBar
except ImportError:
    ProgressBar = None


class FreeFlow_AdaptiveEngine:
    """
    Adaptive training engine for Gaussian Splats.
    Supports initialization from COLMAP sparse point cloud for faster convergence.
    """
    
    def __init__(self):
        self._last_error = None
        self._cached_seconds_per_step = None  # Adaptive: Measured on first frame
        self._preview_step_counter = {}  # File-based progress: {frame_name: current_step}

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # --- 1. Data Inputs ---
                "multicam_feed": ("MULTICAM_DICT",),
                "colmap_anchor": ("COLMAP_DATA",),
            },
            "optional": {
                # --- 2. Visualization & Monitoring ---
                "visualize_training": (["Off", "Save Preview Images", "Spawn Native GUI"], {"default": "Off", "tooltip": "Visual feedback during training. 'Off' runs silently. 'Save Preview Images' renders snapshots to disk. 'Spawn Native GUI' opens the Brush app window for real-time viewing."}),
                "preview_interval": ("INT", {"default": 500, "min": 100, "max": 5000, "tooltip": "How often to capture preview images (in training steps). Lower = more frequent updates but slower training."}),
                "preview_camera_filter": ("STRING", {"default": "", "multiline": False, "placeholder": "e.g. cam01 (Empty = Show All/Latest)", "tooltip": "Filter preview output to a specific camera by name. Leave empty to show all cameras or the latest available."}),
                "eval_camera_index": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "Render every Nth camera for preview. With 16 cameras: 1=all, 8=cam0+cam8. Higher values = faster previews."}),

                # --- 3. Topology Control ---
                "topology_mode": (["Dynamic (Default-Flicker)", "Fixed (Stable)"], {"default": "Dynamic (Default-Flicker)", "tooltip": "Dynamic: Points grow/shrink each frame (may flicker). Fixed: Lock point count after Frame 0 for smooth, consistent video output."}),
                "apply_smoothing": ("BOOLEAN", {"default": False, "tooltip": "Apply Savitzky-Golay temporal filter to smooth point positions across frames. Eliminates jitter. REQUIRES Fixed Topology mode."}),
                
                # --- 4. Core Model Parameters ---
                "splat_count": ("INT", {"default": 800000, "min": 1000, "max": 10000000, "tooltip": "Maximum number of Gaussian splats. Higher = more detail but slower rendering. 800k optimized for high-end film production."}),
                "sh_degree": ("INT", {"default": 3, "min": 0, "max": 3, "tooltip": "Spherical Harmonics degree controls view-dependent effects. 3=full specular/reflections (best quality). 0=flat diffuse colors (faster, more stable)."}),
                "iterations": ("INT", {"default": 30000, "min": 1000, "max": 100000, "tooltip": "Training steps per frame. More steps = higher quality and less flicker. 30k is good for production, 5k-10k for testing."}),
                
                # --- 5. Advanced Optimization ---
                "learning_rate": ("FLOAT", {"default": 0.00002, "min": 0.000001, "max": 0.001, "step": 0.000001, "tooltip": "Controls how fast point POSITIONS update. Lower = smoother, more stable results. Higher = faster convergence but may overshoot."}),
                "densification_interval": ("INT", {"default": 200, "min": 10, "max": 1000, "tooltip": "How often to add/remove points (refinement). Lower = more aggressive point growth. Higher = more conservative, stable topology."}),
                "densify_grad_threshold": ("FLOAT", {"default": 0.00015, "min": 0.000001, "max": 0.01, "step": 0.000001, "tooltip": "Gradient threshold to trigger point splitting. 0.00015 optimized for film: sufficient detail growth while maintaining temporal stability."}),
                "growth_select_fraction": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 1.0, "step": 0.01, "tooltip": "Fraction of high-gradient points selected for densification. 0.12 optimized for film: controlled growth for stable 4D sequences without flicker."}),
                
                # --- Learning Rates ---
                "feature_lr": ("FLOAT", {"default": 0.0025, "min": 0.0001, "max": 0.05, "step": 0.0001, "tooltip": "Controls how fast splat COLORS update. Set low (0.0025) for stable colors across frames. Higher values may cause color flickering."}),
                "gaussian_lr": ("FLOAT", {"default": 0.0003, "min": 0.00001, "max": 0.1, "step": 0.00001, "tooltip": "Controls how fast splat SIZE/SHAPE updates. Maps to Brush --lr-scale. 0.0003 optimized for film: faster convergence than ultra-low values while maintaining 4D stability."}),
                "opacity_lr": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.0001, "tooltip": "Learning rate for opacity values. Controls how fast transparency updates. Lower = more stable opacity."}),
                "scale_loss_weight": ("FLOAT", {"default": 1e-8, "min": 0.0, "max": 1e-5, "step": 1e-9, "tooltip": "Regularization weight for scale loss. Higher values constrain splat sizes. Helps prevent overly large splats."}),
                
                # --- 6. Initialization & Selection ---
                "frame_selection": ("STRING", {"default": "all", "multiline": False, "tooltip": "Which frames to process. Examples: 'all', '0-50', '0,5,10,15', '100-200'. Useful for testing or distributed rendering."}),
                "init_from_sparse": ("BOOLEAN", {"default": True, "tooltip": "Initialize first frame from COLMAP sparse point cloud. Essential for correct 3D scale and camera alignment. Only disable for testing."}),
                "masking_method": (["Optical Flow (Robust)", "Simple Diff (Fast)"], {"default": "Optical Flow (Robust)", "tooltip": "How to detect motion between frames. Optical Flow: accurate but slower. Simple Diff: fast but may flicker on subtle motion."}),
                "motion_sensitivity": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.1, "tooltip": "How aggressively to mask static areas. 0=no masking. 1=mask everything static. 0.3 is safe default to prevent background 'swimming'."}),
                
                # --- 7. Output & Storage ---
                "custom_output_path": ("STRING", {"default": "", "multiline": False, "placeholder": "leave empty for default output", "tooltip": "Custom folder for PLY output. Leave empty to use ComfyUI output folder. Use absolute path like D:/Project/Splats."}),
                "filename_prefix": ("STRING", {"default": "FreeFlow_Splat", "multiline": False, "tooltip": "Prefix for output PLY filenames. Result: {prefix}_frame_0001.ply, {prefix}_frame_0002.ply, etc."}),
                "use_symlinks": ("BOOLEAN", {"default": True, "label": "Use Symlinks (Save Disk)", "tooltip": "Use filesystem links instead of copying images. Saves gigabytes of disk space. Disable if you have permission issues on Windows."}),
                "cleanup_work_dirs": ("BOOLEAN", {"default": True, "label": "Auto-Delete Work Folders", "tooltip": "Delete temporary work folders after each frame. Saves disk space. Disable to keep intermediate files for debugging."}),
                
                # --- 9. Distributed Training ---
                "distributed_anchor": ("BOOLEAN", {"default": False, "tooltip": "Enable for multi-machine rendering. Saves Frame 0 as shared anchor so all machines start from identical point cloud."}),
                "distributed_anchor_path": ("STRING", {"default": "", "multiline": False, "placeholder": "path/to/anchor.ply (Empty = Auto)", "tooltip": "Path to load a pre-trained anchor PLY. Leave empty to auto-generate from Frame 0. Use for consistent multi-machine runs."}),
                "distributed_anchor_frame": ("STRING", {"default": "", "multiline": False, "placeholder": "Frame Number e.g. 0 (Empty = First Frame)", "tooltip": "Which frame number to use as anchor. Leave empty to use the first frame in your selection."}),
                "warmup_frames": ("INT", {"default": 0, "min": 0, "max": 1000, "tooltip": "Train N frames without saving output. Use for overlapping frame ranges in distributed mode to ensure smooth transitions."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("raw_output", "realigned_output", "smoothed_output")
    FUNCTION = "adapt_flow"
    CATEGORY = "FreeFlow"
    OUTPUT_NODE = True

    def _parse_frames(self, frame_str, max_frames):
        """Parse string for frame indices (e.g. '0', '0-10', '1,5,10', 'all')."""
        frame_str = str(frame_str).lower().strip()
        
        if frame_str == "all":
            return list(range(max_frames))
            
        indices = set()
        parts = frame_str.split(',')
        
        for part in parts:
            part = part.strip()
            if not part: continue
            
            if '-' in part:
                # Range: "start-end"
                try:
                    start, end = map(int, part.split('-'))
                    # Inclusive range
                    for i in range(start, end + 1):
                        if 0 <= i < max_frames:
                            indices.add(i)
                except ValueError:
                    FreeFlowUtils.log(f"Invalid frame range format: {part}", "WARN")
            else:
                # Single index
                try:
                    i = int(part)
                    if 0 <= i < max_frames:
                        indices.add(i)
                except ValueError:
                    FreeFlowUtils.log(f"Invalid frame index: {part}", "WARN")
                    
        sorted_indices = sorted(list(indices))
        if not sorted_indices:
            FreeFlowUtils.log("No valid frames selected. Defaulting to all.", "WARN")
            return list(range(max_frames))
            
        return sorted_indices

    def _extract_frame_number(self, image_path):
        """
        Extract the real frame number from an image filename.
        Examples: 
            'cam01_frame_0001.jpg' -> 1
            'frame_0042.png' -> 42
            '00005.exr' -> 5
        Returns the last number found in the filename (common convention).
        """
        import re
        filename = Path(image_path).stem
        nums = re.findall(r'\d+', filename)
        if nums:
            return int(nums[-1])  # Last number is typically the frame number
        return 0  # Fallback


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
            # Binary is required because _convert_ply_to_points3d expects binary
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
                
                # Pack data
                # x, y, z (double in struct above, but usually float in PLY? 
                # Points3D.bin has doubles. PLY usually floats for tools.
                # Let's write floats to match standard downstream usage.
                
                for v in vertices:
                    # v = (x, y, z, r, g, b)
                    # Pack: 3 floats (12 bytes) + 3 uchars (3 bytes) = 15 bytes per point?
                    # No padding in standard packed PLY usually.
                    f.write(struct.pack("<fffBBB", float(v[0]), float(v[1]), float(v[2]), v[3], v[4], v[5]))
            
            FreeFlowUtils.log(f"Exported {len(vertices)} sparse points to {output_ply.name} (Binary)")
            return output_ply
            
        except Exception as e:
            FreeFlowUtils.log(f"Failed to export sparse PLY: {e}", "WARN")
            return None

    def _get_anchor_filename_map(self, colmap_anchor, cameras):
        """
        Scans the anchor's images folder to find the expected filenames for each camera.
        Returns: {camera_name: filename} e.g. {'cam01': 'cam01_0000.jpg'}
        """
        anchor_images = Path(colmap_anchor) / "images"
        if not anchor_images.exists():
            return {}
            
        mapping = {}
        # List all files
        # We assume the anchor images contain the camera name as a substring or prefix
        files = [f.name for f in anchor_images.iterdir() if f.is_file()]
        
        for cam in cameras:
            # Heuristic 1: Exact prefix match {cam}_
            matches = [f for f in files if f.startswith(f"{cam}_")]
            if matches:
                 # Take the first one (usually 0000)
                 mapping[cam] = matches[0]
                 continue
            
            # Heuristic 2: Cam name is in the filename?
            matches = [f for f in files if cam in f]
            if matches:
                mapping[cam] = matches[0]
                continue
                
        return mapping

    def _safe_copy_or_link(self, src, dst, use_symlinks=True):
        """
        Creates a symlink if enabled and supported. 
        Falls back to copy if symlink fails or is disabled.
        """
        if use_symlinks:
            try:
                if dst.exists():
                     if dst.is_symlink() or dst.is_file():
                         dst.unlink()
                         
                os.symlink(src, dst)
                return True
            except OSError as e:
                FreeFlowUtils.log(f"Symlink failed (Permissions?): {e}. Falling back to copy.", "WARN")
        
        # Fallback to copy
        shutil.copy2(src, dst)
        return False

    def _prepare_brush_folder(self, frame_work_dir, multicam_feed, frame_idx, anchor_sparse, filename_map, use_symlinks=True, mask_engine=None, prev_images_paths=None, masking_method=None):
        """
        Prepare Brush-compatible folder structure:
        /images (images for this frame from all cameras)
        /sparse/0 (COLMAP data from anchor)
        
        CRITICAL: We MUST rename the new frame's images to match the filenames 
        in the anchor reconstruction (sparse/images.bin).
        """
        img_dir = frame_work_dir / "images"
        img_dir.mkdir(parents=True, exist_ok=True)
        
        cameras = list(multicam_feed.keys())
        active_cams = 0
        
        for cam in cameras:
            frames = multicam_feed[cam]

            if frame_idx < len(frames):
                src_img = Path(frames[frame_idx])
                
                # Determine destination name
                dst_name = filename_map.get(cam, f"{cam}{src_img.suffix}")
                dst_img = img_dir / dst_name

                # --- MASKING LOGIC ---
                mask_applied = False
                if mask_engine and prev_images_paths and prev_images_paths.get(cam):
                    prev_img = prev_images_paths[cam]
                    # Compute mask
                    mask_path = frame_work_dir / f"mask_{cam}.png"
                    generated_mask = mask_engine.compute_mask(src_img, prev_img, mask_path, method=masking_method)
                    
                    if generated_mask:
                        # Mask generated successfully.
                        # We used to apply it to Alpha, but now we use sidecar 'masks' folder.
                        # Proceed to move the mask file.

                                
                                # CORRECT APPROACH for 3DGS Video:
                                # "Motion Mask" usually masks OUT the DYNAMIC objects to train background, OR masks OUT background to train dynamic.
                                # User wants STABILITY.
                                # If we lock toplogy (Fixed Mode), we don't need masking for densification.
                                # If Dynamic Mode: Flicker comes from densifying background noise.
                                # So we want to PREVENT densification on background (Static).
                                # But we still want to render background!
                                
                                # If we pass Alpha=0 for background, it disappears.
                                # So we cannot use simple Alpha Channel masking for "Stabilization" if Brush treats Alpha as transparency.
                                # Unless Brush has a specific "--mask-path" arg?
                                # Checking help... no --mask-path.
                                
                                # ALTERNATIVE:
                                # Brush might support `mask` folder?
                                # Loading dataset from folder (Colmap structure).
                                # Colmap creates 'masks' folder.
                                # If we put masks in `masks/` parallel to `images/`...
                                # Brush/Colmap loader might pick them up.
                                # Let's try creating a `masks` folder!
                                # If `masks` folder exists, Colmap loaders usually use it.
                                # And usually Mask=1 (Train), Mask=0 (Ignore).
                                # So if we want to PROTECT background from densification... we can't just ignore it, or it will fade out?
                                # No, 3DGS "Ignore" means "Don't calculate loss here".
                                # If we don't calc loss, the gaussians covering it won't change (won't split, won't move).
                                # So existing background gaussians (from Frame 0) will remain, but won't be updated.
                                # THIS IS EXACTLY WHAT WE WANT!
                                # We want background to stay as is (Frame 0 state) and only update moving parts.
                                
                                # So, Mask = 1 (Motion) -> Update.
                                # Mask = 0 (Static) -> Ignore (Keep old state).
                                
                                mask_dir = frame_work_dir / "masks"
                                mask_dir.mkdir(exist_ok=True)
                                dst_mask = mask_dir / f"{dst_name}.png" # Must match image name usually
                                shutil.move(str(generated_mask), dst_mask)
                                
                                # We don't modify the image alpha, we provide sidecar mask.
                                # IMPORTANT: We still need to copy/link the image.
                                mask_applied = True # Mask file created
                            
                # Copy/Link Image
                if not use_symlinks or mask_applied: 
                    # If using sidecar masks, we can still symlink the image!
                    # Brush loader looks for masks/ folder.
                    # So we allow symlink for image even if mask exists.
                    self._safe_copy_or_link(src_img, dst_img, use_symlinks)
                else:
                    self._safe_copy_or_link(src_img, dst_img, use_symlinks)
                
                # Update Prev Path
                if prev_images_paths is not None:
                     prev_images_paths[cam] = src_img
                     
                active_cams += 1
        
        # Copy sparse reconstruction
        sparse_dest = frame_work_dir / "sparse"
        sparse_dest.mkdir(exist_ok=True)
        shutil.copytree(anchor_sparse, sparse_dest / "0", dirs_exist_ok=True)
        
        return active_cams

    def adapt_flow(self, multicam_feed, colmap_anchor, splat_count, sh_degree,
                   topology_mode="Dynamic (Default-Flicker)", apply_smoothing=False,
                   visualize_training="Off", preview_interval=500, eval_camera_index=10,
                   iterations=10000, learning_rate=0.0005, densification_interval=300,
                   densify_grad_threshold=0.0002, growth_select_fraction=0.1,
                   feature_lr=0.0025, gaussian_lr=0.0003, opacity_lr=0.01, scale_loss_weight=1e-8,
                   frame_selection="all", init_from_sparse=True, masking_method="Optical Flow (Robust)", motion_sensitivity=0.3,
                   custom_output_path="", filename_prefix="FreeFlow", use_symlinks=True, cleanup_work_dirs=True,
                   distributed_anchor=False, distributed_anchor_path="", distributed_anchor_frame="", warmup_frames=0, # Distributed params
                   unique_id=None, preview_camera_filter=""):
        """
        Execute adaptive training pipeline.
        Frame 0: Initialize from sparse cloud (if enabled)
        Frame 1+: Warm-start from previous frame's output
        """
        # 1. Setup Output Directory
        if custom_output_path and custom_output_path.strip():
            # Handle "~" expansion for Home directory
            expanded_path = os.path.expanduser(custom_output_path.strip())
            output_dir = Path(expanded_path)
        else:
            # Default to ComfyUI Output / FreeFlow_Timestamp
            output_dir = Path(folder_paths.get_output_directory()) / f"{filename_prefix}_{int(time.time())}"
            
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            FreeFlowUtils.log(f"✅ Output Directory set to: {output_dir}")
        except Exception as e:
            raise PermissionError(f"Could not create output directory {output_dir}: {e}")

        # -- NATIVE BRUSH BINARY CHECK (Auto-Install) --
        from .modules.brush_loader import ensure_brush_binary
        brush_bin = ensure_brush_binary()
        
        if not brush_bin:
             raise FileNotFoundError("Brush binary not found and auto-install failed. Please check internet connection or install manually.")

        # Initialize Components
        from .modules.motion_masking import MotionMasking
        mask_engine = MotionMasking(sensitivity=motion_sensitivity)
        
        # Get camera list and TOTAL frame count
        cameras = list(multicam_feed.keys())
        total_frames = max(len(multicam_feed[cam]) for cam in cameras)
        
        # PARSE FRAME SELECTION
        indices_to_process = self._parse_frames(frame_selection, total_frames)
        
        FreeFlowUtils.log("=" * 50)
        FreeFlowUtils.log("FreeFlow Adaptive Engine - Starting 4D Generation")
        FreeFlowUtils.log(f"Cameras: {len(cameras)} | Total Frames: {total_frames}")
        FreeFlowUtils.log(f"Selected Frames: {len(indices_to_process)} ({indices_to_process})")
        FreeFlowUtils.log("=" * 50)

        # COLMAP Anchor Path
        anchor_sparse = Path(colmap_anchor) / "sparse" / "0"
        if not anchor_sparse.exists():
            raise FileNotFoundError(f"COLMAP sparse reconstruction not found: {anchor_sparse}")

        # Export sparse cloud for initialization (if enabled)
        sparse_init_ply = None
        if init_from_sparse:
            sparse_init_ply = output_dir / "sparse_init.ply"
            sparse_init_ply = self._export_sparse_to_ply(anchor_sparse, sparse_init_ply)

        prev_ply = sparse_init_ply  # Use sparse init for first frame

        # Keep track of previous images for masking
        # NOTE: For masking to work well with skipped frames, we should arguably compare to 
        # the *actual* previous processed frame, OR the immediately preceding video frame.
        # "Motion Masking" usually implies Frame N vs Frame N-1 (Video Time).
        # If we skip frames (e.g. process 0, 5, 10), comparing 5 vs 0 is a huge jump.
        # Comparing 5 vs 4 (even if 4 isn't processed) is better for motion detection.
        # But we only have images for processed frames copied? No, we have input paths.
        
        prev_images_paths = {k: None for k in cameras} # Track paths of N-1 for masking

        # Generate Anchor Filename Map (Map Camera -> Expected Filename in images.bin)
        filename_map = self._get_anchor_filename_map(colmap_anchor, cameras)
        FreeFlowUtils.log(f"Anchor Filename Mapping: {filename_map}")

        # Progress Bar
        total_steps_global = len(indices_to_process) * iterations
        pbar = ProgressBar(total_steps_global) if ProgressBar else None
        current_step_global = 0

        # Build mapping: list_index -> real_frame_id for anchor priority
        first_cam = cameras[0]
        index_to_real_frame = {}
        for list_idx in indices_to_process:
            if list_idx < len(multicam_feed[first_cam]):
                real_id = self._extract_frame_number(multicam_feed[first_cam][list_idx])
            else:
                real_id = list_idx  # Fallback
            index_to_real_frame[list_idx] = real_id
        
        # Default: First frame of THIS batch (use its real frame ID)
        target_anchor_id = index_to_real_frame[indices_to_process[0]]
        
        if distributed_anchor and distributed_anchor_frame and distributed_anchor_frame.strip().isdigit():
            target_anchor_id = int(distributed_anchor_frame.strip())
            # Priority: Find which list_index corresponds to this real frame ID
            anchor_list_index = None
            for list_idx, real_id in index_to_real_frame.items():
                if real_id == target_anchor_id:
                    anchor_list_index = list_idx
                    break
            
            if anchor_list_index is not None and anchor_list_index in indices_to_process:
                indices_to_process.remove(anchor_list_index)
                indices_to_process.insert(0, anchor_list_index)
                print(f"   ⚓ Distributed Priority: Moved Real Frame {target_anchor_id} (index {anchor_list_index}) to start of queue.")
            else:
                print(f"   ⚠️ Warning: Anchor frame {target_anchor_id} not found in selected frames. Using default anchor.")
        
        # -------------------------
        
        # Pre-calculate topology mode (used in loop and after)
        is_fixed_topology = "Fixed" in topology_mode

        for idx, i in enumerate(indices_to_process):
            # Get REAL frame number from actual image filename (not list index!)
            first_cam = cameras[0]
            if i < len(multicam_feed[first_cam]):
                real_frame_id = self._extract_frame_number(multicam_feed[first_cam][i])
            else:
                real_frame_id = i  # Fallback to index if out of bounds
            
            frame_work_dir = output_dir / f"frame_{real_frame_id:04d}_work"
            ply_out = output_dir / f"{filename_prefix}_frame_{real_frame_id:04d}.ply"
            
            # Prepare folder structure
            active_cams = self._prepare_brush_folder(
                frame_work_dir, multicam_feed, i, anchor_sparse, filename_map, use_symlinks,
                mask_engine, prev_images_paths, masking_method 
            )
            
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
            
            # 2. Check Distributed Anchor (Only if NOT warm starting)
            elif distributed_anchor:
                 # Check Path OR Auto-Find anchor in folder
                 anchor_file = None
                 if distributed_anchor_path and Path(distributed_anchor_path).exists():
                     anchor_file = Path(distributed_anchor_path)
                 else:
                     # Auto-find: Look for any anchor_frame_*.ply in Distributed_Anchor folder
                     anchor_dir = output_dir / "Distributed_Anchor"
                     if anchor_dir.exists():
                         anchor_candidates = list(anchor_dir.glob("anchor_frame_*.ply"))
                         if anchor_candidates:
                             anchor_file = anchor_candidates[0]  # Use first found
                 
                 if anchor_file:
                     init_source_ply = anchor_file
                     init_mode = f"Distributed Anchor ({anchor_file.name})"
            
            # Perform Initialization Override
            if init_source_ply:
                converted_ok = self._convert_ply_to_points3d(init_source_ply, sparse_target / "points3D.txt")
                if converted_ok:
                    print(f"   🚀 Initializing Frame {i} from {init_mode}") # FreeFlowUtils.log not imported? use print
            pass # Spacer
            
            # Build Brush command (Only use supported flags)
            # Brush CLI: brush <PATH> --export-path --export-name --total-steps --lr-opac
            cmd = [
                str(brush_bin), 
                str(frame_work_dir),
                "--export-path", str(output_dir),
                "--export-name", ply_out.name,
                "--total-steps", str(iterations),
                # Prevent intermediate exports triggering early kill
                "--export-every", str(iterations), 
            ]
            
            # --- CORE MODEL PARAMS ---
            cmd.extend(["--max-splats", str(splat_count)])
            cmd.extend(["--sh-degree", str(sh_degree)])
            
            # --- LEARNING RATES ---
            # Position learning rate (most important for stability)
            cmd.extend(["--lr-mean", str(learning_rate)])
            
            # Feature/color learning rate (SH coefficients)
            cmd.extend(["--lr-coeffs-dc", str(feature_lr)])
            
            # Scale learning rate (gaussian size/shape)
            cmd.extend(["--lr-scale", str(gaussian_lr)])
            
            # Opacity learning rate
            cmd.extend(["--lr-opac", str(opacity_lr)])
            
            # --- REGULARIZATION ---
            # Scale loss weight (prevents overly large splats)
            cmd.extend(["--scale-loss-weight", str(scale_loss_weight)])
            
            # --- DENSIFICATION (Dynamic mode or Frame 0) ---
            is_first_frame = (idx == 0)
            
            if not is_fixed_topology or is_first_frame:
                # Densification interval (refine-every in Brush)
                cmd.extend(["--refine-every", str(densification_interval)])
                
                # Gradient threshold for growth
                cmd.extend(["--growth-grad-threshold", str(densify_grad_threshold)])
                
                # Fraction of points selected for growth
                cmd.extend(["--growth-select-fraction", str(growth_select_fraction)])
            
            # --- VISUALIZATION LOGIC ---
            if visualize_training == "Spawn Native GUI":
                cmd.append("--with-viewer")
                print(f"   👁️ Spawning Native Brush GUI...")
            elif visualize_training == "Save Preview Images":
                cmd.extend([
                    "--eval-every", str(preview_interval),
                    "--eval-save-to-disk"
                ])
                # Use user-defined eval camera index to control which cameras are rendered
                cmd.extend(["--eval-split-every", str(eval_camera_index)]) 
                print(f"   📸 Saving Preview Image every {preview_interval} steps (rendering camera index {eval_camera_index})...")

            # --- FIXED TOPOLOGY LOGIC (Frames after anchor) ---
            # If Stable Mode enabled AND not the first processed frame (anchor)
            if is_fixed_topology and not is_first_frame:
                # Force Freeze Topology: disable refinement and growth
                # VALID Brush CLI flags (verified from brush --help):
                #   --growth-stop-iter: "Period after which splat growth stops"
                #   --refine-every: "Frequency of refinement where gaussians are replaced and densified"
                
                cmd.extend([
                    "--growth-stop-iter", "0",     # Stop adding points immediately (no growth)
                    "--refine-every", "999999",    # Disable refinement (no split/clone/replace)
                ])
                print(f"   🔒 [Fixed Topology] Frame {i}: Topology Locked (Strict Frozen Mode)")
            
            # Execute training with Real-time Parsing
            frame_pct = ((idx + 1) / len(indices_to_process)) * 100
            FreeFlowUtils.log(f"🎬 Frame {idx+1}/{len(indices_to_process)} ({frame_pct:.1f}% global) - Training frame index {i}...")
            
            # Force unbuffered python output for the subprocess if possible
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1,
                env=env
            )
            
            # Simulated Progress Bar (Time-based estimate)
            # Use ADAPTIVE speed: cached from previous frame, or default for first frame
            
            if self._cached_seconds_per_step is not None:
                seconds_per_step = self._cached_seconds_per_step
                print(f"   ⏱️ Using calibrated speed: {seconds_per_step:.4f}s/step")
            else:
                # Default estimate (will be recalibrated)
                seconds_per_step = 0.035  # Conservative default
                print(f"   ⏱️ First frame - calibrating speed...")
            
            est_time = iterations * seconds_per_step
            
            # Pass reference to self so simulator can update cached speed
            # Also pass visualization mode for file-based progress
            frame_name = f"{filename_prefix}_frame_{real_frame_id:04d}"
            pbar_thread = threading.Thread(
                target=self._pbar_simulator, 
                args=(pbar, iterations, seconds_per_step, current_step_global, process, idx == 0, visualize_training, frame_name)
            )
            pbar_thread.daemon = True
            pbar_thread.start()

            # --- AUTO-COMPLETE MONITOR (For GUI Mode) ---
            def monitor_completion():
                while process.poll() is None:
                    if ply_out.exists():
                        time.sleep(5)
                        if process.poll() is None:
                             print("   ✅ Output found. Auto-Closing Brush GUI...")
                             process.terminate()
                             return
                    time.sleep(2)


            
            if visualize_training == "Spawn Native GUI":
                monitor_thread = threading.Thread(target=monitor_completion)
                monitor_thread.daemon = True
                monitor_thread.start()

            # --- PREVIEW MONITOR (Threaded) ---
            if visualize_training == "Save Preview Images" and unique_id:
                # Initialize step counter for this frame
                self._preview_step_counter[frame_name] = 0
                
                monitor_thread = threading.Thread(
                    target=self._monitor_previews_task,
                    args=(process, output_dir, unique_id, preview_camera_filter, frame_name)
                )
                monitor_thread.daemon = True
                monitor_thread.start()
            

            

            
            # Wait for process with Interruption Check
            import comfy.model_management
            try:
                while process.poll() is None:
                    if comfy.model_management.processing_interrupted():
                        process.terminate()
                        print("🚫 Processing Interrupted by User. Terminating Brush...")
                        comfy.model_management.throw_exception_if_processing_interrupted()
                    time.sleep(0.5)
                
                # Double check return code
                if process.returncode != 0:
                     # Ignore error IF: 
                     # 1. We Auto-Killed it (GUI Mode) AND Output exists
                     # 2. Return code is SIGTERM (-15) or 1 (some windows kills)
                     if visualize_training == "Spawn Native GUI" and ply_out.exists():
                         pass # Considered Success
                     else:
                        raise RuntimeError(f"Brush process failed with return code {process.returncode}")

            except KeyboardInterrupt:
                process.terminate()
                raise
            except Exception:
                if process.poll() is None:
                     process.terminate()
                raise

            current_step_global += iterations
            
            # Output check
            if not ply_out.exists():
                raise RuntimeError(f"Brush failed to generate output: {ply_out}")

            # --- DISTRIBUTED: Save Anchor ---
            # Save IF: Distributed Enabled AND Current Frame is the Target Anchor
            if distributed_anchor and real_frame_id == target_anchor_id and ply_out.exists():
                distributed_dir = output_dir / "Distributed_Anchor"
                distributed_dir.mkdir(exist_ok=True)
                # Use REAL frame ID for anchor filename (not list index)
                anchor_filename = f"anchor_frame_{real_frame_id:04d}.ply"
                anchor_dst = distributed_dir / anchor_filename
                shutil.copy(str(ply_out), str(anchor_dst))
                print(f"   ⚓ Saved Distributed Anchor to: {anchor_dst}")

            # --- WARMUP LOGIC ---
            # If warmup, move result to temp folder to avoid cluttering main output
            is_warmup = (idx < warmup_frames)
            # Force keep if it's the anchor (we usually want the anchor frame in the sequence)
            if distributed_anchor and real_frame_id == target_anchor_id:
                is_warmup = False
            
            if is_warmup:
                 warmup_dir = output_dir / "_warmup_temp"
                 warmup_dir.mkdir(exist_ok=True)
                 warmup_path = warmup_dir / ply_out.name
                 shutil.move(str(ply_out), str(warmup_path))
                 prev_ply = warmup_path
                 print(f"   🔥 Warmup: Moved Frame {real_frame_id} to temp storage.")
            else:
                 prev_ply = ply_out




                

            
            # Cleanup work directory to save space
            if cleanup_work_dirs:
                shutil.rmtree(frame_work_dir, ignore_errors=True)
                
        # --- POST-PROCESS: REALIGNMENT & SMOOTHING (Fixed Topology Only) ---
        realigned_dir = None
        smoothed_dir = None
        
        if is_fixed_topology and len(indices_to_process) > 3:
            FreeFlowUtils.log("🔗 Running Point Realignment...")
            try:
                realigned_dir = self._apply_temporal_smoothing(
                    output_dir, filename_prefix, indices_to_process, 
                    multicam_feed, cameras, do_smoothing=False
                )
                FreeFlowUtils.log(f"✅ Realignment Complete! Saved to: {realigned_dir}")
                
                if apply_smoothing:
                    FreeFlowUtils.log("🌊 Running Temporal Smoothing...")
                    smoothed_dir = self._apply_temporal_smoothing(
                        output_dir, filename_prefix, indices_to_process,
                        multicam_feed, cameras, do_smoothing=True
                    )
                    FreeFlowUtils.log(f"✅ Smoothing Complete! Saved to: {smoothed_dir}")
                    
            except Exception as e:
                FreeFlowUtils.log(f"Post-processing failed: {e}", "WARN")
        elif is_fixed_topology and len(indices_to_process) <= 3:
            FreeFlowUtils.log("⚠️ Post-processing skipped: Sequence too short (<4 frames).", "WARN")
        elif apply_smoothing and not is_fixed_topology:
            FreeFlowUtils.log("⚠️ Smoothing skipped: Requires 'Fixed (Stable)' topology.", "WARN")
                


        FreeFlowUtils.log("=" * 50)
        FreeFlowUtils.log(f"✅ Training Complete!")
        FreeFlowUtils.log(f"   📁 Raw output: {output_dir}")
        if realigned_dir:
            FreeFlowUtils.log(f"   📁 Realigned: {realigned_dir}")
        if smoothed_dir:
            FreeFlowUtils.log(f"   📁 Smoothed: {smoothed_dir}")
        FreeFlowUtils.log("=" * 50)

        # Return both paths (realigned and smoothed)
        return (str(output_dir), str(realigned_dir) if realigned_dir else "", str(smoothed_dir) if smoothed_dir else "")

    def _apply_temporal_smoothing(self, output_dir, prefix, indices, multicam_feed, cameras, do_smoothing=True):
        """
        Applies topology realignment and optionally Savitzky-Golay filtering.
        ONLY works for Fixed Topology mode where point count is constant.
        
        SAVES to separate subfolders to preserve original trained files:
        - realigned/ : Topology realigned only (no smoothing)
        - smoothed/  : Realigned + Savitzky-Golay filtered
        
        Always performs KD-Tree based point ID realignment first to fix
        point order shuffling from Brush. Realignment is essential for both
        realigned and smoothed outputs.
        
        Args:
            do_smoothing: If True, apply Savitzky-Golay filter after realignment
            
        Returns:
            Path to output folder (realigned/ or smoothed/)
        """
        if not indices: return None
        
        # Determine output folder based on mode
        if do_smoothing:
            out_dir = output_dir / "smoothed"
            mode_name = "smoothed"
        else:
            out_dir = output_dir / "realigned"
            mode_name = "realigned"
        
        out_dir.mkdir(exist_ok=True)
        FreeFlowUtils.log(f"   📁 {mode_name.capitalize()} output will be saved to: {out_dir}")
        
        # Extract REAL frame IDs from source image filenames
        first_cam = cameras[0]
        real_frame_ids = []
        for i in indices:
            if i < len(multicam_feed[first_cam]):
                real_id = self._extract_frame_number(multicam_feed[first_cam][i])
            else:
                real_id = i  # Fallback
            real_frame_ids.append(real_id)
        
        # Source PLY files (originals - read only) - use REAL frame IDs
        source_files = [output_dir / f"{prefix}_frame_{fid:04d}.ply" for fid in real_frame_ids]
        # Destination PLY files (output folder) - use REAL frame IDs
        output_files = [out_dir / f"{prefix}_frame_{fid:04d}.ply" for fid in real_frame_ids]
        
        # Check first file to get vertex count and header size
        first_ply = source_files[0]
        if not first_ply.exists(): return
        
        # Helper to get header size and offsets
        def get_ply_meta(path):
            with open(path, "rb") as f:
                header = b""
                while True:
                    line = f.readline()
                    header += line
                    if line.strip() == b"end_header": break
                
                header_str = header.decode("utf-8", errors="ignore")
                m = re.search(r"element vertex (\d+)", header_str)
                count = int(m.group(1))
                
                # Find xyz offsets
                # Simplified assumption: standard FreeFlow output puts x,y,z first as floats
                # "property float x"
                # "property float y"
                # "property float z"
                # ...
                # Let's verify standard header format
                return len(header), count
        
        header_len, vertex_count = get_ply_meta(first_ply)
        
        # 2. Extract XYZ
        # We assume standard packed float/uchar layout from Brush
        # XYZ are usually the first 3 floats (12 bytes)
        # Stride = 12 + colors + etc.
        # Brush export usually: x,y,z, nx,ny,nz, f_dc..., f_rest..., opacity, scale, rot
        # Wait, Brush output varies. 
        # Safest is to read EVERYTHING into a big buffer, modify XYZ, write back.
        
        # For simplicity and safety given unknown complete stride:
        # We won't try to parse everything. The user just wants smoothing.
        # If we can't guarantee stride, we might corrupt data.
        
        # Alternative: Just skip smoothing if too complex?
        # User requested it.
        # Let's try to assume XYZ are at offset 0, 4, 8 (standard) and type float32.
        
        # Load all data
        all_data = [] # List of bytearrays (mutable)
        
        for p in source_files:
            if not p.exists():
                all_data.append(None)
                continue
            with open(p, "rb") as f:
                data = bytearray(f.read()) # Read everything
                all_data.append(data)
        
        # Check consistency
        valid_data = [d for d in all_data if d is not None]
        if not valid_data: return
        
        # 3. Smooth
        # Extract XYZ for all frames
        # We assume header length matches (it should if fixed topology)
        # But actually header length might vary slightly if commented? 
        # Brush output is deterministic.
        
        # Create Numpy array (Time, Points, 3)
        # We rely on "element vertex N" being constant.
        
        # Find offset of data start
        data_start = header_len 
        
        # Stride calculation?
        # File size - header len / vertex count
        total_len = len(valid_data[0])
        stride = (total_len - header_len) // vertex_count
        
        # Ensure divisible
        if (total_len - header_len) % vertex_count != 0:
            FreeFlowUtils.log("Warning: PLY data size mismatch. Skipping smoothing.", "WARN")
            return
            
        # extract
        # Make a view? 
        # data[start:].view(dtype=np.float32) ? 
        # Mixed types (float, uint8) makes simple view hard.
        # But we only need to touch the first 3 floats of every stride.
        
        # We will iterate and build the array. 
        # (Using a structured array might be faster but tricky with unknown schema)
        
        # Let's try to update IN PLACE using memory mapping logic via bytearray
        # Gather all X, Y, Z
        
        T = len(valid_data)
        if T < 5: return # Need window
        
        points = np.zeros((T, vertex_count, 3), dtype=np.float32)
        
        for t, data in enumerate(valid_data):
            # Hacky stride access
            # We assume position is at offset 0 of each vertex
            # Create a strided view
            # offset 0, stride=stride
            
            # View raw bytes as flat uint8
            raw_view = np.frombuffer(data, dtype=np.uint8, offset=header_len)
            
            # Reshape into (N, stride)
            reshaped = raw_view.reshape(vertex_count, stride)
            
            # Take first 12 bytes (3 floats)
            xyz_bytes = reshaped[:, :12]
            
            # View as floats
            xyz_floats = xyz_bytes.view(dtype=np.float32).reshape(vertex_count, 3)
            
            points[t] = xyz_floats
        
        # 3.5. TOPOLOGY REALIGNMENT (KD-Tree based) - ALWAYS PERFORM
        # Fixes point ID shuffling from Brush by matching points incrementally
        FreeFlowUtils.log("   🔧 Realigning topology using KD-Tree matching...")
        reorder_maps = self._compute_topology_realignment(points)
        
        # Apply reorder to both points array and raw data
        for t in range(T):
            indices_map = reorder_maps[t]
            
            # Reorder points array
            points[t] = points[t][indices_map]
            
            # Reorder the full PLY binary data (all properties, not just XYZ)
            all_data[t] = self._reorder_ply_binary(all_data[t], indices_map, header_len, stride, vertex_count)
        
        FreeFlowUtils.log(f"   ✅ Realigned {T} frames")
        
        # 4. Apply Savitzky-Golay (only if do_smoothing is True)
        if do_smoothing:
            # Smooth across Time (axis 0)
            window_length = min(7, T if T % 2 == 1 else T-1)
            if window_length < 3: 
                window_length = 3
            
            points_to_write = scipy.signal.savgol_filter(points, window_length=window_length, polyorder=2, axis=0)
        else:
            points_to_write = points
        
        # 5. Write back
        for t, data in enumerate(valid_data):
            raw_view = np.frombuffer(data, dtype=np.uint8, offset=header_len)
            reshaped = raw_view.reshape(vertex_count, stride)
            
            # Get target slice
            target_bytes = reshaped[:, :12]
            
            # Use positions (smoothed or realigned)
            new_xyz = points_to_write[t].astype(np.float32)
            
            # Cast new_xyz to uint8 view of shape (N, 12)
            new_uint8 = new_xyz.view(np.uint8).reshape(vertex_count, 12)
            
            np.copyto(target_bytes, new_uint8)
            
            # Save file to output folder (preserve originals!)
            output_path = output_files[t]
            if output_path:
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
        from scipy.spatial import cKDTree
        
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
                     if not line: # EOF safety
                         break
                 
                 header_str = header.decode("utf-8", errors="ignore")
                 
                 # 1. Parse Vertex Count
                 m = re.search(r"element vertex (\d+)", header_str)
                 if not m: return False
                 vertex_count = int(m.group(1))
                 
                 # 2. Parse Properties to calculate Stride and Offsets
                 properties = [] # List of (name, type, size)
                 
                 x_offset = -1; y_offset = -1; z_offset = -1
                 r_offset = -1; g_offset = -1; b_offset = -1
                 
                 r_type = 'f'; g_type = 'f'; b_type = 'f' # Default to float
                 
                 current_offset = 0
                 
                 for line in header_str.split('\n'):
                     if line.startswith("property"):
                         parts = line.split()
                         if len(parts) < 3: continue
                         ptype = parts[1]
                         pname = parts[2]
                         
                         size = 4 # default float
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
                 
                 # Safety limit for huge files? 
                 # Usually partial updates aren't massive.
                 
                 batch_limit = 50000 
                 step = max(1, vertex_count // batch_limit)
                 
                 for i in range(0, vertex_count, step):
                     base = i * stride
                     if base + stride > len(body): break
                     
                     # Extract data
                     # Must use offset because properties can be in any order
                     
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
                                 if abs(rf) > 1.0 or abs(rf) < -1.0: # Likely not 0-1, maybe SH?
                                      # Actually SH can be small. 
                                      # Assuming SH if float
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
        # User requested: TrainingPreviews/{prefix}_frame{Number}/
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
                            
                        # Logic:
                        # 1. Identify Step (from parent folder name eval_500)
                        # 2. Identify Camera (filename)
                        parent_name = cand.parent.name # eval_500
                        
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



