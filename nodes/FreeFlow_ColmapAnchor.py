"""
FreeFlow_ColmapAnchor - Power User Edition
Full CLI exposure for COLMAP with granular parameter control.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import threading
from pathlib import Path
from ..utils import FreeFlowUtils


class FreeFlow_ColmapAnchor:
    """
    Master-class wrapper around COLMAP CLI.
    Runs Feature Extraction, Matching, and Mapping with granular user control.
    """
    
    # Quality presets mapping to SIFT parameters
    QUALITY_PRESETS = {
        "Low": {
            "SiftExtraction.num_octaves": 3,
            "SiftExtraction.peak_threshold": 0.02,
            "SiftExtraction.max_num_features": 4096,
        },
        "Medium": {
            "SiftExtraction.num_octaves": 4,
            "SiftExtraction.peak_threshold": 0.0066,
            "SiftExtraction.max_num_features": 8192,
        },
        "High": {
            "SiftExtraction.num_octaves": 4,
            "SiftExtraction.peak_threshold": 0.004,
            "SiftExtraction.max_num_features": 16384,
        },
        "Extreme": {
            "SiftExtraction.num_octaves": 5,
            "SiftExtraction.peak_threshold": 0.002,
            "SiftExtraction.max_num_features": 32768,
        },
    }
    
    def __init__(self):
        self.output_dir = Path(tempfile.gettempdir()) / "FreeFlow_Colmap"
        self._last_error = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multicam_feed": ("MULTICAM_DICT",),
                # --- Quality Preset ---
                "quality": (["Low", "Medium", "High", "Extreme"], {"default": "High", "tooltip": "Reconstruction quality. Higher = more points but slower."}),
                # --- Camera Model ---
                "camera_model": (["SIMPLE_PINHOLE", "PINHOLE", "OPENCV", "SIMPLE_RADIAL"], {"default": "OPENCV", "tooltip": "Camera lens model. 'OPENCV' handles distortion best for real footage. 'PINHOLE' for synthetic."}),
            },
            "optional": {
                # --- Frame Selection ---
                # Flexible string input: "0" (default), "0-10", "1,5,10", "all"
                "frame_selection": ("STRING", {"default": "0", "multiline": False, "tooltip": "Frames to use for anchor. Usually '0' (first frame) is enough. 'all' for full SfM (slow)."}),
                 # --- Persistent Storage ---
                "override_existing": ("BOOLEAN", {"default": False, "label": "Override Existing COLMAP", "tooltip": "If True, deletes previous reconstruction and runs again."}),
                
                # --- SIFT Extraction Options ---
                "sift_estimate_affine_shape": ("BOOLEAN", {"default": False, "tooltip": "Enable for wide-baseline matching (big camera angles)."}),
                "sift_domain_size_pooling": ("BOOLEAN", {"default": False, "tooltip": "Improves feature detection stability."}),
                "sift_use_gpu": ("BOOLEAN", {"default": FreeFlowUtils.get_os() == "Windows", "tooltip": "Use GPU for SIFT. Faster but requires NVIDIA GPU."}),
                # --- Matching Options ---
                "matching_method": (["Exhaustive", "Sequential", "VocabTree", "Spatial"], {"default": "Exhaustive", "tooltip": "Exhaustive = Best. Sequential = Video. VocabTree = Fast (Auto-Downloads 150MB file). Spatial = GPS (Requires Metadata)."}),
                "sift_guided_matching": ("BOOLEAN", {"default": False, "tooltip": "Refine matches using geometry. Slow but accurate."}),
                # --- Mapper Settings ---
                "mapper_ba_tolerance": ("FLOAT", {"default": 1e-5, "min": 1e-8, "max": 0.1, "step": 1e-6, "tooltip": "Bundle Adjustment tolerance. Lower (1e-6) = tighter alignment."}),
                "mapper_min_num_matches": ("INT", {"default": 15, "min": 2, "max": 100, "tooltip": "Min matches to register an image. Fail if < 15."}),
                "mapper_fix_existing_images": ("BOOLEAN", {"default": False, "tooltip": "Lock valid cameras when refining."}),
            }
        }

    RETURN_TYPES = ("COLMAP_DATA",)
    RETURN_NAMES = ("colmap_anchor",)
    FUNCTION = "run_colmap"
    CATEGORY = "FreeFlow"

    def _stream_output(self, process, prefix=""):
        """Stream subprocess output to console in real-time."""
        def reader(pipe, name):
            for line in iter(pipe.readline, ''):
                if line:
                    print(f"🌊 [COLMAP {prefix}] {line.strip()}")
            pipe.close()
        
        stdout_thread = threading.Thread(target=reader, args=(process.stdout, "OUT"))
        stderr_thread = threading.Thread(target=reader, args=(process.stderr, "ERR"))
        
        stdout_thread.daemon = True
        stderr_thread.daemon = True
        
        stdout_thread.start()
        stderr_thread.start()
        
        return stdout_thread, stderr_thread

    def _run_colmap_command(self, cmd, cwd, step_name):
        """Run a COLMAP command with real-time output streaming."""
        FreeFlowUtils.log(f"Running {step_name}...")
        FreeFlowUtils.log(f"Command: {' '.join(cmd)}")
        
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(cwd),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Stream output in real-time
            stdout_thread, stderr_thread = self._stream_output(process, step_name)
            
            # Wait for completion
            return_code = process.wait()
            
            # Wait for output threads to finish
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            
            if return_code != 0:
                self._last_error = f"{step_name} failed with code {return_code}"
                raise RuntimeError(self._last_error)
            
            FreeFlowUtils.log(f"{step_name} completed successfully.")
            return True
            
        except Exception as e:
            self._last_error = str(e)
            raise

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
            FreeFlowUtils.log("No valid frames selected. Defaulting to Frame 0.", "WARN")
            return [0]
            
        return sorted_indices

    def _build_extraction_cmd(self, colmap_bin, database_path, images_dir, params):
        """Build feature extraction command with all user parameters."""
        quality = params.get("quality", "Medium")
        preset = self.QUALITY_PRESETS.get(quality, self.QUALITY_PRESETS["Medium"])
        
        cmd = [
            str(colmap_bin), "feature_extractor",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", params.get("camera_model", "PINHOLE"),
        ]
        
        # Apply quality preset
        for key, value in preset.items():
            cmd.extend([f"--{key}", str(value)])
        
        # SIFT options
        if params.get("sift_estimate_affine_shape", False):
            cmd.extend(["--SiftExtraction.estimate_affine_shape", "1"])
        
        if params.get("sift_domain_size_pooling", False):
            cmd.extend(["--SiftExtraction.domain_size_pooling", "1"])
        
        # GPU usage - Conditional Check
        use_gpu = params.get("sift_use_gpu", True)
        
        # Mac Homebrew/Standard binaries often lack CUDA support AND the flag itself.
        # Passing --SiftExtraction.use_gpu=0 throws error "unrecognised option" on some builds.
        # So we ONLY pass this flag if we are on Windows or explicitly requested by user (via override?)
        # For now, safer to only pass it on Windows.
        if FreeFlowUtils.get_os() == "Windows":
             cmd.extend(["--SiftExtraction.use_gpu", "1" if use_gpu else "0"])
        
        return cmd

    def _build_matching_cmd(self, colmap_bin, database_path, params):
        """Build feature matching command with all user parameters."""
        method = params.get("matching_method", "Exhaustive")
        
        method_map = {
            "Exhaustive": "exhaustive_matcher",
            "Sequential": "sequential_matcher",
            "VocabTree": "vocab_tree_matcher",
            "Spatial": "spatial_matcher",
        }
        
        matcher = method_map.get(method, "exhaustive_matcher")
        
        cmd = [
            str(colmap_bin), matcher,
            "--database_path", str(database_path),
        ]
        
        # Guided matching (slow but high accuracy)
        if params.get("sift_guided_matching", False):
            cmd.extend(["--SiftMatching.guided_matching", "1"])
            
        # Sequential Matching Options
        if method == "Sequential":
            # Default overlap to 10 for safety (generic video)
            cmd.extend(["--SequentialMatching.overlap", "10"])
            # Loop closure? Maybe too advanced for now. Use Exhaustive for full closure.

        # VocabTree Options
        if method == "VocabTree":
             vocab_path = FreeFlowUtils.get_vocab_tree_path()
             if not vocab_path:
                 # Attempt download just in case check failed upstream
                 if FreeFlowUtils.install_vocab_tree():
                     vocab_path = FreeFlowUtils.get_vocab_tree_path()
                 else:
                     raise RuntimeError("Failed to download Vocabulary Tree file.")
             cmd.extend(["--VocabTreeMatching.vocab_tree_path", str(vocab_path)])

    def _build_mapper_cmd(self, colmap_bin, database_path, images_dir, sparse_dir, params):
        """Build mapper command with advanced settings."""
        cmd = [
            str(colmap_bin), "mapper",
            "--database_path", str(database_path),
            "--image_path", str(images_dir),
            "--output_path", str(sparse_dir),
        ]
        
        # Bundle adjustment tolerance
        tolerance = params.get("mapper_ba_tolerance", 0.0001)
        cmd.extend(["--Mapper.ba_global_function_tolerance", str(tolerance)])
        
        # Minimum matches
        min_matches = params.get("mapper_min_num_matches", 15)
        cmd.extend(["--Mapper.min_num_matches", str(min_matches)])
        
        # Fix existing images (for incremental updates)
        if params.get("mapper_fix_existing_images", False):
            cmd.extend(["--Mapper.fix_existing_images", "1"])
        
        return cmd

    def run_colmap(self, multicam_feed, quality="High", camera_model="OPENCV",
                   frame_selection="0", override_existing=False,
                   sift_estimate_affine_shape=False, sift_domain_size_pooling=False,
                   sift_use_gpu=True, matching_method="Exhaustive", sift_guided_matching=False,
                   mapper_ba_tolerance=1e-5, mapper_min_num_matches=15,
                   mapper_fix_existing_images=False):
        """
        Execute complete COLMAP pipeline: Feature Extraction → Matching → Mapping
        """
        # Collect all parameters
        params = {
            "quality": quality,
            "camera_model": camera_model,
            "sift_estimate_affine_shape": sift_estimate_affine_shape,
            "sift_domain_size_pooling": sift_domain_size_pooling,
            "sift_use_gpu": sift_use_gpu,
            "matching_method": matching_method,
            "sift_guided_matching": sift_guided_matching,
            "mapper_ba_tolerance": mapper_ba_tolerance,
            "mapper_min_num_matches": mapper_min_num_matches,
            "mapper_fix_existing_images": mapper_fix_existing_images,
        }
        
        FreeFlowUtils.log("=" * 50)
        FreeFlowUtils.log("COLMAP Power User Edition - Starting Pipeline")
        FreeFlowUtils.log(f"Quality: {quality} | Frames: {frame_selection}")
        FreeFlowUtils.log("=" * 50)
        
        # 1. Determine Output Directory (Parent of input)
        cameras = list(multicam_feed.keys())
        if not cameras: 
             raise ValueError("Multicam feed is empty!")
             
        # Get path of first image from first camera to find parent
        first_cam_files = multicam_feed[cameras[0]]
        if not first_cam_files:
            raise ValueError(f"Camera {cameras[0]} has no images!")
            
        first_image_path = Path(first_cam_files[0]) # e.g. .../Capture/cam01/0000.jpg
        
        # Logic: 
        # If structure is .../Capture/cam01/img.jpg and key is cam01 -> Parent is .../Capture
        parent_dir = first_image_path.parent
        if parent_dir.name == cameras[0]:
             parent_dir = parent_dir.parent
             
        FreeFlowUtils.log(f"Detected Project Parent Directory: {parent_dir}")
        
        # Persistent Output Workspace
        workspace = parent_dir / "FreeFlow_Colmap"
        sparse_dir = workspace / "sparse"
        reconstruction = sparse_dir / "0"
        points_file = reconstruction / "points3D.bin"
        
        # 2. Check for existing result
        if workspace.exists() and reconstruction.exists() and points_file.exists():
            if not override_existing:
                FreeFlowUtils.log("✅ Found existing COLMAP reconstruction. Skipping processing.")
                FreeFlowUtils.log(f"   Path: {reconstruction}")
                return (str(workspace),)
            else:
                 FreeFlowUtils.log("⚠️ Existing reconstruction found, but Override is ON. Overwriting...", "WARN")
                 try:
                     shutil.rmtree(workspace)
                 except Exception as e:
                     FreeFlowUtils.log(f"Failed to delete existing workspace: {e}", "ERROR")

        # 3. Create Workspace
        workspace.mkdir(parents=True, exist_ok=True)
        images_dir = workspace / "images"
        if images_dir.exists():
             shutil.rmtree(images_dir)
        images_dir.mkdir()

        # Copy selected frames from each camera
        FreeFlowUtils.log(f"DEBUG: Processing Multicam Feed with {len(cameras)} cameras.")
        
        image_count = 0
        
        FreeFlowUtils.log("Preparing Images...")
        
        for cam in cameras:
            frames = multicam_feed[cam]
            if not frames:
                continue
            
            # Parse frame selection for THIS camera (assuming sync)
            indices_to_process = self._parse_frames(frame_selection, len(frames))
            
            for idx in indices_to_process:
                src = Path(frames[idx])
                # Naming convention: camName_frameIndex.ext to ensure uniqueness
                dst_name = f"{cam}_{idx:04d}{src.suffix}"
                dst = images_dir / dst_name
                shutil.copy2(src, dst)
                image_count += 1
        
        # Verify physical files
        files_in_dir = list(images_dir.iterdir())
        FreeFlowUtils.log(f"DEBUG: Physical files in {images_dir}: {len(files_in_dir)}")
        if len(files_in_dir) > 0:
            FreeFlowUtils.log(f"DEBUG: Sample file: {files_in_dir[0].name}")

        if image_count == 0:
            raise ValueError("No images selected for processing!")
        
        FreeFlowUtils.log(f"Prepared {image_count} images.")

        # 4. Find COLMAP binary
        colmap_bin = FreeFlowUtils.get_binary_path("colmap")
        if not colmap_bin:
            error_msg = "COLMAP binary not found! "
            if FreeFlowUtils.get_os() == "Darwin":
                error_msg += "Run: brew install colmap"
            elif FreeFlowUtils.get_os() == "Linux":
                error_msg += "Run: sudo apt install colmap"
            else:
                error_msg += "Download from colmap.github.io and place in bin/"
            raise FileNotFoundError(error_msg)

        database_path = workspace / "database.db"
        sparse_dir.mkdir(exist_ok=True) # Ensure sparse dir exists

        # 5. Feature Extraction
        cmd_extract = self._build_extraction_cmd(colmap_bin, database_path, images_dir, params)
        self._run_colmap_command(cmd_extract, workspace, "Feature Extractor")

        # 6. Matching
        cmd_match = self._build_matching_cmd(colmap_bin, database_path, params)
        self._run_colmap_command(cmd_match, workspace, f"{matching_method} Matcher")

        # 7. Mapping
        cmd_mapper = self._build_mapper_cmd(colmap_bin, database_path, images_dir, sparse_dir, params)
        self._run_colmap_command(cmd_mapper, workspace, "Mapper")

        # 8. Verify reconstruction
        if not reconstruction.exists():
            raise RuntimeError("COLMAP failed to create a reconstruction (sparse/0 not found).")

        # Count results
        points_file = reconstruction / "points3D.bin"
        
        FreeFlowUtils.log("=" * 50)
        FreeFlowUtils.log("✅ COLMAP Pipeline Completed Successfully!")
        FreeFlowUtils.log(f"   Reconstruction: {reconstruction}")
        if points_file.exists():
            FreeFlowUtils.log(f"   Points3D: {points_file.stat().st_size / 1024:.1f} KB")
        FreeFlowUtils.log("=" * 50)

        return (str(workspace),)
