"""
FreeFlow GS Engine - Splatfacto Backend
Nerfstudio Splatfacto - Production-quality Gaussian Splatting trainer.

Uses subprocess isolation to avoid ComfyUI dependency conflicts.
The actual nerfstudio runs in an isolated venv at ~/.freeflow/nerfstudio_venv/
"""

import os
import re
import subprocess
import threading
import shutil
import time
import json
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple

from .base_engine import IGSEngine


class SplatfactoEngine(IGSEngine):
    """
    Implementation of the Nerfstudio Splatfacto Backend.
    Production-quality Gaussian Splatting with rich features.
    
    Features:
    - Full nerfstudio splatfacto model variants
    - Fixed topology support (no pruning after densification)
    - Warm start via checkpoint loading
    - Progress parsing from subprocess output
    - PLY export via ns-export
    
    Variants:
    - splatfacto: Default balanced model
    - splatfacto-big: Higher capacity, more splats
    - splatfacto-w: With appearance embedding (for varying lighting)
    """
    
    def __init__(self):
        """Initialize SplatfactoEngine and check environment."""
        self._env = None
        self._init_error = None
        self.last_checkpoint_dir = None  # Stores checkpoint dir from last successful training
        
        try:
            from ..nerfstudio_env import NerfstudioEnvironment
            self._env = NerfstudioEnvironment
        except ImportError as e:
            self._init_error = f"Failed to import NerfstudioEnvironment: {e}"
    
    @property
    def env(self):
        """Get the NerfstudioEnvironment class."""
        return self._env
    
    def is_available(self) -> bool:
        """Check if Nerfstudio is installed and available."""
        if self._env is None:
            return False
        return self._env.is_installed()
    
    def get_name(self) -> str:
        return "Splatfacto (Pro)"
    
    def get_version(self) -> Optional[str]:
        """Get installed nerfstudio version."""
        if not self._env:
            return None
        return self._env.get_version()
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information."""
        if not self._env:
            return {"error": self._init_error}
        return self._env.get_status_info()
    
    def install(self, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """
        Install nerfstudio in the isolated venv.
        
        Args:
            progress_callback: Optional callback(message, progress) for UI feedback
            
        Returns:
            True if installation succeeded.
        """
        if not self._env:
            return False
        return self._env.create_venv(progress_callback)

    def train(self, 
              dataset_path: Path, 
              output_path: Path, 
              params: Dict[str, Any], 
              prev_ply_path: Optional[Path] = None,
              mask_path: Optional[Path] = None, 
              callback_data: Optional[Dict] = None) -> bool:
        """
        Execute Splatfacto training for one frame.
        
        Args:
            dataset_path: Path to COLMAP-format dataset (with images/ and sparse/0/)
            output_path: Path to save the resulting PLY file
            params: Training parameters dict with keys:
                - iterations: int (default 30000)
                - sh_degree: int (default 3)
                - splatfacto_variant: str (splatfacto, splatfacto-big, splatfacto-w)
                - cull_alpha_thresh: float (default 0.005)
                - densify_grad_thresh: float (default 0.0002)
                - use_scale_regularization: bool (default True)
                - continue_cull_post_densification: bool (default True, False for fixed topology)
                - output_dir: str (where nerfstudio saves its outputs)
                - visualize_training: str ("Off", "Save Preview Images", "Spawn Native GUI")
                - preview_interval: int (steps between eval images, default 500)
                - preview_callback: callable (function(image_path, step) for preview updates)
                - previews_dir: Path (where to save organized preview images)
            prev_ply_path: Path to previous checkpoint for warm start (not PLY, but nerfstudio checkpoint dir)
            mask_path: Not used currently (reserved for future mask support)
            callback_data: Dict with 'pbar_func' for progress callback
            
        Returns:
            bool: True if training succeeded and PLY was exported.
        """
        if not self.is_available():
            print(f"[SplatfactoEngine] Error: Nerfstudio not available. {self._init_error}")
            return False
        
        ns_train = self._env.get_ns_train()
        if not ns_train:
            print("[SplatfactoEngine] Error: ns-train not found")
            return False
        
        # Prepare output directory
        output_path = Path(output_path)
        nerfstudio_output_dir = params.get('output_dir', output_path.parent / "nerfstudio_outputs")
        nerfstudio_output_dir = Path(nerfstudio_output_dir)
        nerfstudio_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build ns-train command
        variant = params.get('splatfacto_variant', 'splatfacto')
        
        # Helper to format booleans for nerfstudio CLI (requires 'True'/'False' not 'true'/'false')
        def bool_str(val):
            return 'True' if val else 'False'
        
        # Detect device type: CUDA > MPS > CPU
        # NOTE: Splatfacto requires CUDA due to gsplat dependency
        # MPS (Apple Silicon) is NOT supported by gsplat - it will fall back to CPU
        import platform
        import subprocess as sp
        device_type = "cpu"  # fallback - Note: splatfacto on CPU is VERY slow
        venv_python = self._env.get_python()
        if venv_python:
            try:
                # Check for CUDA first - this is the only fully supported option for splatfacto
                check_cuda = sp.run(
                    [str(venv_python), "-c", 
                     "import torch; print('cuda' if torch.cuda.is_available() else 'no')"],
                    capture_output=True, text=True, timeout=30
                )
                if check_cuda.returncode == 0 and "cuda" in check_cuda.stdout.lower():
                    device_type = "cuda"
                # NOTE: MPS is NOT supported by gsplat/splatfacto
                # The model has hardcoded .cuda() calls that fail on MPS
                # We fall back to CPU which is very slow but should work
            except Exception as e:
                print(f"[SplatfactoEngine] Device detection error: {e}, falling back to CPU")
        
        if device_type != "cuda":
            print(f"[SplatfactoEngine] WARNING: CUDA not available. Splatfacto requires CUDA for GPU acceleration.")
            print(f"[SplatfactoEngine] Training will use CPU which is VERY slow. Consider using CUDA-enabled system.")
        
        print(f"[SplatfactoEngine] Using device type: {device_type}")
        
        # --- VISUALIZATION MODE ---
        visualize_mode = params.get('visualize_training', 'Off')
        preview_interval = params.get('preview_interval', 500)
        
        cmd = [
            str(ns_train),
            variant,
            "--data", str(dataset_path),
            "--output-dir", str(nerfstudio_output_dir),
            "--max-num-iterations", str(params.get('iterations', 30000)),
            "--machine.device-type", device_type,
            
            # Model parameters
            f"--pipeline.model.sh_degree={params.get('sh_degree', 3)}",
            f"--pipeline.model.cull_alpha_thresh={params.get('cull_alpha_thresh', 0.005)}",
            f"--pipeline.model.densify_grad_thresh={params.get('densify_grad_thresh', 0.0002)}",
            f"--pipeline.model.use_scale_regularization={bool_str(params.get('use_scale_regularization', True))}",
        ]
        
        # --- FIXED TOPOLOGY MODE ---
        # Disable culling after densification phase to preserve splat count
        if not params.get('continue_cull_post_densification', True):
            cmd.append("--pipeline.model.continue_cull_post_densification=False")
        
        # --- WARM START ---
        # Load from previous nerfstudio checkpoint directory if available
        # This enables true warm start: transfers learned colors, opacities, covariances
        # (not just positions like the points3D.txt seeding approach)
        nerfstudio_checkpoint_dir = params.get('nerfstudio_checkpoint_dir')
        if nerfstudio_checkpoint_dir and Path(nerfstudio_checkpoint_dir).is_dir():
            cmd.extend(["--load-dir", str(nerfstudio_checkpoint_dir)])
            print(f"[SplatfactoEngine] Warm start from checkpoint: {nerfstudio_checkpoint_dir}")
        elif prev_ply_path and Path(prev_ply_path).exists():
            # PLY file can't be loaded directly as checkpoint - points3D.txt seeding
            # is handled by the orchestrator before calling train()
            print(f"[SplatfactoEngine] Using points3D.txt seeding for initialization")
        
        # --- VIEWER / VISUALIZATION SETTINGS ---
        if visualize_mode == "Spawn Native GUI":
            # Enable Viser web viewer and auto-open browser
            cmd.extend(["--vis", "viewer"])
            viewer_port = params.get('viewer_port', 7007)
            cmd.append(f"--viewer.websocket-port={viewer_port}")
            print(f"[SplatfactoEngine] Viser viewer enabled at http://localhost:{viewer_port}")
            # Auto-open browser after a short delay (give viser time to start)
            def open_viewer_browser(port):
                time.sleep(5)  # Wait for viser server to start
                import webbrowser
                webbrowser.open(f"http://localhost:{port}")
                print(f"[SplatfactoEngine] Opened browser to http://localhost:{port}")
            browser_thread = threading.Thread(target=open_viewer_browser, args=(viewer_port,), daemon=True)
            browser_thread.start()
        elif visualize_mode == "Save Preview Images":
            # Use tensorboard for logging + enable eval image saving
            cmd.extend(["--vis", "tensorboard"])
            # Set eval interval to save preview images periodically
            cmd.extend(["--steps-per-eval-image", str(preview_interval)])
            print(f"[SplatfactoEngine] Preview images will be saved every {preview_interval} steps")
        else:
            # Off - no interactive viewer, minimal logging
            # tensorboard mode doesn't spawn a server, just logs to disk
            cmd.extend(["--vis", "tensorboard"])
        
        # --- LOGGING ---
        # Set experiment name for organized outputs
        experiment_name = params.get('experiment_name', 'freeflow_frame')
        cmd.extend(["--experiment-name", experiment_name])
        
        # --- DATAPARSER ---
        # Add colmap dataparser for COLMAP-format datasets
        # This must come AFTER all model options
        # Dataparser-specific options come AFTER the dataparser subcommand
        cmd.append("colmap")
        
        # Override default colmap path (colmap/sparse/0) to match FreeFlow's output structure (sparse/0)
        # This is critical: nerfstudio expects colmap/sparse/0/ but FreeFlow_COLMAP outputs sparse/0/
        cmd.extend(["--colmap-path", "sparse/0"])
        
        # Disable automatic downscaling to prevent interactive prompts in headless mode
        # User can pre-downscale images if needed for performance
        cmd.extend(["--downscale-factor", "1"])
        
        print(f"[SplatfactoEngine] Running: {' '.join(cmd)}")
        
        # Execute training
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Fix Windows console encoding issues with Rich library
        # Without this, nerfstudio crashes on CONSOLE.rule() due to cp1252 encoding
        env["PYTHONIOENCODING"] = "utf-8"
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=env,
                encoding='utf-8',  # Fix Windows cp1252 decoding errors
                errors='replace'   # Replace undecodable chars instead of crashing
            )
            
            # Progress parsing in background thread
            progress_info = {
                'current': 0, 
                'total': params.get('iterations', 30000),
                'training_complete': False
            }
            
            def parse_progress():
                """Parse nerfstudio output for progress updates."""
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    # Nerfstudio outputs progress like: "Step 1000/30000"
                    match = re.search(r'Step\s+(\d+)/(\d+)', line)
                    if match:
                        progress_info['current'] = int(match.group(1))
                        progress_info['total'] = int(match.group(2))
                    
                    # Also check for loss values
                    loss_match = re.search(r'loss:\s*([\d.e+-]+)', line, re.IGNORECASE)
                    if loss_match:
                        progress_info['loss'] = float(loss_match.group(1))
                    
                    # Detect training completion
                    # Nerfstudio prints "Training Finished" when done but the Viser
                    # viewer keeps the process alive waiting for ctrl+c
                    if "Training Finished" in line:
                        progress_info['training_complete'] = True
                    
                    # Print output for debugging
                    print(f"[ns-train] {line.rstrip()}")
            
            # Start progress thread
            progress_thread = threading.Thread(target=parse_progress, daemon=True)
            progress_thread.start()
            
            # Progress callback - pass process to pbar_func (same pattern as Brush)
            # pbar_func(process) runs _pbar_simulator which monitors process.poll()
            if callback_data and callback_data.get('pbar_func'):
                pbar_thread = threading.Thread(
                    target=callback_data['pbar_func'], args=(process,), daemon=True
                )
                pbar_thread.start()
            
            # --- PREVIEW MONITORING THREAD ---
            # Monitor for eval images and copy them to TrainingPreviews folder
            preview_callback = params.get('preview_callback')
            previews_dir = params.get('previews_dir')
            
            if visualize_mode == "Save Preview Images" and (preview_callback or previews_dir):
                def monitor_nerfstudio_previews():
                    """
                    Monitor nerfstudio output for eval images.
                    
                    Nerfstudio saves eval images to:
                    output_dir/experiment_name/splatfacto/TIMESTAMP/
                    
                    Image types we look for:
                    - imgs/ folder (tensorboard eval images)
                    - *.png in the run directory
                    """
                    seen_images = set()
                    
                    while process.poll() is None:
                        try:
                            # Search for PNG files in nerfstudio output
                            # Structure: output_dir/experiment_name/*/TIMESTAMP/
                            for img_path in nerfstudio_output_dir.rglob("*.png"):
                                if img_path in seen_images:
                                    continue
                                
                                # Skip if file is still being written (check stability)
                                try:
                                    mtime = img_path.stat().st_mtime
                                    if time.time() - mtime < 1.0:
                                        continue
                                except:
                                    continue
                                
                                seen_images.add(img_path)
                                
                                # Extract step number from filename or path
                                # Nerfstudio uses names like: step-000500.png, eval_000500.png
                                step_match = re.search(r'(\d{4,})', img_path.stem)
                                step = int(step_match.group(1)) if step_match else 0
                                
                                # Copy to organized previews folder if specified
                                if previews_dir:
                                    previews_path = Path(previews_dir)
                                    previews_path.mkdir(parents=True, exist_ok=True)
                                    
                                    # Create consistent naming: eval_{step}_{original_name}.png
                                    new_name = f"eval_{step:05d}_{img_path.name}"
                                    dest = previews_path / new_name
                                    
                                    if not dest.exists():
                                        try:
                                            shutil.copy2(img_path, dest)
                                            print(f"[SplatfactoEngine] Saved preview: {new_name}")
                                        except Exception as e:
                                            print(f"[SplatfactoEngine] Failed to copy preview: {e}")
                                
                                # Call the preview callback if provided
                                if preview_callback:
                                    try:
                                        preview_callback(img_path, step)
                                    except Exception as e:
                                        print(f"[SplatfactoEngine] Preview callback error: {e}")
                        
                        except Exception as e:
                            print(f"[SplatfactoEngine] Preview monitor error: {e}")
                        
                        time.sleep(2)  # Check every 2 seconds
                
                preview_thread = threading.Thread(target=monitor_nerfstudio_previews, daemon=True)
                preview_thread.start()
                print(f"[SplatfactoEngine] Preview monitoring started for {nerfstudio_output_dir}")
            
            # Wait for completion with interrupt checking
            # Nerfstudio's Viser viewer keeps the process alive after training finishes.
            # We detect "Training Finished" in the output and terminate after a grace period
            # to ensure checkpoints are fully saved to disk.
            import comfy.model_management
            try:
                while process.poll() is None:
                    # Check for training completion (Viser keeps process alive)
                    if progress_info.get('training_complete'):
                        time.sleep(3)  # Grace period for checkpoint save
                        if process.poll() is None:
                            print("[SplatfactoEngine] Training complete, terminating process...")
                            process.terminate()
                            try:
                                process.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                process.kill()
                            break
                    
                    if comfy.model_management.processing_interrupted():
                        process.terminate()
                        print("🚫 Processing Interrupted by User. Terminating Splatfacto...")
                        comfy.model_management.throw_exception_if_processing_interrupted()
                    time.sleep(0.5)
            except KeyboardInterrupt:
                process.terminate()
                raise
            except Exception:
                if process.poll() is None:
                    process.terminate()
                raise
            
            progress_thread.join(timeout=5)
            
            # Find the config.yml for export
            # Check for config BEFORE checking return code, because on Windows
            # nerfstudio often crashes during cleanup with heap corruption (0xC0000374)
            # even though training completed successfully
            config_path = self._find_config(nerfstudio_output_dir, experiment_name)
            
            if process.returncode != 0:
                print(f"[SplatfactoEngine] ns-train exited with return code {process.returncode}")
                if config_path:
                    print(f"[SplatfactoEngine] Training output found despite exit code - continuing to export")
                else:
                    print(f"[SplatfactoEngine] Training failed - no output found")
                    return False
            
            if not config_path:
                print("[SplatfactoEngine] Error: Could not find config.yml after training")
                return False
            
            # Export PLY
            export_success = self._export_ply(config_path, output_path)
            if not export_success:
                print("[SplatfactoEngine] Error: PLY export failed")
                return False
            
            # Store checkpoint dir for warm start on next frame
            checkpoint_dir = self.get_checkpoint_path(nerfstudio_output_dir, experiment_name)
            if checkpoint_dir:
                self.last_checkpoint_dir = checkpoint_dir
                print(f"[SplatfactoEngine] Checkpoint saved for warm start: {checkpoint_dir}")
            
            print(f"[SplatfactoEngine] Success! PLY exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"[SplatfactoEngine] Execution error: {e}")
            return False
    
    def _find_config(self, output_dir: Path, experiment_name: str) -> Optional[Path]:
        """
        Find the config.yml file from nerfstudio training output.
        
        Nerfstudio saves to: output_dir/experiment_name/splatfacto/TIMESTAMP/config.yml
        """
        # Search pattern
        search_paths = [
            output_dir / experiment_name,
            output_dir,
        ]
        
        for base_path in search_paths:
            if not base_path.exists():
                continue
            
            # Find most recent config.yml
            configs = list(base_path.rglob("config.yml"))
            if configs:
                # Sort by modification time, newest first
                configs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                return configs[0]
        
        return None
    
    def _export_ply(self, config_path: Path, output_path: Path) -> bool:
        """
        Export trained model to PLY format using ns-export.
        
        Args:
            config_path: Path to config.yml from training
            output_path: Desired output PLY path
            
        Returns:
            True if export succeeded.
        """
        ns_export = self._env.get_ns_export()
        if not ns_export:
            print("[SplatfactoEngine] Error: ns-export not found")
            return False
        
        output_path = Path(output_path)
        export_dir = output_path.parent / "ns_export_temp"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            str(ns_export),
            "gaussian-splat",
            "--load-config", str(config_path),
            "--output-dir", str(export_dir),
        ]
        
        print(f"[SplatfactoEngine] Exporting PLY: {' '.join(cmd)}")
        
        # Fix Windows console encoding issues with Rich library
        export_env = os.environ.copy()
        export_env["PYTHONIOENCODING"] = "utf-8"
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minutes
                env=export_env
            )
            
            if result.returncode != 0:
                print(f"[SplatfactoEngine] Export failed: {result.stderr}")
                return False
            
            # Find the exported PLY file
            exported_plys = list(export_dir.glob("*.ply"))
            if not exported_plys:
                # Check subdirectories
                exported_plys = list(export_dir.rglob("*.ply"))
            
            if not exported_plys:
                print("[SplatfactoEngine] Error: No PLY file found after export")
                return False
            
            # Move/rename to desired output path
            source_ply = exported_plys[0]
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(source_ply), str(output_path))
            
            # Cleanup temp export dir
            try:
                shutil.rmtree(export_dir)
            except Exception:
                pass
            
            return output_path.exists()
            
        except subprocess.TimeoutExpired:
            print("[SplatfactoEngine] Export timed out")
            return False
        except Exception as e:
            print(f"[SplatfactoEngine] Export error: {e}")
            return False
    
    def get_checkpoint_path(self, output_dir: Path, experiment_name: str) -> Optional[Path]:
        """
        Get the checkpoint directory for warm start.
        
        Args:
            output_dir: Nerfstudio output directory
            experiment_name: Experiment name used during training
            
        Returns:
            Path to checkpoint directory, or None if not found.
        """
        config_path = self._find_config(output_dir, experiment_name)
        if not config_path:
            return None
        
        # Checkpoint is in nerfstudio_models/ sibling to config.yml
        checkpoint_dir = config_path.parent / "nerfstudio_models"
        if checkpoint_dir.exists():
            return checkpoint_dir
        
        return None
