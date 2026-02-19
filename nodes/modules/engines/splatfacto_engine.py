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
        self._browser_opened = False  # Track if browser was already opened (only open once per run)
        self._lpips_cache_attempted = False
        self._max_gs_num_supported = None  # None = not yet checked, True/False = cached result
        
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

    @staticmethod
    def _is_truthy_env(value: Optional[str]) -> bool:
        if not value:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _detect_default_ca_bundle() -> Optional[str]:
        candidates = [
            "/etc/pki/tls/certs/ca-bundle.crt",
            "/etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem",
            "/etc/ssl/certs/ca-certificates.crt",
            "/etc/ssl/cert.pem",
        ]
        for candidate in candidates:
            if Path(candidate).exists():
                return candidate
        return None

    @staticmethod
    def _get_project_root() -> Path:
        # .../custom_nodes/ComfyUi-FreeFlowGS/nodes/modules/engines/splatfacto_engine.py
        #                                 ^ parents[3] = project root
        try:
            return Path(__file__).resolve().parents[3]
        except Exception:
            return Path(__file__).resolve().parent

    def _build_nerfstudio_env(self, include_unbuffered: bool = False) -> Dict[str, str]:
        run_env = os.environ.copy()

        if include_unbuffered:
            run_env["PYTHONUNBUFFERED"] = "1"

        # Fix console encoding + PyTorch 2.6 checkpoint loading behavior.
        run_env["PYTHONIOENCODING"] = "utf-8"
        run_env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

        # Keep torch cache local to FreeFlow unless user provided TORCH_HOME.
        torch_home = run_env.get("TORCH_HOME")
        if not torch_home:
            torch_home = str(self._get_project_root() / ".torch_cache")
            run_env["TORCH_HOME"] = torch_home

        try:
            checkpoints_dir = Path(torch_home).expanduser() / "hub" / "checkpoints"
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Respect explicit insecure mode only when user opts in.
        if self._is_truthy_env(run_env.get("FREEFLOW_SSL_NO_VERIFY")):
            run_env.setdefault("PYTHONHTTPSVERIFY", "0")
            return run_env

        # Populate CA bundle env vars for stdlib/requests/curl consumers.
        ca_bundle = (
            run_env.get("SSL_CERT_FILE")
            or run_env.get("REQUESTS_CA_BUNDLE")
            or run_env.get("CURL_CA_BUNDLE")
            or self._detect_default_ca_bundle()
        )
        if ca_bundle:
            run_env.setdefault("SSL_CERT_FILE", ca_bundle)
            run_env.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)
            run_env.setdefault("CURL_CA_BUNDLE", ca_bundle)

        return run_env

    def _supports_max_gs_num(self) -> bool:
        """Check if the installed splatfacto model config supports max_gs_num."""
        if self._max_gs_num_supported is not None:
            return self._max_gs_num_supported

        if not self._env:
            self._max_gs_num_supported = False
            return False

        venv_python = self._env.get_python()
        if not venv_python:
            self._max_gs_num_supported = False
            return False

        try:
            import subprocess as sp
            result = sp.run(
                [str(venv_python), "-c",
                 "from nerfstudio.models.splatfacto import SplatfactoModelConfig; "
                 "print(hasattr(SplatfactoModelConfig(), 'max_gs_num'))"],
                capture_output=True, text=True, timeout=30
            )
            supported = result.returncode == 0 and "True" in result.stdout
            self._max_gs_num_supported = supported
            if supported:
                print("[SplatfactoEngine] max_gs_num: supported by installed nerfstudio")
            else:
                print("[SplatfactoEngine] max_gs_num: not supported by installed nerfstudio (skipping)")
        except Exception as e:
            print(f"[SplatfactoEngine] max_gs_num capability check failed ({e}), skipping")
            self._max_gs_num_supported = False

        return self._max_gs_num_supported

    def _ensure_lpips_alexnet_cached(self, run_env: Dict[str, str]) -> None:
        """Best-effort one-time pre-cache for LPIPS AlexNet weights."""
        if self._lpips_cache_attempted:
            return
        self._lpips_cache_attempted = True

        torch_home = run_env.get("TORCH_HOME")
        if not torch_home or not self._env:
            return

        checkpoints_dir = Path(torch_home).expanduser() / "hub" / "checkpoints"
        try:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        if list(checkpoints_dir.glob("alexnet-*.pth")):
            return

        venv_python = self._env.get_python()
        if not venv_python:
            return

        print("[SplatfactoEngine] Pre-caching LPIPS AlexNet weights (one-time)...")
        cmd = [
            str(venv_python),
            "-c",
            (
                "from torchvision.models import alexnet, AlexNet_Weights; "
                "alexnet(weights=AlexNet_Weights.DEFAULT); "
                "print('alexnet weights cached')"
            ),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=180,
                env=run_env,
            )
            if result.returncode == 0:
                cached = list(checkpoints_dir.glob("alexnet-*.pth"))
                if cached:
                    print(f"[SplatfactoEngine] LPIPS cache ready: {cached[0].name}")
                else:
                    print("[SplatfactoEngine] LPIPS pre-cache completed")
            else:
                detail = (result.stderr or result.stdout or "").strip()
                if detail:
                    detail = detail.splitlines()[-1]
                print(f"[SplatfactoEngine] LPIPS pre-cache skipped: {detail or 'unknown error'}")
        except Exception as e:
            print(f"[SplatfactoEngine] LPIPS pre-cache skipped: {e}")
    
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
        
        iterations = params.get('iterations', 30000)
        
        # Check for masks first (needed for proper command ordering)
        masks_dir = dataset_path / "masks"
        has_masks = masks_dir.exists() and any(masks_dir.iterdir())
        if has_masks:
            print(f"[SplatfactoEngine] Masks detected: {masks_dir}")
        
        # Build command in CORRECT ORDER for nerfstudio CLI:
        # 1. Base command (ns-train variant)
        # 2. Model parameters (--pipeline.model.*)
        # 3. Training parameters (--load-dir, --vis, etc.)
        # 4. Data paths (--data, --output-dir)
        # 5. Dataparser subcommand (colmap) and its args
        # 6. Other flags (--downscale-factor)
        
        experiment_name = params.get('experiment_name', 'freeflow_frame')
        viewer_port = params.get('viewer_port', 7007)
        
        # Determine if we're doing cross-frame warm start (checkpoint loading)
        nerfstudio_checkpoint_dir = params.get('nerfstudio_checkpoint_dir')
        use_checkpoint_warmstart = (nerfstudio_checkpoint_dir and 
                                     Path(nerfstudio_checkpoint_dir).is_dir())
        
        # Always use wrapper for PyTorch 2.6 save_checkpoint fix.
        # When warm starting, wrapper also seeds points3D.txt from checkpoint
        # (so Gaussian count matches) and loads model weights post-setup.
        wrapper_script = Path(__file__).parent / "ns_train_wrapper.py"
        cmd = [
            str(venv_python),
            str(wrapper_script),
        ]
        if use_checkpoint_warmstart:
            cmd.extend(["--ff-load-weights", str(nerfstudio_checkpoint_dir)])
            print(f"[SplatfactoEngine] Warm start: seeding from checkpoint {nerfstudio_checkpoint_dir}")
        cmd.append(variant)
        
        # --- MODEL PARAMETERS (must come before data paths) ---
        cmd.extend([
            f"--pipeline.model.sh_degree={params.get('sh_degree', 3)}",
            f"--pipeline.model.cull_alpha_thresh={params.get('cull_alpha_thresh', 0.22)}",
            f"--pipeline.model.densify_grad_thresh={params.get('densify_grad_thresh', 0.0012)}",
            f"--pipeline.model.use_scale_regularization={bool_str(params.get('use_scale_regularization', True))}",
            f"--pipeline.model.refine_every={params.get('refine_every', 120)}",
            f"--pipeline.model.warmup_length={params.get('warmup_length', 800)}",
            f"--pipeline.model.num_downscales={params.get('num_downscales', 1)}",
            f"--pipeline.model.cull_screen_size={params.get('cull_screen_size', 0.15)}",
            f"--pipeline.model.split_screen_size={params.get('split_screen_size', 0.05)}",
            f"--pipeline.model.sh_degree_interval={params.get('sh_degree_interval', 1000)}",
            f"--pipeline.model.background_color={params.get('background_color', 'black')}",
        ])
        
        # --- MAX GAUSSIAN CAP (version-dependent) ---
        # max_gs_num was added in newer nerfstudio versions (MCMC support).
        # Only append when the installed version supports it.
        max_gs_num = params.get('max_gs_num')
        if max_gs_num is not None and self._supports_max_gs_num():
            cmd.append(f"--pipeline.model.max_gs_num={int(max_gs_num)}")
            print(f"[SplatfactoEngine] Gaussian cap: max_gs_num={int(max_gs_num)}")
        
        # --- FIXED TOPOLOGY MODE ---
        is_fixed_topology = not params.get('continue_cull_post_densification', True)
        if is_fixed_topology:
            cmd.extend([
                "--pipeline.model.stop_split_at=0",
                "--pipeline.model.refine_every=999999",
            ])
            print(f"[SplatfactoEngine] Fixed topology: splitting and refinement disabled")
        
        # --- VISUALIZATION (only if not Off) ---
        visualize_mode = params.get('visualize_training', 'Off')
        if visualize_mode != "Off":
            cmd.extend([
                "--vis", "viewer",
                f"--viewer.websocket-port={viewer_port}",
                "--viewer.quit-on-train-completion=True",
            ])
            print(f"[SplatfactoEngine] Viser viewer at http://localhost:{viewer_port}")
            
            # Auto-open browser ONCE on first frame only
            if not self._browser_opened:
                self._browser_opened = True
                def open_viewer_browser(port):
                    time.sleep(5)
                    import webbrowser
                    webbrowser.open(f"http://localhost:{port}")
                    print(f"[SplatfactoEngine] Opened browser to http://localhost:{port}")
                browser_thread = threading.Thread(target=open_viewer_browser, args=(viewer_port,), daemon=True)
                browser_thread.start()
        
        # --- LOGGING ---
        cmd.extend(["--experiment-name", experiment_name])
        
        # --- TRAINING PARAMETERS ---
        cmd.extend([
            "--max-num-iterations", str(iterations),
            "--machine.device-type", device_type,
            "--steps-per-eval-all-images", "0",  # Disable full-dataset eval (extremely slow)
            "--steps-per-eval-image", "0",        # Disable per-image eval renders
            "--steps-per-eval-batch", "0",         # Disable batch eval
        ])
        
        # --- DATA PATHS ---
        cmd.extend([
            "--data", str(dataset_path),
            "--output-dir", str(nerfstudio_output_dir),
        ])
        
        # --- DATAPARSER (must come AFTER all training parameters) ---
        cmd.append("colmap")
        cmd.extend(["--colmap-path", "sparse/0"])
        if has_masks:
            cmd.extend(["--masks-path", "masks"])
        
        # --- OTHER FLAGS ---
        cmd.extend(["--downscale-factor", "1"])
        
        print(f"[SplatfactoEngine] Running: {' '.join(cmd)}")
        
        # Execute training
        env = self._build_nerfstudio_env(include_unbuffered=True)
        self._ensure_lpips_alexnet_cached(env)
        
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
                        time.sleep(5)  # Grace period for checkpoint save
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
            
            # Store checkpoint dir for warm start on next frame
            checkpoint_dir = self.get_checkpoint_path(nerfstudio_output_dir, experiment_name)
            if checkpoint_dir:
                self.last_checkpoint_dir = checkpoint_dir
                print(f"[SplatfactoEngine] Checkpoint saved for warm start: {checkpoint_dir}")

            # Export PLY
            # In Fixed Topology mode, use checkpoint-direct export first to preserve
            # exact Gaussian count (ns-export filters low-opacity/invalid gaussians).
            export_success = False
            if is_fixed_topology and checkpoint_dir:
                print("[SplatfactoEngine] Fixed topology export: using checkpoint-direct path (constant count)")
                export_success = self._export_ply_from_checkpoint(checkpoint_dir, output_path)
                if not export_success:
                    print("[SplatfactoEngine] Fixed export failed; falling back to ns-export")

            if not export_success:
                export_success = self._export_ply(config_path, output_path)

            if not export_success and checkpoint_dir and not is_fixed_topology:
                print("[SplatfactoEngine] ns-export failed; trying checkpoint-direct export fallback")
                export_success = self._export_ply_from_checkpoint(checkpoint_dir, output_path)

            if not export_success:
                print("[SplatfactoEngine] Error: PLY export failed")
                return False
            
            print(f"[SplatfactoEngine] Success! PLY exported to {output_path}")
            return True
            
        except Exception as e:
            print(f"[SplatfactoEngine] Execution error: {e}")
            return False
    
    def _find_config(self, output_dir: Path, experiment_name: str) -> Optional[Path]:
        """
        Find the config.yml file from nerfstudio training output.
        
        Nerfstudio saves to: output_dir/experiment_name/splatfacto/TIMESTAMP/config.yml
        
        IMPORTANT: Only search within the specific experiment directory to avoid
        finding the wrong config when multiple frames are trained.
        """
        # Only search within the specific experiment directory
        experiment_dir = output_dir / experiment_name
        if not experiment_dir.exists():
            return None
        
        # Find config.yml within this specific experiment only
        configs = list(experiment_dir.rglob("config.yml"))
        if configs:
            # Sort by modification time, newest first (in case multiple timestamps exist)
            configs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return configs[0]
        
        return None

    def _export_ply_from_checkpoint(self, checkpoint_dir: Path, output_path: Path) -> bool:
        """
        Export PLY directly from latest checkpoint tensors.

        This path preserves exact Gaussian count (no opacity/NaN filtering),
        which is required for Fixed Topology smoothing workflows.
        """
        try:
            import numpy as np
            import torch

            checkpoint_dir = Path(checkpoint_dir)
            output_path = Path(output_path)

            ckpt_files = sorted(checkpoint_dir.glob("step-*.ckpt"))
            if not ckpt_files:
                print(f"[SplatfactoEngine] Direct export failed: no checkpoint in {checkpoint_dir}")
                return False

            ckpt_path = ckpt_files[-1]
            print(f"[SplatfactoEngine] Direct export from checkpoint: {ckpt_path.name}")

            loaded = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            pipeline_state = loaded.get("pipeline", {})

            def find_tensor(*suffixes):
                for key, value in pipeline_state.items():
                    clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
                    if any(clean_key.endswith(suffix) for suffix in suffixes):
                        return value
                return None

            means_t = find_tensor("_model.gauss_params.means", "model.gauss_params.means", "_model.means", "model.means")
            scales_t = find_tensor("_model.gauss_params.scales", "model.gauss_params.scales", "_model.scales", "model.scales")
            quats_t = find_tensor("_model.gauss_params.quats", "model.gauss_params.quats", "_model.quats", "model.quats")
            opacities_t = find_tensor("_model.gauss_params.opacities", "model.gauss_params.opacities", "_model.opacities", "model.opacities")
            features_dc_t = find_tensor("_model.gauss_params.features_dc", "model.gauss_params.features_dc", "_model.features_dc", "model.features_dc")
            features_rest_t = find_tensor("_model.gauss_params.features_rest", "model.gauss_params.features_rest", "_model.features_rest", "model.features_rest")

            required = {
                "means": means_t,
                "scales": scales_t,
                "quats": quats_t,
                "opacities": opacities_t,
                "features_dc": features_dc_t,
                "features_rest": features_rest_t,
            }
            missing = [name for name, tensor in required.items() if tensor is None]
            if missing:
                print(f"[SplatfactoEngine] Direct export failed: missing tensors {missing}")
                return False

            means = means_t.detach().cpu().float().numpy()
            scales = scales_t.detach().cpu().float().numpy()
            quats = quats_t.detach().cpu().float().numpy()
            opacities = opacities_t.detach().cpu().float().numpy().reshape(-1)
            features_dc = features_dc_t.detach().cpu().float().numpy()
            features_rest = features_rest_t.detach().cpu().float().numpy()

            if features_dc.ndim == 3 and features_dc.shape[1] == 1:
                features_dc = features_dc.squeeze(1)
            if features_dc.ndim != 2:
                print(f"[SplatfactoEngine] Direct export failed: unexpected features_dc shape {features_dc.shape}")
                return False

            if features_rest.ndim == 3:
                # Match nerfstudio exporter ordering: transpose(1,2).reshape(N,-1)
                if features_rest.shape[2] == 3:
                    features_rest_flat = np.transpose(features_rest, (0, 2, 1)).reshape(features_rest.shape[0], -1)
                elif features_rest.shape[1] == 3:
                    features_rest_flat = features_rest.reshape(features_rest.shape[0], -1)
                else:
                    print(f"[SplatfactoEngine] Direct export failed: unexpected features_rest shape {features_rest.shape}")
                    return False
            elif features_rest.ndim == 2:
                features_rest_flat = features_rest
            else:
                print(f"[SplatfactoEngine] Direct export failed: unexpected features_rest rank {features_rest.ndim}")
                return False

            n = int(means.shape[0])
            if n == 0:
                print("[SplatfactoEngine] Direct export failed: zero gaussians")
                return False

            if scales.shape[0] != n or quats.shape[0] != n or opacities.shape[0] != n:
                print(
                    "[SplatfactoEngine] Direct export failed: tensor count mismatch "
                    f"means={n}, scales={scales.shape[0]}, quats={quats.shape[0]}, opacities={opacities.shape[0]}"
                )
                return False

            def sanitize_1d(arr):
                arr = np.asarray(arr, dtype=np.float32).reshape(n)
                return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

            fields = []
            fields.append(("x", sanitize_1d(means[:, 0])))
            fields.append(("y", sanitize_1d(means[:, 1])))
            fields.append(("z", sanitize_1d(means[:, 2])))
            fields.append(("nx", np.zeros(n, dtype=np.float32)))
            fields.append(("ny", np.zeros(n, dtype=np.float32)))
            fields.append(("nz", np.zeros(n, dtype=np.float32)))

            for i in range(features_dc.shape[1]):
                fields.append((f"f_dc_{i}", sanitize_1d(features_dc[:, i])))

            for i in range(features_rest_flat.shape[1]):
                fields.append((f"f_rest_{i}", sanitize_1d(features_rest_flat[:, i])))

            fields.append(("opacity", sanitize_1d(opacities)))
            for i in range(min(3, scales.shape[1])):
                fields.append((f"scale_{i}", sanitize_1d(scales[:, i])))
            for i in range(min(4, quats.shape[1])):
                fields.append((f"rot_{i}", sanitize_1d(quats[:, i])))

            dtype = [(name, "<f4") for name, _ in fields]
            vertices = np.empty(n, dtype=dtype)
            for name, values in fields:
                vertices[name] = values

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(b"ply\n")
                f.write(b"format binary_little_endian 1.0\n")
                f.write(b"comment Generated by FreeFlow checkpoint-direct export\n")
                f.write(b"comment Vertical Axis: z\n")
                f.write(f"element vertex {n}\n".encode("ascii"))
                for name, _ in fields:
                    f.write(f"property float {name}\n".encode("ascii"))
                f.write(b"end_header\n")
                vertices.tofile(f)

            print(f"[SplatfactoEngine] Direct export wrote {n} gaussians (unfiltered): {output_path}")
            return output_path.exists()
        except Exception as e:
            print(f"[SplatfactoEngine] Direct checkpoint export error: {e}")
            return False

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
        
        export_env = self._build_nerfstudio_env(include_unbuffered=False)
        self._ensure_lpips_alexnet_cached(export_env)
        
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
