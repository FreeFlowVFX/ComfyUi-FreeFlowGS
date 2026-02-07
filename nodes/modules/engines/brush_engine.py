"""
FreeFlow GS Engine - Brush Backend
ArthurBrussee/brush - Fast, lightweight, single binary Gaussian Splatting trainer.
"""

import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

from .base_engine import IGSEngine


class BrushEngine(IGSEngine):
    """
    Implementation of the Brush Backend (ArthurBrussee/brush).
    Fast, Lightweight, Single Binary.
    
    Features:
    - Auto-downloads binary from GitHub releases
    - Supports Fixed Topology mode via CLI flags
    - Progress callback support
    - Visualization modes: Native GUI, Save Preview Images
    """
    
    def __init__(self):
        """Initialize BrushEngine and ensure binary is available."""
        self._brush_bin = None
        self._init_error = None
        
        try:
            from ..brush_loader import ensure_brush_binary
            self._brush_bin = ensure_brush_binary()
        except Exception as e:
            self._init_error = str(e)
    
    @property
    def brush_bin(self) -> Optional[Path]:
        """Get path to Brush binary."""
        return self._brush_bin
    
    def is_available(self) -> bool:
        """Check if Brush binary is available."""
        return self._brush_bin is not None and self._brush_bin.exists()
    
    def get_name(self) -> str:
        return "Brush (Fast)"
    
    def get_version(self) -> Optional[str]:
        """Get Brush version by running brush --version."""
        if not self.is_available():
            return None
        try:
            result = subprocess.run(
                [str(self._brush_bin), "--version"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None

    def train(self, 
              dataset_path: Path, 
              output_path: Path, 
              params: Dict[str, Any], 
              prev_ply_path: Optional[Path] = None,
              mask_path: Optional[Path] = None, 
              callback_data: Optional[Dict] = None) -> Tuple[bool, Optional[subprocess.Popen]]:
        """
        Constructs and runs the 'brush train' command.
        
        Supports:
        - Fixed Topology CLI flags for Stable mode
        - Warm Start via --init flag
        - Progress callbacks
        - Visualization modes: Native GUI, Save Preview Images
        
        Args:
            dataset_path: Path to COLMAP-format dataset
            output_path: Path to output PLY file
            params: Training parameters dict with keys:
                Core:
                - iterations: int (default 4000)
                - splat_count: int (default 500000)
                - learning_rate: float (default 0.00002) - position LR
                - sh_degree: int (default 3)
                
                Densification:
                - densification_interval: int (default 200) -> --refine-every
                - densify_grad_threshold: float (default 0.00004) -> --growth-grad-threshold
                - growth_select_fraction: float (default 0.1) -> --growth-select-fraction
                
                Learning Rates:
                - feature_lr: float (default 0.0025) -> --lr-coeffs-dc
                - gaussian_lr: float (default 0.00016) -> --lr-scale (low for 4D stability)
                - opacity_lr: float (default 0.01) -> --lr-opac
                
                Regularization:
                - scale_loss_weight: float (default 1e-8) -> --scale-loss-weight
                - opac_loss_weight: float (default 1e-9) -> --opac-loss-weight
                
                Fixed Topology:
                - growth_stop_iter: int (optional) -> --growth-stop-iter
                - refine_every: int (optional, legacy) -> --refine-every
                
                Visualization:
                - visualize_training: str ("Off", "Save Preview Images", "Spawn Native GUI")
                - preview_interval: int (steps between eval images)
                - eval_camera_index: int (render every Nth camera)
                
            prev_ply_path: Path to previous frame PLY for warm start
            mask_path: Not used by Brush (reserved for future)
            callback_data: Dict with 'pbar_func' for progress callback
            
        Returns:
            Tuple[bool, Optional[Popen]]: (success, process) - process is returned for external monitoring
        """
        if not self.is_available():
            print(f"[BrushEngine] Error: Brush binary not available. {self._init_error}")
            return (False, None)
        
        iterations = params.get('iterations', 4000)
        
        # Build command - Core required flags
        cmd = [
            str(self._brush_bin), 
            str(dataset_path),
            "--export-path", str(output_path.parent),
            "--export-name", output_path.name,
            "--total-steps", str(iterations),
            # Prevent intermediate exports triggering early kill
            "--export-every", str(iterations),
            
            # Core Params (always passed)
            "--max-splats", str(params.get('splat_count', 500000)),
            "--lr-mean", str(params.get('learning_rate', 0.00002)),
            "--sh-degree", str(params.get('sh_degree', 3)),
        ]
        
        # --- DENSIFICATION / REFINEMENT FLAGS ---
        # These control splat growth behavior
        if 'densification_interval' in params:
            cmd.extend(["--refine-every", str(params['densification_interval'])])
        if 'densify_grad_threshold' in params:
            cmd.extend(["--growth-grad-threshold", str(params['densify_grad_threshold'])])
        if 'growth_select_fraction' in params:
            cmd.extend(["--growth-select-fraction", str(params['growth_select_fraction'])])
        
        # --- LEARNING RATE FLAGS ---
        if 'feature_lr' in params:
            cmd.extend(["--lr-coeffs-dc", str(params['feature_lr'])])
        if 'gaussian_lr' in params:
            cmd.extend(["--lr-scale", str(params['gaussian_lr'])])
        if 'opacity_lr' in params:
            cmd.extend(["--lr-opac", str(params['opacity_lr'])])
        
        # --- REGULARIZATION FLAGS ---
        # These help prevent splat drift toward cameras
        if 'scale_loss_weight' in params:
            cmd.extend(["--scale-loss-weight", str(params['scale_loss_weight'])])
        if 'opac_loss_weight' in params:
            cmd.extend(["--opac-loss-weight", str(params['opac_loss_weight'])])
        
        # --- FIXED TOPOLOGY CLI FLAGS ---
        # For Stable mode: disable all growth after frame 0
        if 'growth_stop_iter' in params:
            cmd.extend(["--growth-stop-iter", str(params['growth_stop_iter'])])
        # Legacy support: refine_every can also be set directly
        if 'refine_every' in params:
            cmd.extend(["--refine-every", str(params['refine_every'])])
        
        # --- VISUALIZATION FLAGS ---
        visualize_mode = params.get('visualize_training', 'Off')
        
        if visualize_mode == "Spawn Native GUI":
            cmd.append("--with-viewer")
            print(f"   👁️ Spawning Native Brush GUI...")
        elif visualize_mode == "Save Preview Images":
            preview_interval = params.get('preview_interval', 500)
            eval_camera_index = params.get('eval_camera_index', 10)
            cmd.extend([
                "--eval-every", str(preview_interval),
                "--eval-save-to-disk",
                "--eval-split-every", str(eval_camera_index)
            ])
            print(f"   📸 Saving Preview Images every {preview_interval} steps (camera index {eval_camera_index})...")
        
        # Warm Start - Note: Brush uses points3D.txt, not --init flag
        if prev_ply_path and params.get('use_init_flag', False):
            cmd.extend(["--init", str(prev_ply_path)])
        
        print(f"[BrushEngine] Running: {' '.join(cmd)}")
        
        # Execute
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        try:
            process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT, 
                text=True, 
                bufsize=1, 
                env=env
            )
            
            # Progress Handling via callback (starts the progress bar thread)
            if callback_data and callback_data.get('pbar_func'):
                t = threading.Thread(target=callback_data['pbar_func'], args=(process,))
                t.daemon = True
                t.start()
            
            # Return process for external monitoring (preview monitor, GUI auto-close)
            # The caller is responsible for waiting and checking the result
            return (True, process)
            
        except Exception as e:
            print(f"[BrushEngine] Execution error: {e}")
            return (False, None)
    
    def wait_for_completion(self, process: subprocess.Popen, output_path: Path, 
                           visualize_mode: str = "Off") -> bool:
        """
        Wait for the Brush process to complete with proper handling.
        
        Args:
            process: The running Brush subprocess
            output_path: Expected output PLY path
            visualize_mode: Visualization mode for special handling
            
        Returns:
            bool: True if training completed successfully
        """
        try:
            # For GUI mode, monitor for output file and auto-close
            if visualize_mode == "Spawn Native GUI":
                while process.poll() is None:
                    if output_path.exists():
                        time.sleep(5)  # Wait a bit to ensure file is fully written
                        if process.poll() is None:
                            print("   ✅ Output found. Auto-Closing Brush GUI...")
                            process.terminate()
                            break
                    time.sleep(2)
            
            # Wait for completion
            process.wait()
            
            # Check return code
            if process.returncode != 0:
                # Ignore error if GUI mode AND output exists (user may have closed manually)
                if visualize_mode == "Spawn Native GUI" and output_path.exists():
                    return True
                print(f"[BrushEngine] Training failed with return code {process.returncode}")
                return False
            
            return output_path.exists()
            
        except Exception as e:
            print(f"[BrushEngine] Wait error: {e}")
            if process.poll() is None:
                process.terminate()
            return False
