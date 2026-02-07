"""
FreeFlow GS Engine - OpenSplat Backend
OpenSplat - Production-quality 3D Gaussian Splatting with Metal/CPU support.

OpenSplat is a standalone C++ implementation that supports:
- CUDA (NVIDIA GPUs)
- Metal (Apple Silicon/AMD on macOS)  
- CPU (fallback, slower but works everywhere)

Unlike nerfstudio/splatfacto which requires CUDA, OpenSplat works natively on Mac.
https://github.com/pierotofy/OpenSplat
"""

import os
import re
import subprocess
import threading
import shutil
import platform
from pathlib import Path
from typing import Optional, Dict, Any, Callable, Tuple

from .base_engine import IGSEngine


class OpenSplatEngine(IGSEngine):
    """
    Implementation of the OpenSplat Backend.
    Production-quality Gaussian Splatting with Mac/CPU support.
    
    Features:
    - Native Metal GPU support on macOS (Apple Silicon)
    - CPU fallback for any platform
    - COLMAP format input (same as FreeFlow output)
    - Standard PLY output
    - Warm start via checkpoint loading
    
    Installation:
    - macOS: brew install cmake opencv pytorch && build from source
    - Or download pre-built binary
    """
    
    # Default installation paths to search
    SEARCH_PATHS = [
        Path.home() / ".freeflow" / "opensplat" / "opensplat",  # FreeFlow managed
        Path.home() / ".local" / "bin" / "opensplat",
        Path("/usr/local/bin/opensplat"),
        Path("/opt/homebrew/bin/opensplat"),
    ]
    
    def __init__(self):
        """Initialize OpenSplatEngine and search for binary."""
        self._binary_path: Optional[Path] = None
        self._version: Optional[str] = None
        self._init_error: Optional[str] = None
        
        # Search for opensplat binary
        self._binary_path = self._find_binary()
        if self._binary_path:
            self._version = self._get_binary_version()
    
    def _find_binary(self) -> Optional[Path]:
        """Search for opensplat binary in common locations."""
        # Check PATH first
        which_result = shutil.which("opensplat")
        if which_result:
            return Path(which_result)
        
        # Check common locations
        for path in self.SEARCH_PATHS:
            if path.exists() and os.access(path, os.X_OK):
                return path
        
        # Check if user set environment variable
        env_path = os.environ.get("OPENSPLAT_PATH")
        if env_path:
            path = Path(env_path)
            if path.exists() and os.access(path, os.X_OK):
                return path
        
        return None
    
    def _get_binary_version(self) -> Optional[str]:
        """Get OpenSplat version from binary."""
        if not self._binary_path:
            return None
        
        try:
            result = subprocess.run(
                [str(self._binary_path), "--help"],
                capture_output=True, text=True, timeout=10
            )
            # Parse version from help output
            # OpenSplat typically shows version in help text
            for line in result.stdout.split("\n") + result.stderr.split("\n"):
                if "version" in line.lower() or "opensplat" in line.lower():
                    # Try to extract version number
                    match = re.search(r'(\d+\.\d+\.?\d*)', line)
                    if match:
                        return match.group(1)
            return "unknown"
        except Exception:
            return None
    
    def is_available(self) -> bool:
        """Check if OpenSplat binary is available."""
        return self._binary_path is not None
    
    def get_name(self) -> str:
        return "OpenSplat (Mac/CPU)"
    
    def get_version(self) -> Optional[str]:
        return self._version
    
    def get_binary_path(self) -> Optional[Path]:
        """Return path to opensplat binary."""
        return self._binary_path
    
    def set_binary_path(self, path: Path) -> bool:
        """
        Manually set the opensplat binary path.
        
        Args:
            path: Path to opensplat executable
            
        Returns:
            True if path is valid and executable
        """
        path = Path(path)
        if path.exists() and os.access(path, os.X_OK):
            self._binary_path = path
            self._version = self._get_binary_version()
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed status information."""
        status = {
            "available": self.is_available(),
            "binary_path": str(self._binary_path) if self._binary_path else None,
            "version": self._version,
            "platform": platform.system(),
            "gpu_backend": self._detect_gpu_backend(),
        }
        return status
    
    def _detect_gpu_backend(self) -> str:
        """Detect which GPU backend OpenSplat will use."""
        if not self._binary_path:
            return "none"
        
        system = platform.system()
        if system == "Darwin":
            # macOS - check for Metal
            # OpenSplat built with Metal will use GPU
            return "metal"
        elif system == "Linux" or system == "Windows":
            # Check for CUDA
            try:
                result = subprocess.run(
                    ["nvidia-smi"], capture_output=True, timeout=5
                )
                if result.returncode == 0:
                    return "cuda"
            except Exception:
                pass
            return "cpu"
        return "cpu"

    def train(self, 
              dataset_path: Path, 
              output_path: Path, 
              params: Dict[str, Any], 
              prev_ply_path: Optional[Path] = None,
              mask_path: Optional[Path] = None, 
              callback_data: Optional[Dict] = None) -> bool:
        """
        Execute OpenSplat training for one frame.
        
        Args:
            dataset_path: Path to COLMAP-format dataset (with images/ and sparse/0/)
            output_path: Path to save the resulting PLY file
            params: Training parameters dict with keys:
                - iterations: int (default 30000)
                - sh_degree: int (default 3)  
                - downscale_factor: int (default 1)
                - output_dir: str (where to save intermediate outputs)
            prev_ply_path: Path to previous PLY for warm start (--resume)
            mask_path: Not used currently
            callback_data: Dict with 'pbar_func' for progress callback
            
        Returns:
            bool: True if training succeeded and PLY was saved.
        """
        if not self.is_available():
            print(f"[OpenSplatEngine] Error: opensplat binary not found")
            return False
        
        dataset_path = Path(dataset_path)
        output_path = Path(output_path)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Build command
        cmd = [
            str(self._binary_path),
            str(dataset_path),
            "-n", str(params.get('iterations', 30000)),
            "-o", str(output_path),
        ]
        
        # Add SH degree if specified
        sh_degree = params.get('sh_degree', 3)
        if sh_degree != 3:
            cmd.extend(["--sh-degree", str(sh_degree)])
        
        # Warm start from previous PLY
        if prev_ply_path and Path(prev_ply_path).exists():
            cmd.extend(["--resume", str(prev_ply_path)])
        
        # Additional parameters
        if params.get('downscale_factor', 1) > 1:
            # OpenSplat may have different syntax for downscaling
            pass  # Check opensplat --help for exact flag
        
        print(f"[OpenSplatEngine] Running: {' '.join(cmd)}")
        
        # Execute training
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
            
            # Progress parsing
            progress_info = {'current': 0, 'total': params.get('iterations', 30000)}
            
            def parse_progress():
                """Parse OpenSplat output for progress updates."""
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    # OpenSplat outputs progress in various formats
                    # Try to parse iteration numbers
                    match = re.search(r'(?:Iteration|Step|iter)\s*[:\s]*(\d+)', line, re.IGNORECASE)
                    if match:
                        progress_info['current'] = int(match.group(1))
                    
                    # Also check for percentage
                    pct_match = re.search(r'(\d+(?:\.\d+)?)\s*%', line)
                    if pct_match:
                        pct = float(pct_match.group(1))
                        progress_info['current'] = int(pct / 100 * progress_info['total'])
                    
                    print(f"[opensplat] {line.rstrip()}")
            
            # Start progress thread
            progress_thread = threading.Thread(target=parse_progress, daemon=True)
            progress_thread.start()
            
            # Wait for completion with interrupt checking
            import comfy.model_management
            try:
                while process.poll() is None:
                    if comfy.model_management.processing_interrupted():
                        process.terminate()
                        print("🚫 Processing Interrupted by User. Terminating OpenSplat...")
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
            
            if process.returncode != 0:
                print(f"[OpenSplatEngine] Training failed with return code {process.returncode}")
                return False
            
            # Verify output exists
            if not output_path.exists():
                # OpenSplat might output to different location, check for splat.ply
                default_output = dataset_path / "splat.ply"
                if default_output.exists():
                    shutil.move(str(default_output), str(output_path))
                else:
                    print(f"[OpenSplatEngine] Error: Output PLY not found at {output_path}")
                    return False
            
            print(f"[OpenSplatEngine] Success! PLY saved to {output_path}")
            return True
            
        except Exception as e:
            print(f"[OpenSplatEngine] Execution error: {e}")
            return False
    
    @classmethod
    def get_install_instructions(cls) -> str:
        """Return installation instructions for the current platform."""
        system = platform.system()
        
        if system == "Darwin":
            return """
OpenSplat Installation (macOS):

Option 1 - Homebrew (recommended):
  brew install cmake opencv pytorch
  git clone https://github.com/pierotofy/OpenSplat
  cd OpenSplat && mkdir build && cd build
  cmake -DGPU_RUNTIME=MPS .. && make -j$(sysctl -n hw.logicalcpu)
  # Copy ./opensplat to ~/.freeflow/opensplat/

Option 2 - Download pre-built:
  Visit https://github.com/pierotofy/OpenSplat/releases
  Download macOS binary and place in ~/.freeflow/opensplat/
"""
        elif system == "Linux":
            return """
OpenSplat Installation (Linux):

  sudo apt install libopencv-dev cmake
  # Download libtorch from https://pytorch.org/get-started/locally/
  
  git clone https://github.com/pierotofy/OpenSplat
  cd OpenSplat && mkdir build && cd build
  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch/ .. && make -j$(nproc)
  # Copy ./opensplat to ~/.freeflow/opensplat/
"""
        elif system == "Windows":
            return """
OpenSplat Installation (Windows):

  Download from https://github.com/pierotofy/OpenSplat/releases
  Or build from source with Visual Studio 2022 + CUDA toolkit
  Place opensplat.exe in %USERPROFILE%\\.freeflow\\opensplat\\
"""
        return "See https://github.com/pierotofy/OpenSplat for installation instructions."
