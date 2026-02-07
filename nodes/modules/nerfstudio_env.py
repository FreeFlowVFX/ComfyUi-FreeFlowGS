"""
FreeFlow GS Engine - Nerfstudio Environment Manager
Manages an isolated virtual environment for Nerfstudio/Splatfacto backend.

The venv is created INSIDE the FreeFlow project directory to avoid any
system-wide changes or conflicts with ComfyUI's Python environment.

Location: ComfyUi-FreeFlowGS/.nerfstudio_venv/
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Callable, List, Tuple


class NerfstudioEnvironment:
    """
    Manages the isolated Nerfstudio virtual environment.
    
    Location: ComfyUi-FreeFlowGS/.nerfstudio_venv/
    
    IMPORTANT: The venv is stored INSIDE the FreeFlow project directory,
    NOT in the user's home directory. This ensures:
    - No system-wide changes
    - No conflicts with ComfyUI
    - Easy cleanup (just delete the folder)
    - Portable installation
    
    Features:
    - Detects existing nerfstudio installations (conda/venv)
    - Creates isolated venv if needed
    - Provides Python path for subprocess calls
    - Version detection
    """
    
    # Get the FreeFlow project root (parent of nodes/modules/)
    _MODULE_DIR = Path(__file__).parent  # nodes/modules/
    _NODES_DIR = _MODULE_DIR.parent       # nodes/
    PROJECT_ROOT = _NODES_DIR.parent      # ComfyUi-FreeFlowGS/
    
    # Venv is stored INSIDE the project directory (hidden folder)
    VENV_PATH = PROJECT_ROOT / ".nerfstudio_venv"
    
    # Minimum Python version for nerfstudio
    MIN_PYTHON_VERSION = (3, 10)
    MAX_PYTHON_VERSION = (3, 12)  # nerfstudio works best with 3.10-3.11
    
    # Common conda env names to search
    CONDA_ENV_NAMES = ["nerfstudio", "ns", "nerf", "gsplat"]
    
    # Supported PyTorch CUDA versions (maps detected CUDA to PyTorch wheel)
    # PyTorch only has wheels for specific CUDA versions
    PYTORCH_CUDA_WHEELS = {
        "11.8": "cu118",
        "12.1": "cu121", 
        "12.4": "cu124",
    }
    
    @classmethod
    def get_python(cls) -> Optional[Path]:
        """
        Returns path to the venv's Python executable.
        
        Returns:
            Path to python executable, or None if venv doesn't exist.
        """
        if sys.platform == "win32":
            python_path = cls.VENV_PATH / "Scripts" / "python.exe"
        else:
            python_path = cls.VENV_PATH / "bin" / "python"
        
        if python_path.exists():
            return python_path
        return None
    
    @classmethod
    def get_ns_train(cls) -> Optional[Path]:
        """
        Returns path to the ns-train executable.
        
        Returns:
            Path to ns-train, or None if not available.
        """
        if sys.platform == "win32":
            ns_train = cls.VENV_PATH / "Scripts" / "ns-train.exe"
        else:
            ns_train = cls.VENV_PATH / "bin" / "ns-train"
        
        if ns_train.exists():
            return ns_train
        return None
    
    @classmethod
    def get_ns_export(cls) -> Optional[Path]:
        """
        Returns path to the ns-export executable.
        
        Returns:
            Path to ns-export, or None if not available.
        """
        if sys.platform == "win32":
            ns_export = cls.VENV_PATH / "Scripts" / "ns-export.exe"
        else:
            ns_export = cls.VENV_PATH / "bin" / "ns-export"
        
        if ns_export.exists():
            return ns_export
        return None
    
    @classmethod
    def is_installed(cls) -> bool:
        """
        Check if venv exists and nerfstudio is properly installed.
        
        Returns:
            True if nerfstudio is ready to use.
        """
        python = cls.get_python()
        if not python:
            print(f"[NerfstudioEnv] is_installed: No venv Python found")
            return False
        
        # Verify nerfstudio can be imported
        try:
            result = subprocess.run(
                [str(python), "-c", "import nerfstudio; print('ok')"],
                capture_output=True, text=True, timeout=30
            )
            is_ok = result.returncode == 0 and "ok" in result.stdout
            if not is_ok:
                print(f"[NerfstudioEnv] is_installed: nerfstudio import failed")
                if result.stderr:
                    print(f"[NerfstudioEnv]   stderr: {result.stderr[:200]}")
            return is_ok
        except Exception as e:
            print(f"[NerfstudioEnv] is_installed: exception: {e}")
            return False
    
    @classmethod
    def get_version(cls) -> Optional[str]:
        """
        Get installed nerfstudio version.
        
        Returns:
            Version string (e.g., "1.1.5") or None if not installed.
        """
        python = cls.get_python()
        if not python:
            return None
        
        try:
            # Use importlib.metadata which is the modern way to get package versions
            result = subprocess.run(
                [str(python), "-c", 
                 "import importlib.metadata; print(importlib.metadata.version('nerfstudio'))"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return None
    
    @classmethod
    def get_latest_version(cls) -> Optional[str]:
        """
        Get the latest nerfstudio version from PyPI.
        
        Returns:
            Latest version string or None if check failed.
        """
        try:
            import urllib.request
            import json
            
            url = "https://pypi.org/pypi/nerfstudio/json"
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                return data.get("info", {}).get("version")
        except Exception:
            return None
    
    @classmethod
    def check_for_update(cls) -> Optional[str]:
        """
        Check if a newer version of nerfstudio is available.
        
        Returns:
            New version string if update available, None otherwise.
        """
        current = cls.get_version()
        if not current:
            return None
        
        latest = cls.get_latest_version()
        if not latest:
            return None
        
        # Simple version comparison (works for semantic versioning)
        try:
            from packaging import version
            if version.parse(latest) > version.parse(current):
                return latest
        except ImportError:
            # Fallback to string comparison
            if latest != current:
                # Basic check - if latest starts with higher major/minor
                curr_parts = current.split('.')
                latest_parts = latest.split('.')
                for c, l in zip(curr_parts, latest_parts):
                    try:
                        if int(l) > int(c):
                            return latest
                        elif int(l) < int(c):
                            return None
                    except ValueError:
                        pass
        return None
    
    @classmethod
    def upgrade(cls, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """
        Upgrade nerfstudio to the latest version.
        
        Args:
            progress_callback: Optional callback(message, progress)
            
        Returns:
            True if upgrade succeeded.
        """
        def report(msg: str, progress: float):
            print(f"[NerfstudioEnv] {msg}")
            if progress_callback:
                progress_callback(msg, progress)
        
        python = cls.get_python()
        if not python:
            report("ERROR: Nerfstudio venv not found", 0.0)
            return False
        
        report("Upgrading nerfstudio...", 0.2)
        
        try:
            result = subprocess.run(
                [str(python), "-m", "pip", "install", "--upgrade", "nerfstudio"],
                capture_output=True, text=True, timeout=1800
            )
            if result.returncode != 0:
                report(f"ERROR: Upgrade failed: {result.stderr[:500]}", 0.0)
                return False
            
            # Verify upgrade
            new_version = cls.get_version()
            report(f"SUCCESS: Upgraded to nerfstudio {new_version}", 1.0)
            return True
            
        except Exception as e:
            report(f"ERROR: Upgrade exception: {e}", 0.0)
            return False
    
    @classmethod
    def detect_existing(cls) -> Optional[Tuple[str, Path]]:
        """
        Try to find an existing nerfstudio installation.
        
        Checks:
        1. Conda environments with common names
        2. System PATH for ns-train
        3. Common venv locations
        
        Returns:
            Tuple of (source, path) if found, None otherwise.
            source: "conda", "path", or "venv"
        """
        # 1. Check conda environments
        conda_info = cls._get_conda_envs()
        for env_name in cls.CONDA_ENV_NAMES:
            if env_name in conda_info:
                env_path = Path(conda_info[env_name])
                # Verify nerfstudio is installed in this env
                if cls._verify_nerfstudio_in_env(env_path):
                    return ("conda", env_path)
        
        # 2. Check system PATH
        ns_train = shutil.which("ns-train")
        if ns_train:
            # ns-train found in PATH, get its venv/conda root
            ns_train_path = Path(ns_train)
            if "envs" in str(ns_train_path):
                # Conda environment
                env_path = ns_train_path.parent.parent
                return ("conda", env_path)
            elif "venv" in str(ns_train_path) or ".venv" in str(ns_train_path):
                env_path = ns_train_path.parent.parent
                return ("venv", env_path)
            else:
                return ("path", ns_train_path.parent.parent)
        
        # 3. Check common venv locations
        common_venv_paths = [
            Path.home() / ".nerfstudio" / "venv",
            Path.home() / "nerfstudio_venv",
            Path.home() / ".local" / "nerfstudio",
        ]
        for venv_path in common_venv_paths:
            if venv_path.exists() and cls._verify_nerfstudio_in_env(venv_path):
                return ("venv", venv_path)
        
        return None
    
    @classmethod
    def _get_conda_envs(cls) -> dict:
        """Get dict of conda environment names to paths."""
        envs = {}
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                for env_path in data.get("envs", []):
                    env_name = Path(env_path).name
                    envs[env_name] = env_path
        except Exception:
            pass
        return envs
    
    @classmethod
    def _verify_nerfstudio_in_env(cls, env_path: Path) -> bool:
        """Check if nerfstudio is installed in a given env."""
        if sys.platform == "win32":
            python = env_path / "Scripts" / "python.exe"
        else:
            python = env_path / "bin" / "python"
        
        if not python.exists():
            return False
        
        try:
            result = subprocess.run(
                [str(python), "-c", "import nerfstudio"],
                capture_output=True, timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False
    
    @classmethod
    def get_requirements(cls) -> List[str]:
        """
        Get list of pip packages to install for nerfstudio.
        
        Returns:
            List of package specifiers.
        """
        return [
            "torch>=2.0.0",
            "torchvision",
            "nerfstudio>=1.0.0",
            "gsplat>=1.0.0",
        ]
    
    @classmethod
    def create_venv(cls, progress_callback: Optional[Callable[[str, float], None]] = None) -> bool:
        """
        Create the isolated nerfstudio venv and install dependencies.
        
        Args:
            progress_callback: Optional callback(message, progress) where progress is 0.0-1.0
        
        Returns:
            True if venv was created successfully.
        """
        import platform as plat
        
        def report(msg: str, progress: float):
            print(f"[NerfstudioEnv] {msg}")
            if progress_callback:
                progress_callback(msg, progress)
        
        print(f"[NerfstudioEnv] === Starting venv creation ===")
        print(f"[NerfstudioEnv] Platform: {sys.platform}")
        print(f"[NerfstudioEnv] Architecture: {plat.machine()}")
        print(f"[NerfstudioEnv] PROJECT_ROOT: {cls.PROJECT_ROOT}")
        print(f"[NerfstudioEnv] VENV_PATH: {cls.VENV_PATH}")
        
        # --- PLATFORM CHECK: Fail early on unsupported platforms ---
        if sys.platform == "darwin":
            machine = plat.machine().lower()
            if "arm" in machine or "aarch64" in machine:
                report("ERROR: Apple Silicon Mac (M1/M2/M3) detected - CUDA not supported", 0.0)
                report("Nerfstudio/Splatfacto requires NVIDIA CUDA. Use Brush engine instead.", 0.0)
                return False
            else:
                report("ERROR: macOS Intel detected - no CUDA support (Apple dropped NVIDIA)", 0.0)
                report("Nerfstudio/Splatfacto requires NVIDIA CUDA. Use Brush engine instead.", 0.0)
                return False
        
        if sys.platform == "linux":
            machine = plat.machine().lower()
            if "arm" in machine or "aarch64" in machine:
                report("WARNING: Linux ARM detected - gsplat may not work on this platform", 0.0)
                # Continue anyway - Jetson might work
        
        # Ensure project directory exists (it should, but just in case)
        if not cls.PROJECT_ROOT.exists():
            report(f"ERROR: Project root not found: {cls.PROJECT_ROOT}", 0.0)
            return False
        print(f"[NerfstudioEnv] Project root verified: {cls.PROJECT_ROOT.exists()}")
        
        # Check if venv already exists
        if cls.VENV_PATH.exists():
            print(f"[NerfstudioEnv] VENV_PATH exists, checking if valid...")
            if cls.is_installed():
                report("Nerfstudio venv already exists and is valid", 1.0)
                return True
            else:
                report("Existing venv is corrupted, removing...", 0.05)
                shutil.rmtree(cls.VENV_PATH)
        
        report("Creating virtual environment...", 0.1)
        
        # Find suitable Python
        python_cmd = cls._find_suitable_python()
        if not python_cmd:
            report(f"ERROR: No suitable Python found (need {cls.MIN_PYTHON_VERSION[0]}.{cls.MIN_PYTHON_VERSION[1]} - {cls.MAX_PYTHON_VERSION[0]}.{cls.MAX_PYTHON_VERSION[1]})", 0.0)
            return False
        
        # Create venv
        print(f"[NerfstudioEnv] Creating venv with: {python_cmd} -m venv {cls.VENV_PATH}")
        try:
            result = subprocess.run(
                [python_cmd, "-m", "venv", str(cls.VENV_PATH)],
                capture_output=True, text=True, timeout=120
            )
            if result.returncode != 0:
                report(f"ERROR: Failed to create venv: {result.stderr}", 0.0)
                return False
            print(f"[NerfstudioEnv] venv created successfully (returncode=0)")
        except Exception as e:
            report(f"ERROR: Exception creating venv: {e}", 0.0)
            return False
        
        report("Virtual environment created", 0.2)
        
        # Get venv python - with extra debugging
        print(f"[NerfstudioEnv] Looking for venv Python...")
        if sys.platform == "win32":
            expected_python = cls.VENV_PATH / "Scripts" / "python.exe"
        else:
            expected_python = cls.VENV_PATH / "bin" / "python"
        print(f"[NerfstudioEnv] Expected Python path: {expected_python}")
        print(f"[NerfstudioEnv] Expected Python exists: {expected_python.exists()}")
        
        python = cls.get_python()
        if not python:
            report("ERROR: Could not find venv Python after creation", 0.0)
            # List directory contents for debugging
            try:
                if cls.VENV_PATH.exists():
                    print(f"[NerfstudioEnv] VENV_PATH contents: {list(cls.VENV_PATH.iterdir())}")
                    if sys.platform == "win32":
                        scripts_dir = cls.VENV_PATH / "Scripts"
                    else:
                        scripts_dir = cls.VENV_PATH / "bin"
                    if scripts_dir.exists():
                        print(f"[NerfstudioEnv] {scripts_dir.name}/ contents: {list(scripts_dir.iterdir())}")
            except Exception as e:
                print(f"[NerfstudioEnv] Error listing directory: {e}")
            return False
        
        print(f"[NerfstudioEnv] Found venv Python: {python}")
        
        # Upgrade pip
        report("Upgrading pip...", 0.25)
        try:
            subprocess.run(
                [str(python), "-m", "pip", "install", "--upgrade", "pip"],
                capture_output=True, timeout=120
            )
        except Exception:
            pass  # Non-fatal
        
        # Install PyTorch first (with CUDA if available)
        report("Installing PyTorch (this may take several minutes)...", 0.3)
        torch_result = cls._install_pytorch(python, progress_callback)
        if not torch_result:
            report("WARNING: PyTorch installation may have issues", 0.35)
        
        # Install nerfstudio
        report("Installing nerfstudio (this may take several minutes)...", 0.5)
        try:
            result = subprocess.run(
                [str(python), "-m", "pip", "install", "nerfstudio>=1.0.0"],
                capture_output=True, text=True, timeout=1800  # 30 minutes
            )
            if result.returncode != 0:
                report(f"ERROR: Failed to install nerfstudio: {result.stderr[:500]}", 0.0)
                return False
        except subprocess.TimeoutExpired:
            report("ERROR: Nerfstudio installation timed out (>30 min)", 0.0)
            return False
        except Exception as e:
            report(f"ERROR: Exception installing nerfstudio: {e}", 0.0)
            return False
        
        report("Nerfstudio installed", 0.8)
        
        # Install gsplat (for splatfacto)
        # On Windows, gsplat requires MSVC to JIT-compile CUDA kernels
        # We provide pre-built wheels to avoid this requirement
        report("Installing gsplat...", 0.85)
        gsplat_installed = cls._install_gsplat(python, progress_callback)
        if not gsplat_installed:
            report("WARNING: gsplat installation may have issues", 0.88)
        
        # Verify installation
        report("Verifying installation...", 0.95)
        if cls.is_installed():
            version = cls.get_version()
            report(f"SUCCESS: Nerfstudio {version} installed at {cls.VENV_PATH}", 1.0)
            return True
        else:
            report("ERROR: Installation verification failed", 0.0)
            return False
    
    @classmethod
    def _find_suitable_python(cls) -> Optional[str]:
        """Find a Python executable that meets version requirements."""
        print(f"[NerfstudioEnv] Searching for Python {cls.MIN_PYTHON_VERSION[0]}.{cls.MIN_PYTHON_VERSION[1]}-{cls.MAX_PYTHON_VERSION[0]}.{cls.MAX_PYTHON_VERSION[1]}")
        print(f"[NerfstudioEnv] Platform: {sys.platform}")
        
        # Check common python commands - on Windows, also try py launcher
        if sys.platform == "win32":
            python_commands = ["python3.11", "python3.10", "python3", "python", "py -3.11", "py -3.10", "py"]
        else:
            python_commands = ["python3.11", "python3.10", "python3", "python"]
        
        for cmd in python_commands:
            # Handle Windows py launcher (contains spaces)
            if " " in cmd:
                parts = cmd.split()
                python_path = shutil.which(parts[0])
                if python_path:
                    check_cmd = [python_path] + parts[1:] + ["-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"]
                else:
                    print(f"[NerfstudioEnv]   {cmd}: not found in PATH")
                    continue
            else:
                python_path = shutil.which(cmd)
                if not python_path:
                    print(f"[NerfstudioEnv]   {cmd}: not found in PATH")
                    continue
                check_cmd = [python_path, "-c", "import sys; print(sys.version_info.major, sys.version_info.minor)"]
            
            try:
                result = subprocess.run(
                    check_cmd,
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split()
                    major, minor = int(parts[0]), int(parts[1])
                    print(f"[NerfstudioEnv]   {cmd}: Python {major}.{minor} at {python_path}")
                    if cls.MIN_PYTHON_VERSION <= (major, minor) <= cls.MAX_PYTHON_VERSION:
                        print(f"[NerfstudioEnv]   -> Selected: {python_path}")
                        return python_path
                    else:
                        print(f"[NerfstudioEnv]   -> Version not in required range")
                else:
                    print(f"[NerfstudioEnv]   {cmd}: failed with code {result.returncode}")
                    if result.stderr:
                        print(f"[NerfstudioEnv]      stderr: {result.stderr[:200]}")
            except Exception as e:
                print(f"[NerfstudioEnv]   {cmd}: exception: {e}")
                continue
        
        print(f"[NerfstudioEnv] No suitable Python found!")
        return None
    
    @classmethod
    def _install_pytorch(cls, python: Path, progress_callback: Optional[Callable] = None) -> bool:
        """
        Install PyTorch with appropriate CUDA support.
        
        Attempts to detect CUDA and install matching PyTorch version.
        Maps detected CUDA version to the closest supported PyTorch wheel.
        """
        print(f"[NerfstudioEnv] === Installing PyTorch ===")
        print(f"[NerfstudioEnv] Using venv Python: {python}")
        
        # Try to detect CUDA
        cuda_version = cls._detect_cuda_version()
        
        if cuda_version:
            # Map detected CUDA to supported PyTorch wheel
            # PyTorch only provides wheels for specific CUDA versions
            cuda_suffix = cls._map_cuda_to_pytorch_wheel(cuda_version)
            
            if cuda_suffix:
                pip_args = [
                    str(python), "-m", "pip", "install",
                    "torch", "torchvision",
                    "--index-url", f"https://download.pytorch.org/whl/{cuda_suffix}"
                ]
                print(f"[NerfstudioEnv] Installing PyTorch with CUDA {cuda_version} -> {cuda_suffix}")
            else:
                # No matching wheel, use default PyPI (may get CPU version)
                print(f"[NerfstudioEnv] WARNING: CUDA {cuda_version} not directly supported by PyTorch wheels")
                print(f"[NerfstudioEnv] Trying cu124 (latest stable) as fallback...")
                pip_args = [
                    str(python), "-m", "pip", "install",
                    "torch", "torchvision",
                    "--index-url", "https://download.pytorch.org/whl/cu124"
                ]
        else:
            # CPU only or let pip figure it out
            pip_args = [
                str(python), "-m", "pip", "install",
                "torch", "torchvision"
            ]
            print(f"[NerfstudioEnv] Installing PyTorch (CPU only or auto-detect)")
        
        print(f"[NerfstudioEnv] Command: {' '.join(pip_args)}")
        
        try:
            result = subprocess.run(
                pip_args,
                capture_output=True, text=True, timeout=1800  # 30 minutes
            )
            if result.returncode == 0:
                print(f"[NerfstudioEnv] PyTorch installed successfully")
                return True
            else:
                print(f"[NerfstudioEnv] PyTorch installation failed with code {result.returncode}")
                if result.stderr:
                    print(f"[NerfstudioEnv] stderr (last 500 chars): {result.stderr[-500:]}")
                if result.stdout:
                    print(f"[NerfstudioEnv] stdout (last 500 chars): {result.stdout[-500:]}")
                return False
        except subprocess.TimeoutExpired:
            print(f"[NerfstudioEnv] PyTorch installation timed out (>30 min)")
            return False
        except Exception as e:
            print(f"[NerfstudioEnv] PyTorch installation exception: {e}")
            return False
    
    @classmethod
    def _map_cuda_to_pytorch_wheel(cls, cuda_version: str) -> Optional[str]:
        """
        Map detected CUDA version to closest supported PyTorch wheel.
        
        PyTorch only provides wheels for specific CUDA versions:
        - cu118 (CUDA 11.8)
        - cu121 (CUDA 12.1)
        - cu124 (CUDA 12.4)
        
        Returns:
            PyTorch wheel suffix (e.g., 'cu124') or None if no match.
        """
        try:
            major, minor = cuda_version.split(".")[:2]
            cuda_major = int(major)
            cuda_minor = int(minor.split(".")[0]) if "." in minor else int(minor)
        except (ValueError, IndexError):
            print(f"[NerfstudioEnv] Could not parse CUDA version: {cuda_version}")
            return None
        
        # Map to closest supported version
        if cuda_major == 11:
            if cuda_minor >= 8:
                return "cu118"
            else:
                print(f"[NerfstudioEnv] CUDA 11.{cuda_minor} is too old, need 11.8+")
                return None
        elif cuda_major == 12:
            if cuda_minor >= 4:
                return "cu124"
            elif cuda_minor >= 1:
                return "cu121"
            else:
                return "cu121"  # CUDA 12.0 -> use cu121
        elif cuda_major >= 13:
            # CUDA 13.x is newer than any PyTorch wheel - use latest (cu124)
            print(f"[NerfstudioEnv] CUDA {cuda_version} is newer than PyTorch wheels, using cu124")
            return "cu124"
        else:
            print(f"[NerfstudioEnv] CUDA {cuda_major}.x not supported")
            return None
    
    # GitHub repository for pre-built gsplat wheels
    GSPLAT_WHEELS_REPO = "FreeFlowVFX/ComfyUi-FreeFlowGS"
    GSPLAT_VERSION = "1.5.3"
    
    @classmethod
    def _install_gsplat(cls, python: Path, progress_callback: Optional[Callable] = None) -> bool:
        """
        Install gsplat with pre-built wheels for Windows/Linux.
        
        Platform support:
        - Windows: Pre-built wheels essential (gsplat requires MSVC to JIT-compile)
        - Linux: Pre-built wheels for convenience (also available from gsplat team)
        - macOS: NOT SUPPORTED - gsplat requires CUDA which doesn't exist on Mac
        
        We solve the Windows problem by:
        1. First trying our pre-built wheels from GitHub releases
        2. Falling back to PyPI (which requires MSVC on Windows)
        
        Why gsplat matters:
        - Splatfacto uses gsplat for GPU-accelerated Gaussian rasterization
        - Splatfacto is critical for 4D GS because it supports checkpoint/warm-start
        - Warm-start means Frame N+1 trains from Frame N's state = temporal coherence
        - Brush does NOT support warm-start, so Splatfacto is essential for quality 4D
        """
        print(f"[NerfstudioEnv] === Installing gsplat ===")
        print(f"[NerfstudioEnv] Platform: {sys.platform}")
        
        # Check for macOS - gsplat requires CUDA which doesn't exist on Mac
        if sys.platform == "darwin":
            print(f"[NerfstudioEnv] WARNING: macOS detected - gsplat requires CUDA (NVIDIA GPU)")
            print(f"[NerfstudioEnv] Splatfacto will NOT work on Mac. Use Brush engine instead.")
            print(f"[NerfstudioEnv] Attempting PyPI install anyway (will likely fail)...")
        
        # Detect Python version and CUDA version for wheel matching
        try:
            result = subprocess.run(
                [str(python), "-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
                capture_output=True, text=True, timeout=10
            )
            py_version = result.stdout.strip() if result.returncode == 0 else "3.11"
        except Exception:
            py_version = "3.11"
        
        print(f"[NerfstudioEnv] Python version: {py_version}")
        
        # Get CUDA version used by installed PyTorch
        cuda_suffix = None
        try:
            result = subprocess.run(
                [str(python), "-c", "import torch; print(torch.version.cuda or 'none')"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                torch_cuda = result.stdout.strip()
                if torch_cuda and torch_cuda != 'none':
                    # Convert "12.4" -> "cu124"
                    parts = torch_cuda.split(".")
                    if len(parts) >= 2:
                        cuda_suffix = f"cu{parts[0]}{parts[1]}"
                        print(f"[NerfstudioEnv] PyTorch CUDA version: {torch_cuda} -> {cuda_suffix}")
        except Exception as e:
            print(f"[NerfstudioEnv] Could not detect PyTorch CUDA version: {e}")
        
        # Try pre-built wheel for Windows only (Linux has wheels from gsplat team on PyPI)
        if sys.platform == "win32" and cuda_suffix:
            print(f"[NerfstudioEnv] Windows detected - trying pre-built gsplat wheel...")
            
            # Detect PyTorch version for wheel matching
            torch_version = None
            try:
                result = subprocess.run(
                    [str(python), "-c", "import torch; v=torch.__version__.split('+')[0]; print('.'.join(v.split('.')[:3]))"],
                    capture_output=True, text=True, timeout=30
                )
                if result.returncode == 0:
                    torch_version = result.stdout.strip()  # e.g., "2.6"
                    print(f"[NerfstudioEnv] PyTorch version: {torch_version}")
            except Exception as e:
                print(f"[NerfstudioEnv] Could not detect PyTorch version: {e}")
            
            # Construct wheel URL from our GitHub releases
            # Naming: gsplat-{version}+pt{torch}cu{cuda}-cp{py}-cp{py}-win_amd64.whl
            # Example: gsplat-1.5.3+pt260cu124-cp312-cp312-win_amd64.whl
            py_ver_short = py_version.replace(".", "")  # "3.12" -> "312"
            torch_ver_short = torch_version.replace(".", "") if torch_version else "260"  # "2.6.0" -> "260", fallback to 260
            wheel_name = f"gsplat-{cls.GSPLAT_VERSION}+pt{torch_ver_short}{cuda_suffix}-cp{py_ver_short}-cp{py_ver_short}-win_amd64.whl"
            # URL-encode the + as %2B for the download URL
            wheel_url = f"https://github.com/{cls.GSPLAT_WHEELS_REPO}/releases/download/gsplat-v{cls.GSPLAT_VERSION}-{cuda_suffix}/{wheel_name.replace('+', '%2B')}"
            
            print(f"[NerfstudioEnv] Trying wheel: {wheel_url}")
            
            try:
                result = subprocess.run(
                    [str(python), "-m", "pip", "install", wheel_url],
                    capture_output=True, text=True, timeout=300
                )
                if result.returncode == 0:
                    print(f"[NerfstudioEnv] gsplat installed from pre-built wheel!")
                    return True
                else:
                    print(f"[NerfstudioEnv] Pre-built wheel not available: {result.stderr[:200]}")
                    print(f"[NerfstudioEnv] Falling back to PyPI...")
            except Exception as e:
                print(f"[NerfstudioEnv] Pre-built wheel download failed: {e}")
                print(f"[NerfstudioEnv] Falling back to PyPI...")
        
        # Fallback: install from PyPI
        print(f"[NerfstudioEnv] Installing gsplat from PyPI...")
        if sys.platform == "win32":
            print(f"[NerfstudioEnv] WARNING: This requires Visual Studio Build Tools on Windows!")
            print(f"[NerfstudioEnv] If this fails, install VS Build Tools with 'Desktop development with C++'")
            print(f"[NerfstudioEnv] Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        
        try:
            result = subprocess.run(
                [str(python), "-m", "pip", "install", f"gsplat>={cls.GSPLAT_VERSION}"],
                capture_output=True, text=True, timeout=600
            )
            if result.returncode == 0:
                print(f"[NerfstudioEnv] gsplat installed from PyPI")
                return True
            else:
                print(f"[NerfstudioEnv] gsplat installation failed: {result.stderr[:500]}")
                if sys.platform == "win32" and "cl.exe" in result.stderr.lower() or "msvc" in result.stderr.lower():
                    print(f"[NerfstudioEnv] ")
                    print(f"[NerfstudioEnv] ============================================")
                    print(f"[NerfstudioEnv] MSVC COMPILER NOT FOUND")
                    print(f"[NerfstudioEnv] ============================================")
                    print(f"[NerfstudioEnv] gsplat needs to compile CUDA code but MSVC is missing.")
                    print(f"[NerfstudioEnv] ")
                    print(f"[NerfstudioEnv] To fix this:")
                    print(f"[NerfstudioEnv] 1. Download Visual Studio Build Tools 2022:")
                    print(f"[NerfstudioEnv]    https://visualstudio.microsoft.com/visual-cpp-build-tools/")
                    print(f"[NerfstudioEnv] 2. Install with 'Desktop development with C++' selected")
                    print(f"[NerfstudioEnv] 3. Restart ComfyUI and try again")
                    print(f"[NerfstudioEnv] ============================================")
                return False
        except subprocess.TimeoutExpired:
            print(f"[NerfstudioEnv] gsplat installation timed out")
            return False
        except Exception as e:
            print(f"[NerfstudioEnv] gsplat installation exception: {e}")
            return False
    
    @classmethod
    def _detect_cuda_version(cls) -> Optional[str]:
        """Detect installed CUDA version."""
        print(f"[NerfstudioEnv] Detecting CUDA...")
        
        # Try nvidia-smi
        try:
            nvidia_smi = shutil.which("nvidia-smi")
            print(f"[NerfstudioEnv]   nvidia-smi: {nvidia_smi or 'not found'}")
            
            if nvidia_smi:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    driver_version = result.stdout.strip()
                    print(f"[NerfstudioEnv]   Driver version: {driver_version}")
                    
                    # nvidia-smi works, try to get CUDA version
                    result2 = subprocess.run(
                        ["nvidia-smi"],
                        capture_output=True, text=True, timeout=10
                    )
                    # Parse CUDA Version from nvidia-smi output
                    for line in result2.stdout.split("\n"):
                        if "CUDA Version" in line:
                            # Extract version like "12.1"
                            import re
                            match = re.search(r"CUDA Version:\s*(\d+\.\d+)", line)
                            if match:
                                cuda_ver = match.group(1)
                                print(f"[NerfstudioEnv]   CUDA version: {cuda_ver}")
                                return cuda_ver
                else:
                    print(f"[NerfstudioEnv]   nvidia-smi failed with code {result.returncode}")
                    if result.stderr:
                        print(f"[NerfstudioEnv]   stderr: {result.stderr[:200]}")
        except Exception as e:
            print(f"[NerfstudioEnv]   nvidia-smi exception: {e}")
        
        # Try nvcc
        try:
            nvcc_path = shutil.which("nvcc")
            print(f"[NerfstudioEnv]   nvcc: {nvcc_path or 'not found'}")
            
            if nvcc_path:
                result = subprocess.run(
                    ["nvcc", "--version"],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0:
                    import re
                    match = re.search(r"release (\d+\.\d+)", result.stdout)
                    if match:
                        cuda_ver = match.group(1)
                        print(f"[NerfstudioEnv]   CUDA version (from nvcc): {cuda_ver}")
                        return cuda_ver
        except Exception as e:
            print(f"[NerfstudioEnv]   nvcc exception: {e}")
        
        print(f"[NerfstudioEnv]   No CUDA detected")
        return None
    
    @classmethod
    def run_command(cls, cmd: List[str], timeout: int = 3600, 
                    env: Optional[dict] = None) -> Tuple[int, str, str]:
        """
        Run a command using the nerfstudio venv's Python.
        
        Args:
            cmd: Command list (will prepend venv python if first arg is 'python')
            timeout: Timeout in seconds
            env: Optional environment variables to add
        
        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        python = cls.get_python()
        if not python:
            return (-1, "", "Nerfstudio venv not found")
        
        # Replace 'python' with venv python
        if cmd and cmd[0] in ("python", "python3"):
            cmd = [str(python)] + cmd[1:]
        
        # Set up environment
        run_env = os.environ.copy()
        run_env["PYTHONUNBUFFERED"] = "1"
        if env:
            run_env.update(env)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=run_env
            )
            return (result.returncode, result.stdout, result.stderr)
        except subprocess.TimeoutExpired:
            return (-2, "", f"Command timed out after {timeout}s")
        except Exception as e:
            return (-1, "", str(e))
    
    @classmethod
    def get_status_info(cls) -> dict:
        """
        Get comprehensive status information about the nerfstudio environment.
        
        Returns:
            Dict with status details for UI display.
        """
        info = {
            "venv_path": str(cls.VENV_PATH),
            "venv_exists": cls.VENV_PATH.exists(),
            "is_installed": False,
            "version": None,
            "python_path": None,
            "ns_train_path": None,
            "existing_installation": None,
            "cuda_available": False,
            "cuda_version": None,
        }
        
        if cls.VENV_PATH.exists():
            python = cls.get_python()
            if python:
                info["python_path"] = str(python)
            
            ns_train = cls.get_ns_train()
            if ns_train:
                info["ns_train_path"] = str(ns_train)
            
            info["is_installed"] = cls.is_installed()
            if info["is_installed"]:
                info["version"] = cls.get_version()
        
        # Check for existing installation elsewhere
        existing = cls.detect_existing()
        if existing:
            info["existing_installation"] = {
                "source": existing[0],
                "path": str(existing[1])
            }
        
        # Check CUDA
        cuda_ver = cls._detect_cuda_version()
        if cuda_ver:
            info["cuda_available"] = True
            info["cuda_version"] = cuda_ver
        
        return info
