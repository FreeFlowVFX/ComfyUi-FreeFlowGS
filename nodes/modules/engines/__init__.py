"""
FreeFlow GS Engine - Backend Engines Package
Provides pluggable Gaussian Splatting training backends.

Platform Support:
- BrushEngine: All platforms (Windows, Linux, Mac ARM/Intel)
- SplatfactoEngine: CUDA only (Windows, Linux with NVIDIA GPU)
- OpenSplatEngine: All platforms with OpenCL or CPU fallback
"""

import sys
import platform

from .base_engine import IGSEngine
from .brush_engine import BrushEngine

# --- PLATFORM DETECTION ---
def _check_cuda_platform():
    """
    Check if the current platform can support CUDA-based engines.
    Returns (is_supported, reason_if_not)
    """
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    # macOS doesn't support CUDA at all
    if system == "darwin":
        if "arm" in machine or "aarch64" in machine:
            return False, "Apple Silicon Mac (M1/M2/M3) - no CUDA support"
        else:
            return False, "macOS Intel - no CUDA support (Apple dropped NVIDIA)"
    
    # Linux ARM (like Jetson) might have CUDA, but nerfstudio doesn't support it well
    if system == "linux" and ("arm" in machine or "aarch64" in machine):
        # Could potentially work on Jetson, but let's be conservative
        return False, "Linux ARM - nerfstudio/gsplat may not work"
    
    # Windows and Linux x86_64 - assume CUDA capable (will fail at install if no GPU)
    return True, None

CUDA_SUPPORTED, CUDA_UNSUPPORTED_REASON = _check_cuda_platform()

# SplatfactoEngine - Only available on CUDA-capable platforms
# On Mac/ARM, don't even show the option to prevent confusion
if CUDA_SUPPORTED:
    try:
        from .splatfacto_engine import SplatfactoEngine
        SPLATFACTO_AVAILABLE = True
    except ImportError as e:
        SplatfactoEngine = None
        SPLATFACTO_AVAILABLE = False
        print(f"[Engines] SplatfactoEngine import failed: {e}")
else:
    SplatfactoEngine = None
    SPLATFACTO_AVAILABLE = False
    print(f"[Engines] Splatfacto disabled: {CUDA_UNSUPPORTED_REASON}")

# OpenSplatEngine - Available on all platforms (uses OpenCL or CPU)
try:
    from .opensplat_engine import OpenSplatEngine
    OPENSPLAT_AVAILABLE = True
except ImportError:
    OpenSplatEngine = None
    OPENSPLAT_AVAILABLE = False

__all__ = [
    'IGSEngine', 
    'BrushEngine', 
    'SplatfactoEngine', 
    'SPLATFACTO_AVAILABLE',
    'CUDA_SUPPORTED',
    'CUDA_UNSUPPORTED_REASON',
    'OpenSplatEngine',
    'OPENSPLAT_AVAILABLE',
]
