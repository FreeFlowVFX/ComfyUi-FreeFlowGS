"""
FreeFlow GS Engine - Backend Engines Package
Provides pluggable Gaussian Splatting training backends.
"""

from .base_engine import IGSEngine
from .brush_engine import BrushEngine

# SplatfactoEngine imported conditionally to avoid errors when nerfstudio not installed
try:
    from .splatfacto_engine import SplatfactoEngine
    SPLATFACTO_AVAILABLE = True
except ImportError:
    SplatfactoEngine = None
    SPLATFACTO_AVAILABLE = False

# OpenSplatEngine - always available (binary detection happens at runtime)
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
    'OpenSplatEngine',
    'OPENSPLAT_AVAILABLE',
]
