"""
FreeFlow GS Engine - Backend Engines Package
Provides pluggable Gaussian Splatting training backends.

IMPORTANT: Availability flags indicate whether the engine CLASS can be imported,
NOT whether the underlying tool is installed. Installation checks happen at runtime
when the user first selects an engine, allowing for auto-install on first use.
"""

from .base_engine import IGSEngine
from .brush_engine import BrushEngine

# SplatfactoEngine - class is always available (nerfstudio installation check is at runtime)
# The engine uses subprocess isolation, so nerfstudio doesn't need to be installed
# for the class to import. Auto-install happens on first use.
from .splatfacto_engine import SplatfactoEngine
SPLATFACTO_AVAILABLE = True  # Class always exists; runtime checks if nerfstudio is installed

# OpenSplatEngine - class is always available (binary detection happens at runtime)
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
