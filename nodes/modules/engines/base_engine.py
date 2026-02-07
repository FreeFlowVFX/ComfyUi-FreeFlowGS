"""
FreeFlow GS Engine - Base Engine Interface
Abstract Base Class for all Gaussian Splatting backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path


class IGSEngine(ABC):
    """
    Abstract Base Class for all Gaussian Splatting Backends.
    Enforces feature parity: All engines must support Training, Warm Start, and availability check.
    """
    
    @abstractmethod
    def train(self, 
              dataset_path: Path, 
              output_path: Path, 
              params: Dict[str, Any], 
              prev_ply_path: Optional[Path] = None, 
              mask_path: Optional[Path] = None, 
              callback_data: Optional[Dict] = None) -> bool:
        """
        Execute Training for one frame.
        
        Args:
            dataset_path: Path to prepared dataset (images + sparse).
            output_path: Path to save the resulting PLY.
            params: Dictionary of training parameters (iterations, lr, etc.)
            prev_ply_path: Optional path to N-1 PLY (Warm Start).
            mask_path: Optional path to masking data.
            callback_data: Optional dict for progress/preview callbacks.
            
        Returns:
            bool: True if training succeeded, False otherwise.
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this engine is available/installed.
        
        Returns:
            bool: True if the engine can be used, False otherwise.
        """
        pass
    
    def get_name(self) -> str:
        """Return human-readable engine name."""
        return self.__class__.__name__
    
    def get_version(self) -> Optional[str]:
        """Return engine version if available."""
        return None
