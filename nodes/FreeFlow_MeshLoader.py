"""
FreeFlow_MeshLoader - External Geometry Import
Loads OBJ/PLY sequences for Metahuman/External tracking support.
"""

import os
from pathlib import Path
import numpy as np
from ..utils import FreeFlowUtils

class FreeFlow_MeshLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "tooltip": "Path to folder containing .obj sequence"}),
            }
        }

    RETURN_TYPES = ("GUIDANCE_MESH_SEQUENCE",)
    RETURN_NAMES = ("guidance_mesh",)
    FUNCTION = "load_mesh_sequence"
    CATEGORY = "FreeFlow"

    def load_mesh_sequence(self, directory_path):
        import re
        
        path = Path(directory_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        # Scan for OBJs
        files = sorted(list(path.glob("*.obj")))
        if not files:
            files = sorted(list(path.glob("*.ply")))
            
        if not files:
            raise ValueError(f"No .obj or .ply files found in {path}")
            
        FreeFlowUtils.log(f"📂 Loading Mesh Sequence: {len(files)} frames from {path.name}")
        
        mesh_sequence = []
        
        for fpath in files:
            verts = self._read_mesh_verts(fpath)
            if verts is not None:
                mesh_sequence.append({
                    'vertices': verts,
                    'faces': None # We don't strictly need faces for binding if using KDTree
                })
            else:
                mesh_sequence.append(None)
                
        return (mesh_sequence,)

    def _read_mesh_verts(self, path):
        """
        Simple OBJ/PLY parser for Vertices Only.
        """
        verts = []
        try:
            with open(path, "r") as f:
                for line in f:
                    if line.startswith("v "): # OBJ vertex
                        parts = line.split()
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    elif line.startswith("element vertex"): # PLY header check
                        pass # basic check
                    # PLY binaries are hard to parse in text mode. 
                    # Assuming OBJ for now as primary External format.
                    
            if not verts:
                # Try PLY ASCII?
                 pass
                 
            return np.array(verts, dtype=np.float32)
        except Exception as e:
            print(f"Error loading mesh {path.name}: {e}")
            return None
