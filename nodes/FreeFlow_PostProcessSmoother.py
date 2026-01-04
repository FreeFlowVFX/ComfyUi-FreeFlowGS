import numpy as np
import shutil
from pathlib import Path
from scipy.signal import savgol_filter
from ..utils import FreeFlowUtils

# Minimal PLY Header Parser/Writer to avoid full heavy lib if possible, 
# or use standard logic. Since we just modify positions, we can read/write binary efficiently.
# But for robustness, let's use a simple reliable method.

class FreeFlow_PostProcessSmoother:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ply_sequence": ("PLY_SEQUENCE",),
                "window_size": ("INT", {"default": 5, "min": 3, "max": 25, "step": 2, "tooltip": "Must be odd. Larger = smoother but more latency."}),
                "poly_order": ("INT", {"default": 2, "min": 1, "max": 5, "tooltip": "Polynomial order. 2 is good for preserving some curve."}),
            },
            "optional": {
                "destination_path": ("STRING", {"default": "", "multiline": False, "placeholder": "Optional: Custom save folder path"}),
            }
        }

    RETURN_TYPES = ("PLY_SEQUENCE",)
    RETURN_NAMES = ("smoothed_ply_sequence",)
    FUNCTION = "smooth_sequence"
    CATEGORY = "FreeFlow"

    def smooth_sequence(self, ply_sequence, window_size, poly_order, destination_path=""):
        if len(ply_sequence) < window_size:
            FreeFlowUtils.log("Sequence shorter than window size, returning original.", "WARN")
            return (ply_sequence,)

        import struct

        # 1. Setup Output
        first_ply = Path(ply_sequence[0])
        parent_dir = first_ply.parent
        
        if destination_path and destination_path.strip():
            output_dir = Path(destination_path.strip())
        else:
            output_dir = parent_dir.parent / f"{parent_dir.name}_smoothed"
            
        output_dir.mkdir(parents=True, exist_ok=True)
        
        FreeFlowUtils.log(f"Smoother: Processing {len(ply_sequence)} frames...")
        FreeFlowUtils.log(f"   Window: {window_size}, Poly: {poly_order}")
        FreeFlowUtils.log(f"   Output: {output_dir}")

        # 2. Load Data (T, N, 3)
        # We need to read binary PLY. 
        # Structure: Header \n end_header \n [float x, float y, float z ...]
        # We assume standard 3DGS format where first 3 floats are XYZ.
        
        # Helper to read positions
        def read_ply_params(path):
            with open(path, "rb") as f:
                content = f.read()
            
            # Find end of header
            header_end = content.find(b"end_header\n") + len(b"end_header\n")
            header = content[:header_end]
            body = content[header_end:]
            return header, body

        # Read Frame 0 to establish baseline
        ref_header, _ = read_ply_params(ply_sequence[0])
        
        # We need to parse N (vertex count) from header to ensure safety
        # "element vertex 12345"
        import re
        match = re.search(b"element vertex (\d+)", ref_header)
        if not match:
             raise ValueError("Could not parse vertex count from PLY header")
        
        N = int(match.group(1))
        
        # Now read all frames
        # We only strictly need to smooth XYZ (first 3 floats)
        # But scale/rotation also jitter!
        # Ideally we smooth EVERYTHING.
        # A standard splat has: x,y,z, nx,ny,nz, f_dc_0,1,2, f_rest..., opac, scale_0,1,2, rot_0,1,2
        # That's a lot of floats.
        # For "Cinema Smooth", visual stability mostly comes from XYZ. 
        # Scaling jitter is also bad.
        
        # Strategy: Load ALL floats into a massive buffer (T, N, D).
        # Smooth entire buffer?
        # N=100k, D~60 (SH dims). 100k * 60 * 4 bytes = 24MB per frame. Matches manageable RAM for 100 frames (2.4GB).
        
        # Let's try full smoothing!
        
        frames_data = []
        headers = [] # Keep headers just in case they differ? (They shouldn't for Fixed topology)

        for p in ply_sequence:
            h, b = read_ply_params(p)
            headers.append(h)
            # data is float32 little endian
            data_arr = np.frombuffer(b, dtype=np.float32)
            if len(data_arr) % N != 0:
                 # Topology changed?
                 raise ValueError(f"Frame {p} has inconsistent data length! Fixed topology required.")
                 
            # Reshape to (N, D)
            D = len(data_arr) // N
            frames_data.append(data_arr.reshape(N, D))

        # Stack -> (T, N, D)
        full_tensor = np.stack(frames_data, axis=0)
        
        # Apply Savitzky-Golay along axis 0 (Time)
        # Check window size vs T
        if window_size > len(ply_sequence):
             window_size = len(ply_sequence) // 2 * 2 + 1
        
        FreeFlowUtils.log("   Applying Filter...")
        smoothed_tensor = savgol_filter(full_tensor, window_length=window_size, polyorder=poly_order, axis=0)
        
        # 3. Save
        new_paths = []
        for i, p_str in enumerate(ply_sequence):
            orig_path = Path(p_str)
            new_path = output_dir / orig_path.name
            
            # Write header
            with open(new_path, "wb") as f:
                f.write(headers[i])
                f.write(smoothed_tensor[i].astype(np.float32).tobytes())
            
            new_paths.append(str(new_path))
            
        FreeFlowUtils.log("✅ Smoothing Complete.")
        return (new_paths,)
