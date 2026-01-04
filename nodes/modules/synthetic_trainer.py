
import struct
import math
import random
import time
from pathlib import Path
import logging

class SyntheticTrainer:
    """
    A Synthetic 4D Gaussian Splatting Trainer for Mac/CPU fallback.
    It takes a sparse point cloud (COLMAP) and generates a sequence of PLY files
    that represent a 4D Gaussian Splat (Initialized points + procedural motion).
    
    This ensures the 'FreeFlow' pipeline works end-to-end even without CUDA.
    """
    
    def __init__(self):
        self.logger = logging.getLogger("FreeFlow.SyntheticTrainer")

    def train_sequence(self, sparse_path, output_dir, num_frames=30, splat_count=30000):
        """
        Generates a sequence of PLY files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Load Sparse Points
        points = self._load_colmap_points(sparse_path / "points3D.bin")
        if not points:
            self.logger.warning("No points found in sparse path. Generating random cloud.")
            points = self._generate_random_points(1000)
            
        self.logger.info(f"Loaded {len(points)} sparse points. Densifying to {splat_count}...")
        
        # 2. Densify (Simple replication with noise)
        gaussians = self._densify_points(points, splat_count)
        
        # 3. Generate Frames
        ply_files = []
        start_time = time.time()
        
        for i in range(num_frames):
            frame_time = i / float(num_frames)
            
            # Apply procedural motion (4D Simulation)
            # e.g. a gentle sine wave ripple based on position
            frame_gaussians = self._apply_motion(gaussians, frame_time)
            
            ply_path = output_dir / f"frame_{i:04d}.ply"
            self._save_ply(frame_gaussians, ply_path)
            ply_files.append(str(ply_path))
            
            # Simulate "Training" effort
            if i % 5 == 0:
                self.logger.info(f"Synthesizing Frame {i}/{num_frames}...")
                
        total_time = time.time() - start_time
        self.logger.info(f"Synthetic Training Complete in {total_time:.2f}s")
        return ply_files
    
    def _load_colmap_points(self, path):
        if not path.exists():
            return None
        points = []
        try:
            with open(path, "rb") as f:
                num_points = struct.unpack("<Q", f.read(8))[0]
                for _ in range(num_points):
                    # COLMAP Binary format: ID(u64), X(f64), Y, Z, R(u8), G, B, Error(f64), TrackLen(u64)
                    _ = f.read(8) # ID
                    x, y, z = struct.unpack("<ddd", f.read(24))
                    r, g, b = struct.unpack("<BBB", f.read(3))
                    _ = f.read(8) # Error
                    track_len = struct.unpack("<Q", f.read(8))[0]
                    f.read(track_len * 8) # Tracks (ImageId, Point2DIdx)
                    points.append({'pos': [x, y, z], 'color': [r, g, b]})
        except Exception as e:
            self.logger.error(f"Error reading points3D.bin: {e}")
            return None
        return points

    def _generate_random_points(self, count):
        points = []
        for _ in range(count):
             points.append({
                 'pos': [random.uniform(-2,2), random.uniform(-2,2), random.uniform(-2,2)],
                 'color': [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
             })
        return points

    def _densify_points(self, points, target_count):
        """Turn sparse points into 3D Gaussian attributes"""
        gaussians = []
        src_len = len(points)
        
        for i in range(target_count):
            src = points[i % src_len]
            # Gaussian Attributes
            # xyz: float32
            # normal: float32 (3) (not strictly needed for basic splat)
            # f_dc: float32 (3) (SH coefficients, 0-th band is color)
            # scale: float32 (3)
            # rotation: float32 (4) (quaternion)
            # opacity: float32 (sigmoid)
            
            # Position noise
            noise = 0.05
            pos = [
                src['pos'][0] + random.uniform(-noise, noise),
                src['pos'][1] + random.uniform(-noise, noise),
                src['pos'][2] + random.uniform(-noise, noise)
            ]
            
            # Color to SH (approx: (color/255 - 0.5) / 0.28209)
            # Simplified: Just store 0.5 + color
            # Actually standard PLY for splats stores f_dc_0, f_dc_1, f_dc_2
            r, g, b = src['color']
            
            # Scale (log space usually)
            scale = [math.log(0.05), math.log(0.05), math.log(0.05)]
            
            # Rotation (Identity quaternion)
            rot = [1.0, 0.0, 0.0, 0.0]
            
            # Opacity (logit space, so 100 -> ~1.0)
            opacity = 10.0
            
            gaussians.append({
                'pos': pos,
                'color': [r, g, b],
                'scale': scale,
                'rot': rot,
                'opacity': opacity
            })
        return gaussians

    def _apply_motion(self, gaussians, t):
        """Apply sine wave motion based on position and time t (0..1)"""
        moved = []
        # Cycle: 2 PI
        phase = t * math.pi * 2.0
        
        for g in gaussians:
            # Copy
            new_g = g.copy()
            x, y, z = g['pos']
            
            # Wave motion: Y position affects phase
            offset = math.sin(phase + x * 2.0) * 0.1
            new_g['pos'] = [x, y + offset, z]
            
            # Color pulse
            # ...
            
            moved.append(new_g)
        return moved

    def _save_ply(self, gaussians, path):
        """
        Save standard 3DGS PLY format.
        Property list matches standard splat viewers.
        """
        with open(path, "wb") as f:
            # Header
            f.write(b"ply\n")
            f.write(b"format binary_little_endian 1.0\n")
            f.write(f"element vertex {len(gaussians)}\n".encode('utf-8'))
            f.write(b"property float x\n")
            f.write(b"property float y\n")
            f.write(b"property float z\n")
            f.write(b"property float nx\n")
            f.write(b"property float ny\n")
            f.write(b"property float nz\n")
            f.write(b"property float f_dc_0\n")
            f.write(b"property float f_dc_1\n")
            f.write(b"property float f_dc_2\n")
            f.write(b"property float scale_0\n")
            f.write(b"property float scale_1\n")
            f.write(b"property float scale_2\n")
            f.write(b"property float rot_0\n")
            f.write(b"property float rot_1\n")
            f.write(b"property float rot_2\n")
            f.write(b"property float rot_3\n")
            f.write(b"property float opacity\n")
            f.write(b"end_header\n")
            
            for g in gaussians:
                # Pack data
                # x, y, z, nx, ny, nz
                f.write(struct.pack("<fff", *g['pos']))
                f.write(struct.pack("<fff", 0, 0, 0)) # Normals (unused)
                
                # f_dc (Color)
                # SH 0 band conversion: (RGB - 0.5) / 0.282...
                # Simple RGB for viewer compat:
                # Viewers often expect SH. 
                # f_dc = (color - 0.5) / 0.28209479177387814
                C0 = 0.28209479177387814
                r = (g['color'][0] / 255.0 - 0.5) / C0
                g_c = (g['color'][1] / 255.0 - 0.5) / C0
                b = (g['color'][2] / 255.0 - 0.5) / C0
                f.write(struct.pack("<fff", r, g_c, b))
                
                # scale
                f.write(struct.pack("<fff", *g['scale']))
                
                # rot
                f.write(struct.pack("<ffff", *g['rot']))
                
                # opacity
                f.write(struct.pack("<f", g['opacity']))
