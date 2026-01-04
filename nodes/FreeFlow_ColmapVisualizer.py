"""
FreeFlow_ColmapVisualizer - 3D Visualization Node
Parses COLMAP binary files and provides data for Three.js frontend visualization.
"""

import struct
import json
import os
from pathlib import Path
from ..utils import FreeFlowUtils


class ColmapBinaryParser:
    """
    Parse COLMAP binary files (cameras.bin, images.bin, points3D.bin).
    Supports COLMAP binary format version 1 and 2.
    """
    
    @staticmethod
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_char="<"):
        """Read and unpack bytes from a binary file."""
        data = fid.read(num_bytes)
        return struct.unpack(endian_char + format_char_sequence, data)

    @staticmethod
    def parse_cameras_bin(path):
        """
        Parse cameras.bin file.
        Returns dict: {camera_id: {model, width, height, params}}
        """
        cameras = {}
        
        with open(path, "rb") as fid:
            num_cameras = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
            
            for _ in range(num_cameras):
                camera_id = ColmapBinaryParser.read_next_bytes(fid, 4, "I")[0]
                model_id = ColmapBinaryParser.read_next_bytes(fid, 4, "i")[0]
                width = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                height = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                
                # Camera model determines number of params
                CAMERA_MODEL_PARAMS = {
                    0: 3,   # SIMPLE_PINHOLE
                    1: 4,   # PINHOLE
                    2: 4,   # SIMPLE_RADIAL
                    3: 5,   # RADIAL
                    4: 8,   # OPENCV
                    5: 12,  # OPENCV_FISHEYE
                    6: 12,  # FULL_OPENCV
                }
                
                num_params = CAMERA_MODEL_PARAMS.get(model_id, 4)
                params = ColmapBinaryParser.read_next_bytes(fid, 8 * num_params, "d" * num_params)
                
                cameras[camera_id] = {
                    "model_id": model_id,
                    "width": width,
                    "height": height,
                    "params": list(params)
                }
        
        return cameras

    @staticmethod
    def parse_images_bin(path):
        """
        Parse images.bin file.
        Returns dict: {image_id: {qvec, tvec, camera_id, name, point2D_ids}}
        """
        images = {}
        
        with open(path, "rb") as fid:
            num_images = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
            
            for _ in range(num_images):
                image_id = ColmapBinaryParser.read_next_bytes(fid, 4, "I")[0]
                qvec = ColmapBinaryParser.read_next_bytes(fid, 32, "dddd")
                tvec = ColmapBinaryParser.read_next_bytes(fid, 24, "ddd")
                camera_id = ColmapBinaryParser.read_next_bytes(fid, 4, "I")[0]
                
                # Read image name (null-terminated string)
                name = ""
                while True:
                    char = fid.read(1)
                    if char == b"\x00":
                        break
                    name += char.decode("utf-8")
                
                # Read 2D points
                num_points2D = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                
                # Skip 2D point data (x, y, point3D_id for each)
                fid.read(num_points2D * 24)
                
                images[image_id] = {
                    "qvec": list(qvec),
                    "tvec": list(tvec),
                    "camera_id": camera_id,
                    "name": name
                }
        
        return images

    @staticmethod
    def parse_points3D_bin(path, max_points=100000):
        """
        Parse points3D.bin file.
        Returns dict with points array and statistics.
        """
        points = []
        errors = []
        
        with open(path, "rb") as fid:
            num_points = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
            
            for i in range(min(num_points, max_points)):
                point3D_id = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                xyz = ColmapBinaryParser.read_next_bytes(fid, 24, "ddd")
                rgb = ColmapBinaryParser.read_next_bytes(fid, 3, "BBB")
                error = ColmapBinaryParser.read_next_bytes(fid, 8, "d")[0]
                
                # Number of observations (tracks)
                track_length = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                
                # Skip track data
                fid.read(track_length * 8)
                
                points.append({
                    "id": point3D_id,
                    "xyz": list(xyz),
                    "rgb": list(rgb),
                    "error": error
                })
                errors.append(error)
            
            # Skip remaining points if over limit
            if num_points > max_points:
                FreeFlowUtils.log(f"Truncated point cloud from {num_points} to {max_points} for visualization", "WARN")
        
        return {
            "total_count": num_points,
            "displayed_count": len(points),
            "points": points,
            "mean_error": sum(errors) / len(errors) if errors else 0,
            "max_error": max(errors) if errors else 0,
            "min_error": min(errors) if errors else 0
        }


class FreeFlow_ColmapVisualizer:
    """
    Visualizer node that parses COLMAP data and sends it to Three.js frontend.
    Replicates COLMAP Desktop GUI 3D view inside ComfyUI.
    """
    
    def __init__(self):
        self._cached_data = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "colmap_data": ("COLMAP_DATA",),
            },
            "optional": {
                "colorize_by_error": ("BOOLEAN", {"default": False}),
                "max_points": ("INT", {"default": 100000, "min": 1000, "max": 500000}),
                "frustum_scale": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("visualization_data",)
    FUNCTION = "visualize"
    CATEGORY = "FreeFlow"
    OUTPUT_NODE = True

    def visualize(self, colmap_data, colorize_by_error=False, max_points=100000, frustum_scale=0.1):
        """
        Parse COLMAP binary files and prepare visualization data.
        """
        workspace = Path(colmap_data)
        sparse_dir = workspace / "sparse" / "0"
        
        if not sparse_dir.exists():
            raise FileNotFoundError(f"COLMAP sparse reconstruction not found: {sparse_dir}")
        
        cameras_file = sparse_dir / "cameras.bin"
        images_file = sparse_dir / "images.bin"
        points_file = sparse_dir / "points3D.bin"
        
        FreeFlowUtils.log("Parsing COLMAP binary files...")
        
        # Parse all files
        cameras = {}
        if cameras_file.exists():
            cameras = ColmapBinaryParser.parse_cameras_bin(cameras_file)
            FreeFlowUtils.log(f"  Cameras: {len(cameras)}")
        
        images = {}
        if images_file.exists():
            images = ColmapBinaryParser.parse_images_bin(images_file)
            FreeFlowUtils.log(f"  Images: {len(images)}")
        
        points_data = {"total_count": 0, "displayed_count": 0, "points": []}
        if points_file.exists():
            points_data = ColmapBinaryParser.parse_points3D_bin(points_file, max_points)
            FreeFlowUtils.log(f"  Points: {points_data['displayed_count']}/{points_data['total_count']}")
            FreeFlowUtils.log(f"  Mean Error: {points_data['mean_error']:.4f}")
        
        # Build visualization payload
        viz_data = {
            "type": "colmap_visualization",
            "stats": {
                "num_cameras": len(cameras),
                "num_images": len(images),
                "num_points": points_data["total_count"],
                "displayed_points": points_data["displayed_count"],
                "mean_error": points_data["mean_error"],
                "max_error": points_data["max_error"],
            },
            "settings": {
                "colorize_by_error": colorize_by_error,
                "frustum_scale": frustum_scale,
                "error_range": [points_data["min_error"], points_data["max_error"]]
            },
            "cameras": cameras,
            "images": images,
            # Convert points to compact format for frontend
            "points": self._compact_points(points_data["points"], colorize_by_error, 
                                           points_data["min_error"], points_data["max_error"])
        }
        
        # Store for UI message
        self._cached_data = viz_data
        
        FreeFlowUtils.log("✅ Visualization data prepared successfully.")
        
        # Return JSON string for potential downstream use
        return {"ui": {"colmap_viz": [viz_data]}, "result": (json.dumps(viz_data["stats"]),)}

    def _compact_points(self, points, colorize_by_error, min_error, max_error):
        """
        Convert points to compact arrays for efficient Three.js rendering.
        Returns flat arrays: positions, colors, errors
        """
        positions = []
        colors = []
        
        error_range = max_error - min_error if max_error > min_error else 1.0
        
        for p in points:
            # Position
            positions.extend(p["xyz"])
            
            if colorize_by_error:
                # Map error to color: green (low) → red (high)
                t = (p["error"] - min_error) / error_range
                r = int(255 * t)
                g = int(255 * (1 - t))
                b = 0
                colors.extend([r, g, b])
            else:
                # Original RGB
                colors.extend(p["rgb"])
        
        return {
            "positions": positions,
            "colors": colors,
            "count": len(points)
        }
