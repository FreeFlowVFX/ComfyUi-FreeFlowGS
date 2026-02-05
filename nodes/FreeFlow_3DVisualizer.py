"""
FreeFlow_3DVisualizer - Universal 3D Viewer
Upgraded visualizer that supports both COLMAP Sparse Clouds and Guidance Meshes.
"""

import struct
import json
import os
import numpy as np
from pathlib import Path
from ..utils import FreeFlowUtils

# Reuse parser logic (could be moved to utils, but keeping self-contained for safety)
class ColmapBinaryParser:
    @staticmethod
    def read_next_bytes(fid, num_bytes, format_char_sequence, endian_char="<"):
        data = fid.read(num_bytes)
        return struct.unpack(endian_char + format_char_sequence, data)

    @staticmethod
    def parse_cameras_bin(path):
        cameras = {}
        with open(path, "rb") as fid:
            num_cameras = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_cameras):
                camera_id = ColmapBinaryParser.read_next_bytes(fid, 4, "I")[0]
                model_id = ColmapBinaryParser.read_next_bytes(fid, 4, "i")[0]
                width = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                height = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                # Approximation: Assume max 12 params (read safe chunk)
                # Actually need robustness.
                CAMERA_MODEL_PARAMS = {0: 3, 1: 4, 2: 4, 3: 5, 4: 8, 5: 12, 6: 12}
                num_params = CAMERA_MODEL_PARAMS.get(model_id, 4)
                params = ColmapBinaryParser.read_next_bytes(fid, 8 * num_params, "d" * num_params)
                cameras[camera_id] = {"model_id": model_id, "width": width, "height": height, "params": list(params)}
        return cameras

    @staticmethod
    def parse_images_bin(path):
        images = {}
        with open(path, "rb") as fid:
            num_images = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
            for _ in range(num_images):
                image_id = ColmapBinaryParser.read_next_bytes(fid, 4, "I")[0]
                qvec = ColmapBinaryParser.read_next_bytes(fid, 32, "dddd")
                tvec = ColmapBinaryParser.read_next_bytes(fid, 24, "ddd")
                camera_id = ColmapBinaryParser.read_next_bytes(fid, 4, "I")[0]
                name = ""
                while True:
                    char = fid.read(1)
                    if char == b"\x00": break
                    name += char.decode("utf-8")
                num_points2D = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                fid.read(num_points2D * 24) # Skip 2D
                images[image_id] = {"qvec": list(qvec), "tvec": list(tvec), "camera_id": camera_id, "name": name}
        return images

    @staticmethod
    def parse_points3D_bin(path, max_points=100000):
        points = []
        errors = []
        with open(path, "rb") as fid:
            num_points = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
            for i in range(min(num_points, max_points)):
                point3D_id = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                xyz = ColmapBinaryParser.read_next_bytes(fid, 24, "ddd")
                rgb = ColmapBinaryParser.read_next_bytes(fid, 3, "BBB")
                error = ColmapBinaryParser.read_next_bytes(fid, 8, "d")[0]
                track_length = ColmapBinaryParser.read_next_bytes(fid, 8, "Q")[0]
                fid.read(track_length * 8)
                points.append({"id": point3D_id, "xyz": list(xyz), "rgb": list(rgb), "error": error})
                errors.append(error)
            if num_points > max_points:
                FreeFlowUtils.log(f"Truncated point cloud to {max_points}", "WARN")
        return {
            "total_count": num_points, "displayed_count": len(points), "points": points,
            "mean_error": sum(errors)/len(errors) if errors else 0,
            "max_error": max(errors) if errors else 0,
            "min_error": min(errors) if errors else 0
        }


class FreeFlow_3DVisualizer:
    """
    Universal 3D Visualizer.
    Can display COLMAP Scene, Guidance Mesh, or Both.
    """
    
    def __init__(self):
        self._cached_data = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {},
            "optional": {
                "colmap_data": ("COLMAP_DATA",),
                "guidance_mesh": ("GUIDANCE_MESH_SEQUENCE",),
                "colorize_by_error": ("BOOLEAN", {"default": False}),
                "max_points": ("INT", {"default": 100000, "min": 1000, "max": 500000}),
                "frustum_scale": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("visualization_payload",)
    FUNCTION = "visualize"
    CATEGORY = "FreeFlow"
    OUTPUT_NODE = True

    def visualize(self,  colorize_by_error=False, max_points=100000, frustum_scale=0.1, colmap_data=None, guidance_mesh=None):
        
        viz_data = {
            "type": "colmap_visualization", # Keeping type ID for compat, or update to "freeflow_3d_viz"
            "stats": {},
            "settings": {
                "colorize_by_error": colorize_by_error,
                "frustum_scale": frustum_scale,
                "error_range": [0, 0]
            }
        }

        # 1. Process COLMAP (If provided)
        if colmap_data:
            try:
                workspace = Path(colmap_data)
                sparse_dir = workspace / "sparse" / "0"
                if sparse_dir.exists():
                    cameras_file = sparse_dir / "cameras.bin"
                    images_file = sparse_dir / "images.bin"
                    points_file = sparse_dir / "points3D.bin"
                    
                    if cameras_file.exists(): viz_data["cameras"] = ColmapBinaryParser.parse_cameras_bin(cameras_file)
                    if images_file.exists(): viz_data["images"] = ColmapBinaryParser.parse_images_bin(images_file)
                    if points_file.exists():
                        points_data = ColmapBinaryParser.parse_points3D_bin(points_file, max_points)
                        viz_data["points"] = self._compact_points(points_data["points"], colorize_by_error, points_data["min_error"], points_data["max_error"])
                        viz_data["settings"]["error_range"] = [points_data["min_error"], points_data["max_error"]]
                        viz_data["stats"].update({
                            "num_points": points_data["total_count"],
                            "mean_error": points_data["mean_error"]
                        })
                    
                    if "images" in viz_data:
                        # Extract Frame Numbers for Timeline (Sorted by Filename)
                        # Used by Frontend to show "195 (1/26)" instead of "0 (1/26)"
                        try:
                            sorted_imgs = sorted(viz_data["images"].values(), key=lambda x: x["name"])
                            import re
                            frame_nums = []
                            for img in sorted_imgs:
                                 nums = re.findall(r'\d+', img["name"])
                                 if nums: frame_nums.append(int(nums[-1]))
                                 else: frame_nums.append(0)
                            viz_data["frame_numbers"] = frame_nums
                        except Exception as e:
                            FreeFlowUtils.log(f"Visualizer: Failed to extract frame numbers: {e}", "WARN")

            except Exception as e:
                FreeFlowUtils.log(f"Visualizer Error (COLMAP): {e}", "ERROR")

        # 2. Process Mesh (If provided)
        if guidance_mesh:
            try:
                # Guidance Mesh is a List of Dicts [{'vertices': np.arr, 'faces': ...}, ...]
                # Verification is usually done on the ANCHOR frame (Frame 0).
                # Sending full sequence is too heavy (json limit ~100MB usually, but laggy).
                # Send Frame 0.
                
                # Send Full Mesh Sequence for Timeline
                if len(guidance_mesh) > 0:
                     mesh_seq_data = []
                     for frame_idx, frame_data in enumerate(guidance_mesh):
                         if frame_data and 'vertices' in frame_data:
                            # Optimize: Precision reduction? Or just raw floats.
                            # 478 verts * 3 floats is small.
                            mesh_seq_data.append(frame_data['vertices'].tolist())
                         else:
                            mesh_seq_data.append(None)
                     
                     viz_data["mesh_sequence"] = mesh_seq_data
                     viz_data["mesh_faces"] = guidance_mesh[0].get('faces', []) # Faces are static topology
                     
                     viz_data["stats"]["mesh_verts"] = len(guidance_mesh[0]['vertices']) if guidance_mesh[0] else 0
                     viz_data["stats"]["mesh_frames"] = len(guidance_mesh)
                     
                     # Extract proper frame numbers if provided by Tracker
                     mesh_frame_nums = []
                     has_nums = False
                     for m in guidance_mesh:
                         if m and 'frame_number' in m:
                             mesh_frame_nums.append(m['frame_number'])
                             has_nums = True
                         else:
                             mesh_frame_nums.append(0)
                    
                     if has_nums:
                         viz_data["frame_numbers"] = mesh_frame_nums
                         FreeFlowUtils.log(f"Visualizer: Synced Timeline with Tracker Metadata ({len(mesh_frame_nums)} frames).")
                        
            except Exception as e:
                FreeFlowUtils.log(f"Visualizer Error (Mesh): {e}", "ERROR")

        self._cached_data = viz_data
        return {"ui": {"freeflow_3d_viz": [viz_data]}, "result": (json.dumps(viz_data["stats"]),)} # Updated key to avoid conflict with legacy

    def _compact_points(self, points, colorize_by_error, min_error, max_error):
        positions = []
        colors = []
        error_range = max_error - min_error if max_error > min_error else 1.0
        
        for p in points:
            positions.extend(p["xyz"])
            if colorize_by_error:
                t = (p["error"] - min_error) / error_range
                r = int(255 * t); g = int(255 * (1 - t)); b = 0
                colors.extend([r, g, b])
            else:
                colors.extend(p["rgb"])
        return {"positions": positions, "colors": colors, "count": len(points)}
