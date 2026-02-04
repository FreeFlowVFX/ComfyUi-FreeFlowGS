import os
from pathlib import Path

class FreeFlow_MultiCamLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "C:/Projects/MyCapture"}),
            },
        }

    RETURN_TYPES = ("MULTICAM_DICT",)
    RETURN_NAMES = ("multicam_feed",)
    FUNCTION = "load_cameras"
    CATEGORY = "FreeFlow"

    def load_cameras(self, directory_path):
        root = Path(directory_path)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # Find camera subfolders (assume alphanumeric sort acts as ID)
        # Looking for folders that might contain images. 
        # Heuristic: any folder that is not named 'sparse' or 'database'
        # Or stricter: folders starting with 'cam' or just all folders.
        # User prompt example: "cam01/, cam02/"
        
        # Strategy: Scan ALL subdirectories.
        # Any subdirectory containing images is treated as a camera.
        # We exclude specific known non-camera folders (outputs, system folders).
        
        ignored_folders = {
            "sparse", "dense", "database", "colmap_out", "output", "__macosx", 
            "checkpoints", "logs", "config", "freeflow_colmap", "images"
        }
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff", ".bmp", ".webp"}
        multicam_data = {}
        
        # 1. Iterate over all items in root
        all_subdirs = sorted([f for f in root.iterdir() if f.is_dir()])
        
        for folder in all_subdirs:
            if folder.name.lower() in ignored_folders:
                continue
                
            # Check content
            images = sorted([img for img in folder.iterdir() 
                           if img.is_file() and img.suffix.lower() in valid_extensions])
            
            if images:
                # Valid Camera found
                multicam_data[folder.name] = [str(img.absolute()) for img in images]
                # print(f"DEBUG: Found camera {folder.name} with {len(images)} frames.")
            else:
                # print(f"DEBUG: Skipping {folder.name} (No images).")
                pass

        if not multicam_data:
             print(f"DEBUG: No valid camera folders found in {root}.")
             raise ValueError(f"No camera subfolders with images found in {directory_path}")

        # Validate consistency (Soft check)
        max_frames = 0
        reference_frame_count = -1
        for cam, frames in multicam_data.items():
            count = len(frames)
            max_frames = max(max_frames, count)
            
            if reference_frame_count == -1:
                reference_frame_count = count
            elif count != reference_frame_count:
                print(f"⚠️ [FreeFlow] Warning: Frame count mismatch! {cam}: {count} (Ref: {reference_frame_count})")

        print(f"Loaded {len(multicam_data)} cameras. Max sequence length: {max_frames}")
        return (multicam_data,)
