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
                "frame_range": ("STRING", {"default": "*", "multiline": False, "placeholder": "e.g. 0-100, 0,10,20, or * (All)"}),
            },
        }

    RETURN_TYPES = ("MULTICAM_DICT",)
    RETURN_NAMES = ("multicam_feed",)
    FUNCTION = "load_cameras"
    CATEGORY = "FreeFlow"

    def load_cameras(self, directory_path, frame_range="*"):
        import re
        from ..utils import FreeFlowUtils # Import inside function to avoid circular deps if any

        root = Path(directory_path)
        if not root.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")

        # --- Helper for parsing ranges (Reused from PLY Loader) ---
        def parse_frame_range(fr_str):
            fr = fr_str.strip()
            if fr == "*" or fr.lower() == "all" or not fr:
                return None # None means ALL
            
            allowed = set()
            try:
                parts = fr.split(",")
                for p in parts:
                    p = p.strip()
                    if "-" in p:
                        r_parts = p.split("-")
                        if len(r_parts) == 2:
                            start = int(r_parts[0])
                            end = int(r_parts[1])
                            allowed.update(range(start, end + 1))
                    else:
                        allowed.add(int(p))
                return allowed
            except ValueError:
                FreeFlowUtils.log(f"Invalid frame range '{fr_str}'. Loading ALL.", "WARN")
                return None

        allowed_frames = parse_frame_range(frame_range)
        
        def extract_frame_num(p):
            nums = re.findall(r'\d+', p.stem)
            return int(nums[-1]) if nums else -1

        # --- Scan Folders ---
        ignored_folders = {
            "sparse", "dense", "database", "colmap_out", "output", "__macosx", 
            "checkpoints", "logs", "config", "freeflow_colmap", "images"
        }
        
        valid_extensions = {".jpg", ".jpeg", ".png", ".exr", ".tif", ".tiff", ".bmp", ".webp"}
        multicam_data = {}
        
        all_subdirs = sorted([f for f in root.iterdir() if f.is_dir()])
        
        for folder in all_subdirs:
            if folder.name.lower() in ignored_folders:
                continue
                
            # Check content
            images = sorted([img for img in folder.iterdir() 
                           if img.is_file() and img.suffix.lower() in valid_extensions])
            
            if images:
                # Apply Frame Filter
                if allowed_frames is not None:
                    filtered_images = []
                    for img in images:
                        num = extract_frame_num(img)
                        if num != -1 and num in allowed_frames:
                            filtered_images.append(img)
                    images = filtered_images
                
                if images:
                    multicam_data[folder.name] = [str(img.absolute()) for img in images]
        
        if not multicam_data:
             print(f"DEBUG: No valid camera folders found in {root} (Range: {frame_range}).")
             # Don't crash, just warn? Or crash if critical? 
             # For logic flow, crashing is safer if essential input.
             raise ValueError(f"No matching images found in {directory_path} with range '{frame_range}'")

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
