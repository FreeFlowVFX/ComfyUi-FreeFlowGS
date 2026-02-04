import os
from pathlib import Path
from ..utils import FreeFlowUtils

class FreeFlow_PLYSequenceLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Absolute path to folder containing sequential .ply files. No need to select a file, just the folder."}),
                "frame_range": ("STRING", {"default": "*", "multiline": False, "placeholder": "e.g. 0-100, 0,10,20, or * (All)"}),
            }
        }

    RETURN_TYPES = ("PLY_SEQUENCE",)
    RETURN_NAMES = ("ply_sequence",)
    FUNCTION = "load_sequence"
    CATEGORY = "FreeFlow"

    def load_sequence(self, directory_path, frame_range="*"):
        path = Path(directory_path.strip())
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        # Scan for PLYs
        ply_files = list(path.glob("*.ply"))
        
        # Helper to extract frame number (last number in filename)
        import re
        def extract_frame_num(p):
            # Find all sequences of digits
            nums = re.findall(r'\d+', p.stem)
            if nums:
                return int(nums[-1]) # Assume last number is frame number
            return -1 # No number found

        # Tuple list: (path, frame_number)
        files_with_nums = []
        for p in ply_files:
            num = extract_frame_num(p)
            files_with_nums.append({'path': p, 'num': num})

        # Sort by number, or name if no number
        # Sort by number, or name if no number
        # Use tuple (has_num, num, name) to ensure consistent types for comparison
        # 0 = has number, 1 = no number (so numbered files come first?)
        # Actually simpler: (num, name) but ensure num is always int.
        # If num is -1, it deals with it.
        # TypeError: '<' not supported between instances of 'str' and 'int' happened because 
        # lambda returned either int OR str. Python 3 cannot compare int < str.
        
        files_with_nums.sort(key=lambda x: (x['num'], x['path'].name))

        if not files_with_nums:
            raise ValueError(f"No .ply files found in {path}")
            
        # --- Frame Filtering Logic ---
        final_files = []
        fr = frame_range.strip() #.replace(" ", "") # Remove spaces? "99 - 140" -> "99-140"
        
        if fr == "*" or fr.lower() == "all" or not fr:
             final_files = [x['path'] for x in files_with_nums]
        else:
            # Parse allowed numbers
            allowed_frames = set()
            try:
                parts = fr.split(",")
                for p in parts:
                    p = p.strip()
                    if "-" in p:
                        # Handle range
                        r_parts = p.split("-")
                        if len(r_parts) == 2:
                            start = int(r_parts[0])
                            end = int(r_parts[1])
                            # User likely expects Inclusive for Frame Numbers (e.g. 99-100 includes 100)
                            # Python range is exclusive. Let's make it INCLUSIVE for frame numbers.
                            allowed_frames.update(range(start, end + 1))
                    else:
                        # Single number
                        allowed_frames.add(int(p))
            except ValueError:
                FreeFlowUtils.log(f"Invalid frame range format '{frame_range}'. Loading ALL.", "WARN")
                final_files = [x['path'] for x in files_with_nums]
            
            if not final_files:
                # Filter based on extracted numbers
                # Only include files that have a valid number AND are in the allowed set
                
                filtered = []
                for item in files_with_nums:
                    if item['num'] != -1 and item['num'] in allowed_frames:
                        filtered.append(item['path'])
                    # DEBUG: Uncomment to see what's happening
                    # else:
                    #     FreeFlowUtils.log(f"Skipped {item['path'].name}: num={item['num']}, allowed={allowed_frames}")
                
                FreeFlowUtils.log(f"Frame filter: allowed={sorted(list(allowed_frames))[:10]}... matched {len(filtered)}/{len(files_with_nums)} files")
                final_files = filtered

        if not final_files:
             FreeFlowUtils.log(f"Frame range '{frame_range}' matched 0 files in {path.name}. Reverting to ALL.", "WARN")
             final_files = [x['path'] for x in files_with_nums]

        FreeFlowUtils.log(f"Loaded {len(final_files)} frames from {path.name} (Range: {frame_range})")
        
        # Return list of strings
        return ([str(p) for p in final_files],)
