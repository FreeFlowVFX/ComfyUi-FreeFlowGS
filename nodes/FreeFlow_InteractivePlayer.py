class FreeFlow_InteractivePlayer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "camera_far": ("INT", {"default": 5000, "min": 100, "max": 100000, "step": 100}),
             },
            "optional": {
                "ply_sequence_path": ("STRING", {"forceInput": True}),
                "ply_sequence_list": ("PLY_SEQUENCE",),
            }
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True 
    FUNCTION = "play"
    CATEGORY = "FreeFlow"

    def play(self, camera_far, ply_sequence_path=None, ply_sequence_list=None):
        target_path = ""
        files = []
        
        # Prefer list if available (from Smoother/Loader) - USE IT DIRECTLY!
        if ply_sequence_list and len(ply_sequence_list) > 0:
            from pathlib import Path
            # Use the filtered list directly - DO NOT re-glob!
            first_file = Path(ply_sequence_list[0])
            target_path = first_file.parent
            # Extract just filenames from full paths
            files = [Path(f).name for f in ply_sequence_list]
        elif ply_sequence_path:
            from pathlib import Path
            target_path = Path(ply_sequence_path)
            if target_path.exists():
                # Only re-glob if using the path input (no filter applied)
                files = sorted([f.name for f in target_path.glob("*.ply")])
              
        if not target_path or not files:
            return {"ui": {"player_data": []}}

        # Return data expected by JS
        return {"ui": {"player_data": [{
            "output_dir": str(target_path),
            "files": files,
            "fps": 24,
            "camera_far": camera_far
        }]}}
