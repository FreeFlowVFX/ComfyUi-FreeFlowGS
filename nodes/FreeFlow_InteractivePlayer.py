class FreeFlow_InteractivePlayer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                 "camera_far": ("INT", {"default": 5000, "min": 100, "max": 100000, "step": 100}),
                 "depth_sorting": ("BOOLEAN", {"default": True}),
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

    def play(self, camera_far, depth_sorting, ply_sequence_path=None, ply_sequence_list=None):
        target_path = ""
        
        # Prefer list if available (from Smoother/Loader)
        if ply_sequence_list:
             from pathlib import Path
             first_file = Path(ply_sequence_list[0])
             target_path = first_file.parent
        elif ply_sequence_path:
             from pathlib import Path
             target_path = Path(ply_sequence_path)
             
        if not target_path or not target_path.exists():
            return {"ui": {"player_data": []}}

        # Scan for PLY files
        files = sorted([f.name for f in target_path.glob("*.ply")])
        
        # Return data expected by JS
        return {"ui": {"player_data": [{
            "output_dir": str(target_path),
            "files": files,
            "fps": 24,
            "camera_far": camera_far,
            "depth_sorting": depth_sorting
        }]}}
