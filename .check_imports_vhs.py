import sys
import os

# Add the repo root to sys.path
repo_root = "/Volumes/Models/Google2Comfy/comfyui-videohelpersuite"
if repo_root not in sys.path:
    sys.path.append(repo_root)

try:
    from videohelpersuite.nodes import (
        VideoCombine,
        get_video_formats,
        apply_format_widgets,
        tensor_to_bytes,
        tensor_to_shorts,
        to_pingpong,
        ffmpeg_process,
        gifski_process,
    )
    from videohelpersuite.utils import (
        imageOrLatent,
        floatOrInt,
        ContainsAll,
        ffmpeg_path,
        merge_filter_args,
        requeue_workflow,
    )
    from videohelpersuite.logger import logger
    print("Imports successful")
except ImportError as e:
    print(f"Import failed: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
