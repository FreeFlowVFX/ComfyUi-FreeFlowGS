import traceback
import os
from aiohttp import web
from server import PromptServer

# --- Custom Server Routes for External File Serving ---
routes = PromptServer.instance.routes

@routes.get("/freeflow/view")
async def freeflow_view_image(request):
    if "filename" not in request.rel_url.query:
        print("DEBUG: /freeflow/view - No filename param")
        return web.Response(status=404)
    
    filename = request.rel_url.query["filename"]
    if not os.path.exists(filename):
        print(f"DEBUG: /freeflow/view - File not found: {filename}")
        return web.Response(status=404)

    # Security: We might want to restrict this, but for this dev tool, 
    # we allow reading the provided absolute path.
    # Serve the file content
    # print(f"DEBUG: Serving {filename}") # Verbose, maybe comment out if too spammy
    with open(filename, "rb") as f:
        data = f.read()
    
    # Determine basic content type
    ctype = "application/octet-stream"
    if filename.lower().endswith(".jpg") or filename.lower().endswith(".jpeg"):
        ctype = "image/jpeg"
    elif filename.lower().endswith(".png"):
        ctype = "image/png"
    elif filename.lower().endswith(".webp"):
        ctype = "image/webp"

    return web.Response(body=data, content_type=ctype)


try:
    from .utils import FreeFlowUtils
    
    # --- OS & Dependency Checks ---
    print("\n" + "="*50)
    print("🌊 Initializing FreeFlow 4D Splats")
    print(f"   • OS Detected: {FreeFlowUtils.get_os()}")
    
    # STARTUP VERSION CHECK (Threaded)
    FreeFlowUtils.check_updates()

    # AUTO-INSTALL CHECK
    try:
        FreeFlowUtils.check_and_install()
    except Exception as e:
        FreeFlowUtils.log(f"Auto-Install Failed: {e}", "ERROR")
        traceback.print_exc()

    brush_path = FreeFlowUtils.get_binary_path("brush")
    colmap_path = FreeFlowUtils.get_binary_path("colmap")

    if not brush_path:
        FreeFlowUtils.log("Brush binary MISSING after auto-install attempt.", "ERROR")
    else:
        FreeFlowUtils.ensure_executable(brush_path)
        print(f"   • Brush: Found ({brush_path.name})")

    if not colmap_path:
        FreeFlowUtils.log("COLMAP NOT FOUND.", "WARN")
        if FreeFlowUtils.get_os() == "Darwin":
            print("   >>> ACTION: Run 'brew install colmap' in Terminal (Auto-install not supported on Mac due to dylib dependencies)")
        elif FreeFlowUtils.get_os() == "Linux":
            print("   >>> ACTION: Linux auto-install is local to this extension. If needed set FREEFLOW_COLMAP_LINUX_URL or FREEFLOW_COLMAP_LINUX_LOCAL_ARCHIVE.")
            print("   >>>         Optional strategy override: FREEFLOW_COLMAP_INSTALL_STRATEGY=binary-first|binary-only|mamba-only")
    else:
        print(f"   • COLMAP: Found ({colmap_path.name})")

    # Check Nerfstudio status (Splatfacto backend)
    try:
        from .nodes.modules.nerfstudio_env import NerfstudioEnvironment
        if NerfstudioEnvironment.is_installed():
            ns_version = NerfstudioEnvironment.get_version()
            cuda_ver = NerfstudioEnvironment._detect_cuda_version()
            cuda_str = f", CUDA {cuda_ver}" if cuda_ver else ", CPU only (slow)"
            print(f"   • Nerfstudio: Ready (v{ns_version}{cuda_str})")
        else:
            print("   • Nerfstudio: Not installed (Splatfacto engine unavailable)")
    except Exception as e:
        print(f"   • Nerfstudio: Error checking status: {e}")

    print("="*50 + "\n")

    # --- Node Imports ---
    from .nodes.FreeFlow_MultiCamLoader import FreeFlow_MultiCamLoader
    from .nodes.FreeFlow_SmartGridMonitor import FreeFlow_SmartGridMonitor
    from .nodes.FreeFlow_ColmapAnchor import FreeFlow_ColmapAnchor
    from .nodes.FreeFlow_ColmapVisualizer import FreeFlow_ColmapVisualizer
    from .nodes.FreeFlow_3DVisualizer import FreeFlow_3DVisualizer
    from .nodes.FreeFlow_AdaptiveEngine import FreeFlow_AdaptiveEngine
    from .nodes.FreeFlow_InteractivePlayer import FreeFlow_InteractivePlayer
    from .nodes.FreeFlow_PLYSequenceLoader import FreeFlow_PLYSequenceLoader
    from .nodes.FreeFlow_PostProcessSmoother import FreeFlow_PostProcessSmoother
    from .nodes.FreeFlow_GS_Engine import FreeFlow_GS_Engine
    from .nodes.FreeFlow_FlameTracker import FreeFlow_FlameTracker
    from .nodes.FreeFlow_MeshLoader import FreeFlow_MeshLoader

    NODE_CLASS_MAPPINGS = {
        "FreeFlow_MultiCamLoader": FreeFlow_MultiCamLoader,
        "FreeFlow_SmartGridMonitor": FreeFlow_SmartGridMonitor,
        "FreeFlow_ColmapAnchor": FreeFlow_ColmapAnchor,
        "FreeFlow_ColmapVisualizer": FreeFlow_ColmapVisualizer,
        "FreeFlow_3DVisualizer": FreeFlow_3DVisualizer,
        "FreeFlow_AdaptiveEngine": FreeFlow_AdaptiveEngine,
        "FreeFlow_InteractivePlayer": FreeFlow_InteractivePlayer,
        "FreeFlow_PLYSequenceLoader": FreeFlow_PLYSequenceLoader,
        "FreeFlow_PostProcessSmoother": FreeFlow_PostProcessSmoother,
        "FreeFlow_GS_Engine": FreeFlow_GS_Engine,
        "FreeFlow_FlameTracker": FreeFlow_FlameTracker,
        "FreeFlow_MeshLoader": FreeFlow_MeshLoader
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "FreeFlow_MultiCamLoader": "FreeFlow Multi-Camera Loader",
        "FreeFlow_SmartGridMonitor": "FreeFlow Smart Grid Monitor",
        "FreeFlow_ColmapAnchor": "FreeFlow COLMAP Anchor",
        "FreeFlow_ColmapVisualizer": "FreeFlow COLMAP 3D Visualizer (Legacy)",
        "FreeFlow_3DVisualizer": "FreeFlow 3D Visualizer",
        "FreeFlow_AdaptiveEngine": "FreeFlow 4D Adaptive Engine",
        "FreeFlow_InteractivePlayer": "FreeFlow 4D Player",
        "FreeFlow_PLYSequenceLoader": "FreeFlow PLY Sequence Loader",
        "FreeFlow_PostProcessSmoother": "FreeFlow Post-Process Smoother (Savitzky-Golay)",
        "FreeFlow_GS_Engine": "FreeFlow GS Engine (4D)",
        "FreeFlow_FlameTracker": "FreeFlow 3D/Face Tracker (Auto-Rig)",
        "FreeFlow_MeshLoader": "FreeFlow External Mesh Loader (Metahuman)"
    }

except Exception as e:
    print("\n\033[91m🌊 FreeFlowGS IMPORT FAILED:\033[0m")
    traceback.print_exc()
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

WEB_DIRECTORY = "./js"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
