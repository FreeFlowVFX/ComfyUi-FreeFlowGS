import importlib.util
import subprocess
import sys
import os

def check_and_install_requirements():
    try:
        # Check for critical dependencies
        import google.genai
        import loguru
    except ImportError:
        print("ComfyUI-TimNodes: Dependencies missing. Installing from requirements.txt...")
        req_path = os.path.join(os.path.dirname(__file__), "requirements.txt")
        if os.path.exists(req_path):
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_path])
                print("ComfyUI-TimNodes: Dependencies installed. Restarting ComfyUI might be required if it still fails.")
            except Exception as e:
                print(f"ComfyUI-TimNodes: Failed to auto-install requirements: {e}")

check_and_install_requirements()

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

WEB_DIRECTORY = "js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
