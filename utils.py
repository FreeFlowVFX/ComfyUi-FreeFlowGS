
# --- FreeFlow Utility Class ---
import platform
import os
import sys
import shutil
import urllib.request
import zipfile
import tarfile
from pathlib import Path
import folder_paths

# --- CONSTANTS ---
# --- CONSTANTS ---
BRUSH_VERSION = "0.3.0"
BRUSH_REPO = "ArthurBrussee/brush"
COLMAP_REPO = "colmap/colmap"
GSPLAT_VERSION = "1.2.9"  # gsplat.js npm version
GSPLAT_CDN_URL = f"https://cdn.jsdelivr.net/npm/gsplat@{GSPLAT_VERSION}/dist/index.es.js"

# Using placeholder URLs - In production, use exact GitHub Release Asset URLs
BRUSH_URL_WIN = f"https://github.com/ArthurBrussee/brush/releases/download/v{BRUSH_VERSION}/brush-app-x86_64-pc-windows-msvc.zip"
BRUSH_URL_MAC = f"https://github.com/ArthurBrussee/brush/releases/download/v{BRUSH_VERSION}/brush-app-aarch64-apple-darwin.tar.xz"
BRUSH_URL_LINUX = f"https://github.com/ArthurBrussee/brush/releases/download/v{BRUSH_VERSION}/brush-app-x86_64-unknown-linux-gnu.tar.xz"

COLMAP_URL_WIN = "https://github.com/colmap/colmap/releases/download/3.13.0/colmap-x64-windows-cuda.zip"
# VOCAB TREE MIRRORS (Try in order) - FAISS VERSION FOR COLMAP 3.12+
VOCAB_TREE_MIRRORS = [
    "https://github.com/colmap/colmap/releases/download/3.11.1/vocab_tree_faiss_flickr100K_words32K.bin",
    "https://demuc.de/colmap/vocab_tree_faiss_flickr100K_words32K.bin", 
    "https://cvg-data.inf.ethz.ch/colmap/vocab_tree_faiss_flickr100K_words32K.bin",
]

class FreeFlowUtils:
    @staticmethod
    def get_os():
        """Returns 'Windows', 'Darwin' (macOS), or 'Linux'."""
        system = platform.system()
        # if system == "Darwin" and platform.machine() != "arm64":
        #    print("Warning: Brush Mac release is optimized for Apple Silicon (arm64).")
        return system

    @staticmethod
    def get_extension_dir():
        """Returns the base directory of this extension."""
        return Path(__file__).parent.absolute()

    @staticmethod
    def get_bin_dir():
        return FreeFlowUtils.get_extension_dir() / "bin"

    @staticmethod
    def log(message, level="INFO"):
        print(f"🌊 [FreeFlow] {level}: {message}")

    @staticmethod
    def run_command(cmd, cwd=None, live_output=False):
        """
        Execute a command with optional live output streaming.
        Returns dict with returncode, stdout, stderr.
        """
        import subprocess
        import threading
        
        stdout_lines = []
        stderr_lines = []
        
        if live_output:
            process = subprocess.Popen(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            def read_stream(pipe, storage, prefix=""):
                for line in iter(pipe.readline, ''):
                    if line:
                        stripped = line.strip()
                        storage.append(stripped)
                        if prefix:
                            print(f"🌊 [{prefix}] {stripped}")
                        else:
                            print(f"   {stripped}")
                pipe.close()
            
            stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, stdout_lines, ""))
            stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, stderr_lines, "ERR"))
            
            stdout_thread.daemon = True
            stderr_thread.daemon = True
            
            stdout_thread.start()
            stderr_thread.start()
            
            returncode = process.wait()
            
            stdout_thread.join(timeout=2)
            stderr_thread.join(timeout=2)
            
            return {
                'returncode': returncode,
                'stdout': '\n'.join(stdout_lines),
                'stderr': '\n'.join(stderr_lines)
            }
        else:
            result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
            return {
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr
            }

    @staticmethod
    def download_file(url, dest_path):
        FreeFlowUtils.log(f"Downloading {url} to {dest_path}...")
        try:
            # Create request with User-Agent to avoid some 403s
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
                total_size = int(response.getheader('Content-Length', 0))
                block_size = 1024 * 1024 # 1MB chunks
                downloaded = 0
                
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    
                    if total_size > 0:
                        percent = downloaded * 100 / total_size
                        sys.stdout.write(f"\rDownload progress: {percent:.1f}%")
                        sys.stdout.flush()
            print() # Newline
            
            # Validate file size (min 100KB)
            if dest_path.stat().st_size < 100 * 1024:
                FreeFlowUtils.log("Downloaded file is too small! (Likely 404 or error page)", "ERROR")
                with open(dest_path, 'r', errors='ignore') as f:
                    print(f"File content start: {f.read(200)}")
                return False
                
            return True
        except Exception as e:
            FreeFlowUtils.log(f"Download Failed: {e}", "ERROR")
            return False

    @staticmethod
    def extract_archive(archive_path, extract_to):
        FreeFlowUtils.log(f"Extracting {archive_path.name}...")
        try:
            if archive_path.suffix == ".zip":
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    print(f"DEBUG: Zip contents: {zip_ref.namelist()[:5]}...")
                    zip_ref.extractall(extract_to)
            elif archive_path.name.endswith(".tar.gz") or archive_path.name.endswith(".tgz"):
                 with tarfile.open(archive_path, "r:gz") as tar:
                    tar.extractall(extract_to)
            elif archive_path.name.endswith(".tar.xz") or archive_path.name.endswith(".txz"):
                 with tarfile.open(archive_path, "r:xz") as tar:
                    tar.extractall(extract_to)
            else:
                FreeFlowUtils.log(f"Unknown archive format: {archive_path.suffix}", "ERROR")
                return False
            return True
        except Exception as e:
            FreeFlowUtils.log(f"Extraction Failed: {e}", "ERROR")
            return False

    @staticmethod
    def ensure_executable(path):
        if platform.system() != "Windows":
             try:
                st = os.stat(path)
                os.chmod(path, st.st_mode | 0o111)
             except Exception as e:
                 print(f"Warning: Could not make {path} executable: {e}")

    @staticmethod
    def get_local_version(tool_name):
        """Reads installed version from bin/versions.json"""
        import json
        v_file = FreeFlowUtils.get_bin_dir() / "versions.json"
        if not v_file.exists(): return None
        try:
            with open(v_file, 'r') as f:
                data = json.load(f)
                return data.get(tool_name)
        except: return None

    @staticmethod
    def save_local_version(tool_name, version):
        """Updates bin/versions.json"""
        import json
        bin_dir = FreeFlowUtils.get_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)
        v_file = bin_dir / "versions.json"
        
        data = {}
        if v_file.exists():
            try:
                with open(v_file, 'r') as f: data = json.load(f)
            except: pass
        
        data[tool_name] = version
        with open(v_file, 'w') as f:
            json.dump(data, f, indent=4)

    @staticmethod
    def install_brush(version=BRUSH_VERSION):
        bin_dir = FreeFlowUtils.get_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        system = FreeFlowUtils.get_os()
        # Dynamic URL Construction
        if system == "Windows":
            url = f"https://github.com/ArthurBrussee/brush/releases/download/v{version}/brush-app-x86_64-pc-windows-msvc.zip"
            archive_path = bin_dir / "brush.zip"
        elif system == "Darwin":
            url = f"https://github.com/ArthurBrussee/brush/releases/download/v{version}/brush-app-aarch64-apple-darwin.tar.xz"
            archive_path = bin_dir / "brush.tar.xz"
        else:
            url = f"https://github.com/ArthurBrussee/brush/releases/download/v{version}/brush-app-x86_64-unknown-linux-gnu.tar.xz"
            archive_path = bin_dir / "brush.tar.xz"
            
        FreeFlowUtils.log(f"Installing Brush v{version}...")
            
        if FreeFlowUtils.download_file(url, archive_path):
            if FreeFlowUtils.extract_archive(archive_path, bin_dir):
                # Cleanup archive
                try: archive_path.unlink() 
                except: pass
                
                # Verify extraction
                if system == "Windows":
                    # Check recursively for brush.exe
                    brush_exe = None
                    for path in bin_dir.rglob("brush.exe"):
                        brush_exe = path
                        break
                    
                    if brush_exe:
                        pass
                else:
                    # Mac/Linux tar usually extracts 'brush' binary
                    brush_bin = bin_dir / "brush"
                    if not brush_bin.exists():
                         # Check recursively
                         for path in bin_dir.rglob("brush"):
                             # Make sure it's a file
                             if path.is_file() and not path.name.endswith(".zip") and not path.name.endswith(".tar"):
                                 brush_bin = path
                                 break
                    
                    if brush_bin and brush_bin.exists():
                        FreeFlowUtils.ensure_executable(brush_bin)
                
                # SAVE VERSION
                FreeFlowUtils.save_local_version("brush", version)
                return True
        return False
        
    @staticmethod
    def _fetch_github_version(repo_name):
        """Fetches 'tag_name' from GitHub Releases API."""
        import json
        url = f"https://api.github.com/repos/{repo_name}/releases/latest"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'FreeFlow-Version-Checker'})
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    data = json.loads(response.read())
                    return data.get("tag_name", "").lstrip("v")
        except Exception:
            pass # Fail silently on network errors
        return None

    @staticmethod
    def _fetch_npm_version(package_name):
        """Fetches latest version from npm registry API."""
        import json
        url = f"https://registry.npmjs.org/{package_name}/latest"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'FreeFlow-Version-Checker'})
            with urllib.request.urlopen(req, timeout=3) as response:
                if response.status == 200:
                    data = json.loads(response.read())
                    return data.get("version", "")
        except Exception:
            pass # Fail silently on network errors
        return None

    @staticmethod
    def check_updates():
        """Spawns a thread to check for updates without blocking startup."""
        def run_check():
            # Check Brush
            latest_brush = FreeFlowUtils._fetch_github_version(BRUSH_REPO)
            if latest_brush and latest_brush != BRUSH_VERSION:
                print(f"\\n🌊 \\033[93m[Update Available] Brush v{latest_brush} is out! (Current: v{BRUSH_VERSION})\\033[0m")
                print(f"   See: https://github.com/{BRUSH_REPO}/releases\\n")
            
            # Check gsplat.js npm package
            latest_gsplat = FreeFlowUtils._fetch_npm_version("gsplat")
            if latest_gsplat and latest_gsplat != GSPLAT_VERSION:
                print(f"\\n🌊 \\033[93m[Update Available] gsplat.js v{latest_gsplat} is out! (Current: v{GSPLAT_VERSION})\\033[0m")
                print(f"   Update GSPLAT_VERSION in utils.py and player.js to benefit from improvements.\\n")
        
        # Run in thread
        import threading
        t = threading.Thread(target=run_check)
        t.daemon = True
        t.start()

    @staticmethod
    def install_colmap(version="3.13.0"):
        # ONLY SUPPORTED FOR WINDOWS AUTO-INSTALL
        if FreeFlowUtils.get_os() != "Windows":
             FreeFlowUtils.log("Auto-Installation of COLMAP is only supported on Windows.", "WARN")
             return False

        bin_dir = FreeFlowUtils.get_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        # Dynamic URL
        url = f"https://github.com/colmap/colmap/releases/download/{version}/colmap-x64-windows-cuda.zip"
        archive_path = bin_dir / "colmap.zip"
        
        FreeFlowUtils.log(f"Installing COLMAP {version}...")
        
        if FreeFlowUtils.download_file(url, archive_path):
            if FreeFlowUtils.extract_archive(archive_path, bin_dir):
                try: archive_path.unlink()
                except: pass
                # Windows COLMAP zip extracts to a folder named "COLMAP-3.10-windows-cuda" or similar
                # We need to find it and maybe flatten it or update logic?
                # Actually, simplest is to let it sit there and find it dynamically.
                
                FreeFlowUtils.save_local_version("colmap", version)
                return True
        return False

    @staticmethod
    def install_ffmpeg():
        FreeFlowUtils.log("Auto-Installing FFmpeg...")
        bin_dir = FreeFlowUtils.get_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        system = FreeFlowUtils.get_os()
        # URLs for static builds
        if system == "Windows":
            # Using Gyan.dev Release
            url = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
            archive_path = bin_dir / "ffmpeg.zip"
        elif system == "Darwin": # Mac
            # Evermeet.cx or similar static build
            url = "https://evermeet.cx/ffmpeg/ffmpeg-6.0.zip" # Valid static build
            archive_path = bin_dir / "ffmpeg.zip"
        else: # Linux
            url = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
            archive_path = bin_dir / "ffmpeg.tar.xz"
            
        if FreeFlowUtils.download_file(url, archive_path):
            if FreeFlowUtils.extract_archive(archive_path, bin_dir):
                archive_path.unlink()
                
                # Locate and move binary to root of bin/
                found = False
                exe_name = "ffmpeg.exe" if system == "Windows" else "ffmpeg"
                
                # Search recursively
                for path in bin_dir.rglob(exe_name):
                    if path.is_file():
                         # Move to bins root
                         target = bin_dir / exe_name
                         if path != target:
                             shutil.move(str(path), str(target))
                         FreeFlowUtils.ensure_executable(target)
                         found = True
                         break
                
                if found:
                    FreeFlowUtils.log("FFmpeg installed successfully.")
                    return True
                else:
                    FreeFlowUtils.log("Extracted FFmpeg but could not find binary.", "ERROR")

        return False

    @staticmethod
    def install_vocab_tree():
        FreeFlowUtils.log("Auto-Installing COLMAP Vocabulary Tree (150MB)...")
        bin_dir = FreeFlowUtils.get_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)
        
        target_path = bin_dir / "vocab_tree_faiss_flickr100K_words32K.bin"
        if target_path.exists():
            return True
            
        success = False
        for url in VOCAB_TREE_MIRRORS:
            FreeFlowUtils.log(f"Attempting download from: {url}")
            if FreeFlowUtils.download_file(url, target_path):
                FreeFlowUtils.log("Vocabulary Tree downloaded successfully.")
                success = True
                break
            else:
                FreeFlowUtils.log(f"Mirror failed. Trying next...", "WARN")
                
        if not success:
             FreeFlowUtils.log("ALL Mirrors failed. Please download 'vocab_tree_faiss_flickr100K_words32K.bin' manually and place in: " + str(bin_dir), "ERROR")
        
        return success
        
    @staticmethod
    def get_vocab_tree_path():
        bin_dir = FreeFlowUtils.get_bin_dir()
        path = bin_dir / "vocab_tree_faiss_flickr100K_words32K.bin"
        if path.exists():
            return path
        return None

    @staticmethod
    def check_and_install():
        """Automatically checks for binaries, installs if missing, AND updates if available."""
        print("🌊 Checking FreeFlow dependencies...")
        
        # 1. CHECK BRUSH
        brush_ver = FreeFlowUtils.get_local_version("brush")
        latest_brush = FreeFlowUtils._fetch_github_version(BRUSH_REPO) # e.g. "0.4.0"
        
        # Logic: If missing OR (installed_ver != latest_ver and latest is valid)
        if not FreeFlowUtils.get_binary_path("brush"):
            print("   • Brush binary missing. Auto-Installing...")
            target_ver = latest_brush if latest_brush else BRUSH_VERSION
            FreeFlowUtils.install_brush(target_ver)
        elif latest_brush and brush_ver != latest_brush:
            print(f"   • 🌊 Update Found: Brush ({brush_ver} -> {latest_brush}). Auto-Updating...")
            FreeFlowUtils.install_brush(latest_brush)
        
        # 2. CHECK COLMAP (Windows Only)
        if FreeFlowUtils.get_os() == "Windows":
            colmap_ver = FreeFlowUtils.get_local_version("colmap")
            if not colmap_ver: colmap_ver = "Unknown"
            
            # Fetch latest COLMAP (Note: Releases might be "3.12.0" or "dev")
            latest_colmap = FreeFlowUtils._fetch_github_version(COLMAP_REPO)
            
            if not FreeFlowUtils.get_binary_path("colmap"):
                 print("   • COLMAP binary missing. Auto-Installing...")
                 if latest_colmap: FreeFlowUtils.install_colmap(latest_colmap)
                 else: FreeFlowUtils.install_colmap() # Uses default
            elif latest_colmap and colmap_ver != latest_colmap:
                 print(f"   • 🌊 Update Found: COLMAP ({colmap_ver} -> {latest_colmap}). Auto-Updating...")
                 FreeFlowUtils.install_colmap(latest_colmap)

        # Check FFmpeg
        if not FreeFlowUtils.get_binary_path("ffmpeg"):
             print("   • FFmpeg binary missing. Auto-Installing...")
             FreeFlowUtils.install_ffmpeg()
             
        # Check Python Packages (Fix for shared environment conflicts)
        FreeFlowUtils.install_python_packages()

    @staticmethod
    def install_python_packages():
        """Checks and installs critical python dependencies."""
        required = ["scikit-image", "numpy", "scipy", "mediapipe"]
        import importlib.util
        import subprocess
        
        missing = []
        for pkg in required:
            # Map package name to import name if different
            import_name = pkg
            if pkg == "scikit-image": import_name = "skimage"
            
            if importlib.util.find_spec(import_name) is None:
                missing.append(pkg)
                
        if missing:
            print(f"   • Missing Python Packages: {missing}. Installing...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
                print("   • Installed successfully.")
            except Exception as e:
                FreeFlowUtils.log(f"Failed to pip install {missing}: {e}", "ERROR")

        # Force MediaPipe Version Check (Fix for v0.10.21 breaking legacy API)
        try:
            import mediapipe
            # Pin to 0.10.14 which supports Python 3.12 AND keeps mp.solutions
            TARGET_MP_VERSION = "0.10.14" 
            if mediapipe.__version__ != TARGET_MP_VERSION:
                print(f"   • 🌊 Enforcing MediaPipe v{TARGET_MP_VERSION} (Current: {mediapipe.__version__} is unstable/breaking).")
                print("     Re-installing correct version...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"mediapipe=={TARGET_MP_VERSION}"])
        except Exception as e:
            pass # Module not found yet, will be caught next startup or handled above

    @staticmethod
    def get_binary_path(binary_name):
        """
        Resolves binary path based on OS.
        binary_name: 'brush' or 'colmap'
        """
        system = FreeFlowUtils.get_os()
        bin_dir = FreeFlowUtils.get_bin_dir()

        if binary_name == "brush":
            if system == "Windows":
                 # 1. Check common locations first (NOTE: Binary is brush_app.exe, not brush.exe!)
                possible_paths = [
                    bin_dir / "brush_app.exe",
                    bin_dir / "brush-app-x86_64-pc-windows-msvc" / "brush_app.exe",
                    bin_dir / "brush.exe",  # Fallback in case they rename it
                ]
                for p in possible_paths:
                    if p.exists(): return p

                # 2. Recursive Search (Fallback)
                print(f"DEBUG: Searching recursively for brush_app.exe in {bin_dir}...")
                for item in bin_dir.rglob("brush_app.exe"):
                    print(f"DEBUG: Found brush_app.exe at {item}")
                    return item
                
                # Last resort: try old name
                for item in bin_dir.rglob("brush.exe"):
                    return item
                
                return None
            else:
                path = bin_dir / "brush"
                if path.exists(): return path
                
                # Check potential subfolders (recursive)
                for item in bin_dir.rglob("brush"):
                     # Make sure it's a file and not a dir
                     if item.is_file() and not item.name.endswith(".zip") and not item.name.endswith(".tar"):
                         return item
                return None
            
            return None

        elif binary_name == "colmap":
            if system == "Windows":
                # Check common locations first
                path = bin_dir / "colmap.exe"
                if path.exists(): return path
                
                 # Check recursively (often inside COLMAP-3.10-windows-cuda folder)
                for item in bin_dir.rglob("colmap.exe"):
                    return item
                    
            elif system == "Darwin":
                # Check brew install location (Apple Silicon)
                path = Path("/opt/homebrew/bin/colmap")
                if path.exists(): return path
                
                # Check brew install location (Intel)
                path = Path("/usr/local/bin/colmap")
                if path.exists(): return path
                
                # Check absolute bin_dir
                path = bin_dir / "colmap"
                if path.exists(): return path
                
                # Check system path
                if shutil.which("colmap"):
                    return Path(shutil.which("colmap"))
                 
                # Final check for .app bundle (if user installed GUI app manually)
                path = Path("/Applications/COLMAP.app/Contents/MacOS/colmap")
                if path.exists(): return path
            else:
                 # linux
                path = bin_dir / "colmap"
                if path.exists(): return path
                if shutil.which("colmap"):
                    return Path(shutil.which("colmap"))

        elif binary_name == "ffmpeg":
             # Try system ffmpeg first
             if shutil.which("ffmpeg"):
                 return Path(shutil.which("ffmpeg"))
             # Fallback to local
             path = bin_dir / "ffmpeg"
             if path.exists(): return path
             path = bin_dir / "ffmpeg.exe"
             if path.exists(): return path
             
             print(f"DEBUG: FFmpeg not found in PATH or {bin_dir}")
             return None
             
        return None
