
import os
import sys
import platform
import shutil
import urllib.request
import urllib.error
import json
import zipfile
import tarfile
import stat
import ssl
from pathlib import Path
import logging

logger = logging.getLogger("FreeFlow.BrushLoader")


def _is_truthy_env(var_name):
    value = os.environ.get(var_name, "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _build_ssl_contexts():
    if _is_truthy_env("FREEFLOW_SSL_NO_VERIFY"):
        logger.warning("FREEFLOW_SSL_NO_VERIFY=1 set: TLS verification DISABLED for Brush downloads")
        return [("insecure", ssl._create_unverified_context())]

    contexts = []

    for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        cert_path = os.environ.get(env_name)
        if not cert_path:
            continue
        try:
            contexts.append((f"{env_name}:{cert_path}", ssl.create_default_context(cafile=cert_path)))
        except Exception as e:
            logger.warning(f"Ignoring invalid CA bundle in {env_name}: {e}")

    contexts.append(("system", ssl.create_default_context()))

    try:
        import certifi
        certifi_path = certifi.where()
        if certifi_path:
            contexts.append((f"certifi:{certifi_path}", ssl.create_default_context(cafile=certifi_path)))
    except Exception:
        pass

    return contexts


def _urlopen_with_ssl_fallback(url, headers=None, timeout=None):
    merged_headers = {'User-Agent': 'Mozilla/5.0'}
    if headers:
        merged_headers.update(headers)

    req = urllib.request.Request(url, headers=merged_headers)
    contexts = _build_ssl_contexts()

    for idx, (ctx_name, ctx) in enumerate(contexts):
        try:
            kwargs = {'context': ctx}
            if timeout is not None:
                kwargs['timeout'] = timeout
            return urllib.request.urlopen(req, **kwargs)
        except Exception as e:
            is_ssl_error = False

            if isinstance(e, ssl.SSLError):
                is_ssl_error = True

            if isinstance(e, urllib.error.URLError):
                reason = getattr(e, "reason", None)
                if isinstance(reason, ssl.SSLError):
                    is_ssl_error = True
                elif reason and "CERTIFICATE_VERIFY_FAILED" in str(reason):
                    is_ssl_error = True

            if "CERTIFICATE_VERIFY_FAILED" in str(e):
                is_ssl_error = True

            if is_ssl_error and idx < len(contexts) - 1:
                logger.warning(f"SSL open failed using {ctx_name}, trying fallback trust store...")
                continue

            raise

    raise RuntimeError("Unable to open URL with available SSL contexts")

def ensure_brush_binary():
    """
    Checks for Brush binary in local bin/ folder.
    If missing, attempts to finding it in system PATH.
    If still missing, attempts to download latest release from GitHub.
    """
    
    # 1. Check Local BIN
    base_path = Path(__file__).parent.parent.parent # Root of custom_node
    bin_dir = base_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    
    binary_name = "brush.exe" if platform.system() == "Windows" else "brush"
    local_bin = bin_dir / binary_name
    
    if local_bin.exists():
        # Ensure executable
        if platform.system() != "Windows":
             st = os.stat(local_bin)
             os.chmod(local_bin, st.st_mode | stat.S_IEXEC)
        return local_bin

    # 2. Check System PATH
    sys_path = shutil.which("brush")
    if sys_path:
        logger.info(f"Brush found in system PATH: {sys_path}")
        return Path(sys_path)
        
    # 3. Auto-Download
    logger.info("Brush binary not found. Attempting auto-download from GitHub...")
    
    try:
        # Determine architecture
        system = platform.system().lower() # darwin, linux, windows
        arch = platform.machine().lower() # arm64, x86_64, amd64
        
        # GitHub API
        url = "https://api.github.com/repos/ArthurBrussee/brush/releases/latest"
        with _urlopen_with_ssl_fallback(url, headers={'User-Agent': 'Mozilla/5.0'}) as response:
            data = json.loads(response.read().decode())
            
        assets = data.get('assets', [])
        target_asset = None
        
        assets = data.get('assets', [])
        target_asset = None
        
        # Debug: Log available assets
        asset_names = [a['name'] for a in assets]
        logger.info(f"Available release assets: {asset_names}")
        
        # Keywords
        mac_keywords = ["macos", "mac", "darwin", "osx", "apple"]
        
        system_lower = system.lower()
        
        for asset in assets:
            name = asset['name'].lower()
            
            # Match System
            is_mac = any(k in name for k in mac_keywords)
            is_win = "windows" in name or "win64" in name
            is_linux = "linux" in name
            
            if system_lower == "darwin" and is_mac:
                # Check Arch priority
                if "arm64" in name or "aarch64" in name:
                    target_asset = asset
                    break # Exact match
                elif "universal" in name or "unified" in name:
                     target_asset = asset # Good candidate, keep looking for specific arch? No, universal is usually fine.
                     break
                elif "x86" in name or "amd64" in name or "intel" in name:
                     if "x86" in arch or "amd64" in arch:
                         target_asset = asset
                         break
                
                # Fallback: If we haven't found a better match yet, take this generic mac one
                if not target_asset:
                     target_asset = asset

            elif system_lower == "windows" and is_win:
                target_asset = asset
                break
            
            elif system_lower == "linux" and is_linux:
                target_asset = asset
                break
        
        if not target_asset:
             raise RuntimeError(f"No suitable binary found for {system} {arch}. Available: {asset_names}")
             
        # Download
        download_url = target_asset['browser_download_url']
        logger.info(f"Downloading {target_asset['name']} from {download_url}...")
        
        zip_path = bin_dir / target_asset['name']
        with _urlopen_with_ssl_fallback(download_url) as response, open(zip_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            
        # Extract
        logger.info(f"Extracting {zip_path.name}...")
        
        try:
            if zip_path.suffix == '.zip':
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(bin_dir)
            else:
                # Use system tar for .tar.xz / .tar.gz reliability on Mac/Linux
                # Python's tarfile can be finicky with xz module availability
                if platform.system() != "Windows":
                    import subprocess
                    subprocess.run(["tar", "-xf", str(zip_path), "-C", str(bin_dir)], check=True)
                else:
                    # Windows usually gets .zip, but fallback for tar
                     with tarfile.open(zip_path, 'r:*') as tar_ref:
                         tar_ref.extractall(bin_dir)
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            raise e
        
        # Cleanup archive
        if zip_path.exists():
            os.remove(zip_path)
            
        # FIND BINARY
        # 1. List all files for debug
        found_files = []
        for root, dirs, files in os.walk(bin_dir):
            for f in files:
                found_files.append(Path(root) / f)
        
        logger.info(f"Extracted files: {[f.name for f in found_files]}")
        
        # 2. Search for candidate
        candidate = None
        # Strict check first
        for f in found_files:
            if f.name == binary_name:
                candidate = f
                break
        
        # Fuzzy check (e.g. brush-app, brush-cli)
        if not candidate:
            for f in found_files:
                if f.name.startswith("brush") and not f.name.endswith(".txt") and not f.name.endswith(".md"):
                    # Improve check: looks like a binary?
                    # On Mac/Linux no extension usually.
                    candidate = f
                    break
        
        if candidate:
            logger.info(f"Found executable candidate: {candidate.name}")
            
            # Normalize to bin/brush
            final_path = bin_dir / binary_name
            
            # Move if needed
            if candidate.resolve() != final_path.resolve():
                # If candidate is in a subfolder, move it up
                if final_path.exists():
                    os.remove(final_path) # overwrite old
                shutil.move(str(candidate), str(final_path))
                logger.info(f"Moved {candidate.name} -> {final_path}")
            
            # Set Executable Permission
            if platform.system() != "Windows":
                 st = os.stat(final_path)
                 os.chmod(final_path, st.st_mode | stat.S_IEXEC)
            
            return final_path
        else:
            raise FileNotFoundError(f"Extracted ok, but could not identify 'brush' binary in: {[f.name for f in found_files]}")

    except Exception as e:
        logger.error(f"Auto-download failed: {e}")
        logger.warning("Please download Brush manually from https://github.com/ArthurBrussee/brush/releases and place it in the 'bin' folder.")
        return None
