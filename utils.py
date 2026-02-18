
# --- FreeFlow Utility Class ---
import platform
import os
import sys
import shutil
import urllib.request
import urllib.error
import zipfile
import tarfile
import ssl
from pathlib import Path
import folder_paths

# --- CONSTANTS ---
# --- CONSTANTS ---
BRUSH_VERSION = "0.3.0"
BRUSH_REPO = "ArthurBrussee/brush"
COLMAP_REPO = "colmap/colmap"
COLMAP_DEFAULT_VERSION = "3.13.0"
GSPLAT_VERSION = "1.2.9"  # gsplat.js npm version
GSPLAT_CDN_URL = f"https://cdn.jsdelivr.net/npm/gsplat@{GSPLAT_VERSION}/dist/index.es.js"

# Using placeholder URLs - In production, use exact GitHub Release Asset URLs
BRUSH_URL_WIN = f"https://github.com/ArthurBrussee/brush/releases/download/v{BRUSH_VERSION}/brush-app-x86_64-pc-windows-msvc.zip"
BRUSH_URL_MAC = f"https://github.com/ArthurBrussee/brush/releases/download/v{BRUSH_VERSION}/brush-app-aarch64-apple-darwin.tar.xz"
BRUSH_URL_LINUX = f"https://github.com/ArthurBrussee/brush/releases/download/v{BRUSH_VERSION}/brush-app-x86_64-unknown-linux-gnu.tar.xz"

COLMAP_URL_WIN = f"https://github.com/colmap/colmap/releases/download/{COLMAP_DEFAULT_VERSION}/colmap-x64-windows-cuda.zip"
COLMAP_STRATEGY_ENV = "FREEFLOW_COLMAP_INSTALL_STRATEGY"
COLMAP_LINUX_URL_ENV = "FREEFLOW_COLMAP_LINUX_URL"
COLMAP_LINUX_LOCAL_ARCHIVE_ENV = "FREEFLOW_COLMAP_LINUX_LOCAL_ARCHIVE"
MICROMAMBA_LINUX_X64_URL = "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-64"
MICROMAMBA_LINUX_AARCH64_URL = "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-linux-aarch64"
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
    def _is_truthy_env(var_name):
        value = os.environ.get(var_name, "")
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _build_ssl_contexts():
        """
        Build SSL contexts for robust cross-platform downloads.

        Order:
        1) Explicit CA bundle from env vars
        2) System trust store
        3) certifi trust store (if available)

        FREEFLOW_SSL_NO_VERIFY=1 forces insecure mode as an explicit opt-in.
        """
        if FreeFlowUtils._is_truthy_env("FREEFLOW_SSL_NO_VERIFY"):
            FreeFlowUtils.log("FREEFLOW_SSL_NO_VERIFY=1 set: TLS verification DISABLED for FreeFlow downloads", "WARN")
            return [("insecure", ssl._create_unverified_context())]

        contexts = []
        seen = set()

        for env_name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            cert_path = os.environ.get(env_name)
            if not cert_path:
                continue
            key = (env_name, cert_path)
            if key in seen:
                continue
            seen.add(key)
            try:
                contexts.append((f"{env_name}:{cert_path}", ssl.create_default_context(cafile=cert_path)))
            except Exception as e:
                FreeFlowUtils.log(f"Ignoring invalid CA bundle in {env_name}: {e}", "WARN")

        contexts.append(("system", ssl.create_default_context()))

        try:
            import certifi
            certifi_path = certifi.where()
            if certifi_path:
                contexts.append((f"certifi:{certifi_path}", ssl.create_default_context(cafile=certifi_path)))
        except Exception:
            pass

        return contexts

    @staticmethod
    def _urlopen_with_ssl_fallback(url, headers=None, timeout=None):
        merged_headers = {'User-Agent': 'Mozilla/5.0'}
        if headers:
            merged_headers.update(headers)

        req = urllib.request.Request(url, headers=merged_headers)
        contexts = FreeFlowUtils._build_ssl_contexts()

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
                    FreeFlowUtils.log(f"SSL open failed using {ctx_name}, trying fallback trust store...", "WARN")
                    continue

                raise

        raise RuntimeError("Unable to open URL with available SSL contexts")

    @staticmethod
    def download_file(url, dest_path):
        FreeFlowUtils.log(f"Downloading {url} to {dest_path}...")
        try:
            with FreeFlowUtils._urlopen_with_ssl_fallback(url) as response, open(dest_path, 'wb') as out_file:
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
            elif archive_path.name.endswith(".tar"):
                 with tarfile.open(archive_path, "r:") as tar:
                    tar.extractall(extract_to)
            elif archive_path.name.endswith(".tar.bz2") or archive_path.name.endswith(".tbz2"):
                 with tarfile.open(archive_path, "r:bz2") as tar:
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
            with FreeFlowUtils._urlopen_with_ssl_fallback(url, headers={'User-Agent': 'FreeFlow-Version-Checker'}, timeout=3) as response:
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
            with FreeFlowUtils._urlopen_with_ssl_fallback(url, headers={'User-Agent': 'FreeFlow-Version-Checker'}, timeout=3) as response:
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
    def _get_colmap_install_strategy():
        raw = os.environ.get(COLMAP_STRATEGY_ENV, "binary-first").strip().lower()
        alias_map = {
            "default": "binary-first",
            "auto": "binary-first",
            "binary": "binary-first",
            "mamba": "mamba-only",
            "conda": "mamba-only",
        }
        strategy = alias_map.get(raw, raw)
        if strategy not in {"binary-first", "binary-only", "mamba-only"}:
            FreeFlowUtils.log(
                f"Invalid {COLMAP_STRATEGY_ENV}='{raw}', using 'binary-first'",
                "WARN",
            )
            return "binary-first"
        return strategy

    @staticmethod
    def _get_colmap_env_path():
        return FreeFlowUtils.get_extension_dir() / ".colmap_env"

    @staticmethod
    def _get_mamba_root_path():
        return FreeFlowUtils.get_extension_dir() / ".mamba"

    @staticmethod
    def _get_micromamba_path():
        exe = "micromamba.exe" if FreeFlowUtils.get_os() == "Windows" else "micromamba"
        return FreeFlowUtils.get_bin_dir() / exe

    @staticmethod
    def _run_subprocess_with_env(cmd, cwd=None, env=None):
        import subprocess

        try:
            result = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return 1, "", str(e)

    @staticmethod
    def _find_colmap_linux_asset_url(version):
        """
        Look for official Linux binary assets. Returns URL or None.
        Currently COLMAP publishes Windows binaries, but this keeps support
        forward-compatible if Linux assets are added later.
        """
        import json

        normalized = (version or COLMAP_DEFAULT_VERSION).lstrip("v")
        tag_urls = [
            f"https://api.github.com/repos/{COLMAP_REPO}/releases/tags/{normalized}",
            f"https://api.github.com/repos/{COLMAP_REPO}/releases/tags/v{normalized}",
            f"https://api.github.com/repos/{COLMAP_REPO}/releases/latest",
        ]

        for api_url in tag_urls:
            try:
                with FreeFlowUtils._urlopen_with_ssl_fallback(
                    api_url,
                    headers={"User-Agent": "FreeFlow-Colmap-Installer"},
                    timeout=4,
                ) as response:
                    if response.status != 200:
                        continue
                    data = json.loads(response.read())
            except Exception:
                continue

            for asset in data.get("assets", []):
                name = (asset.get("name") or "").lower()
                if not name or name.endswith(".sha256"):
                    continue

                looks_linux = any(token in name for token in ("linux", "ubuntu", "appimage"))
                looks_colmap = "colmap" in name
                if looks_linux and looks_colmap:
                    return asset.get("browser_download_url")

        return None

    @staticmethod
    def _find_colmap_candidate_in_dir(search_dir):
        if not search_dir.exists():
            return None

        candidates = []
        for pattern in ("colmap", "COLMAP", "*.AppImage"):
            for item in search_dir.rglob(pattern):
                if item.is_file():
                    lower = item.name.lower()
                    if lower.endswith((".sha256", ".txt", ".json", ".md")):
                        continue
                    candidates.append(item)

        if not candidates:
            return None

        def _rank(path_obj):
            lower = path_obj.name.lower()
            if lower == "colmap":
                return 0
            if lower.endswith(".appimage"):
                return 1
            return 2

        candidates.sort(key=_rank)
        return candidates[0]

    @staticmethod
    def _install_colmap_linux_binary(version=COLMAP_DEFAULT_VERSION):
        """
        Try Linux binary install in this order:
        1) FREEFLOW_COLMAP_LINUX_LOCAL_ARCHIVE local file path
        2) FREEFLOW_COLMAP_LINUX_URL
        3) Official release Linux asset (if ever provided upstream)
        """
        if FreeFlowUtils.get_os() != "Linux":
            return False

        from urllib.parse import urlparse, unquote

        bin_dir = FreeFlowUtils.get_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)

        source_path = None
        downloaded_temp = False

        local_archive = os.environ.get(COLMAP_LINUX_LOCAL_ARCHIVE_ENV)
        if local_archive:
            candidate = Path(local_archive).expanduser()
            if candidate.exists() and candidate.is_file():
                source_path = candidate
                FreeFlowUtils.log(f"Using local Linux COLMAP source: {source_path}")
            else:
                FreeFlowUtils.log(
                    f"{COLMAP_LINUX_LOCAL_ARCHIVE_ENV} set but file not found: {candidate}",
                    "WARN",
                )

        if source_path is None:
            custom_url = os.environ.get(COLMAP_LINUX_URL_ENV)
            source_url = custom_url or FreeFlowUtils._find_colmap_linux_asset_url(version)

            if not source_url:
                FreeFlowUtils.log(
                    "No Linux COLMAP binary URL found (set FREEFLOW_COLMAP_LINUX_URL for custom binary/archive)",
                    "WARN",
                )
                return False

            filename = Path(unquote(urlparse(source_url).path)).name or "colmap-linux-download"
            source_path = bin_dir / f"_tmp_{filename}"

            FreeFlowUtils.log(f"Downloading Linux COLMAP binary source from {source_url}...")
            if not FreeFlowUtils.download_file(source_url, source_path):
                return False
            downloaded_temp = True

        target_bin = bin_dir / "colmap"
        extract_dir = bin_dir / "_colmap_extract_tmp"

        try:
            source_lower = source_path.name.lower()
            is_archive = source_lower.endswith((".zip", ".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2", ".tbz2", ".tar"))

            if is_archive:
                if extract_dir.exists():
                    shutil.rmtree(extract_dir, ignore_errors=True)
                extract_dir.mkdir(parents=True, exist_ok=True)

                if not FreeFlowUtils.extract_archive(source_path, extract_dir):
                    return False

                found = FreeFlowUtils._find_colmap_candidate_in_dir(extract_dir)
                if not found:
                    FreeFlowUtils.log("Linux COLMAP archive extracted but no executable was found", "ERROR")
                    return False
                source_binary = found
            else:
                source_binary = source_path

            same_file = False
            try:
                same_file = source_binary.resolve() == target_bin.resolve()
            except Exception:
                same_file = False

            if not same_file:
                if target_bin.exists():
                    try:
                        target_bin.unlink()
                    except Exception:
                        pass

                shutil.copy2(source_binary, target_bin)

            FreeFlowUtils.ensure_executable(target_bin)

            if not target_bin.exists():
                FreeFlowUtils.log("COLMAP binary installation failed: target missing after copy", "ERROR")
                return False

            FreeFlowUtils.save_local_version("colmap", f"{version}-linux-binary")
            FreeFlowUtils.log(f"COLMAP installed locally at {target_bin}")
            return True
        finally:
            if extract_dir.exists():
                shutil.rmtree(extract_dir, ignore_errors=True)
            if downloaded_temp and source_path and source_path.exists():
                try:
                    source_path.unlink()
                except Exception:
                    pass

    @staticmethod
    def _install_micromamba_local():
        if FreeFlowUtils.get_os() != "Linux":
            return False

        bin_dir = FreeFlowUtils.get_bin_dir()
        bin_dir.mkdir(parents=True, exist_ok=True)

        micromamba_path = FreeFlowUtils._get_micromamba_path()
        if micromamba_path.exists():
            FreeFlowUtils.ensure_executable(micromamba_path)
            return True

        machine = platform.machine().lower()
        if machine in ("x86_64", "amd64"):
            url = MICROMAMBA_LINUX_X64_URL
        elif machine in ("aarch64", "arm64"):
            url = MICROMAMBA_LINUX_AARCH64_URL
        else:
            FreeFlowUtils.log(f"Unsupported Linux architecture for micromamba bootstrap: {machine}", "WARN")
            return False

        temp_path = micromamba_path.with_suffix(".tmp")
        FreeFlowUtils.log(f"Installing local micromamba ({machine})...")
        if not FreeFlowUtils.download_file(url, temp_path):
            return False

        if micromamba_path.exists():
            try:
                micromamba_path.unlink()
            except Exception:
                pass

        shutil.move(str(temp_path), str(micromamba_path))
        FreeFlowUtils.ensure_executable(micromamba_path)
        return micromamba_path.exists()

    @staticmethod
    def _install_colmap_linux_mamba(version=COLMAP_DEFAULT_VERSION):
        if FreeFlowUtils.get_os() != "Linux":
            return False

        env_path = FreeFlowUtils._get_colmap_env_path()
        target_bin = env_path / "bin" / "colmap"
        if target_bin.exists():
            FreeFlowUtils.ensure_executable(target_bin)
            return True

        if not FreeFlowUtils._install_micromamba_local():
            return False

        micromamba = FreeFlowUtils._get_micromamba_path()
        mamba_root = FreeFlowUtils._get_mamba_root_path()
        mamba_root.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["MAMBA_ROOT_PREFIX"] = str(mamba_root)
        env["MAMBA_NO_BANNER"] = "1"

        FreeFlowUtils.log("Installing COLMAP via project-local micromamba env (.colmap_env)...")
        cmd_create = [
            str(micromamba),
            "create",
            "-y",
            "-p",
            str(env_path),
            "-c",
            "conda-forge",
            "colmap",
        ]
        rc, stdout, stderr = FreeFlowUtils._run_subprocess_with_env(
            cmd_create,
            cwd=str(FreeFlowUtils.get_extension_dir()),
            env=env,
        )

        if rc != 0:
            # If environment exists but create failed, try install into prefix.
            cmd_install = [
                str(micromamba),
                "install",
                "-y",
                "-p",
                str(env_path),
                "-c",
                "conda-forge",
                "colmap",
            ]
            rc, stdout, stderr = FreeFlowUtils._run_subprocess_with_env(
                cmd_install,
                cwd=str(FreeFlowUtils.get_extension_dir()),
                env=env,
            )

        if rc != 0:
            FreeFlowUtils.log("COLMAP micromamba install failed", "ERROR")
            if stderr:
                FreeFlowUtils.log(stderr[-1000:], "ERROR")
            return False

        if not target_bin.exists():
            FreeFlowUtils.log("COLMAP install finished but binary not found in .colmap_env/bin", "ERROR")
            return False

        FreeFlowUtils.ensure_executable(target_bin)
        FreeFlowUtils.save_local_version("colmap", f"{version}-linux-mamba")
        FreeFlowUtils.log(f"COLMAP installed locally at {target_bin}")
        return True

    @staticmethod
    def install_colmap(version=COLMAP_DEFAULT_VERSION):
        system = FreeFlowUtils.get_os()

        if system == "Windows":
            bin_dir = FreeFlowUtils.get_bin_dir()
            bin_dir.mkdir(parents=True, exist_ok=True)

            # Dynamic URL
            url = f"https://github.com/colmap/colmap/releases/download/{version}/colmap-x64-windows-cuda.zip"
            archive_path = bin_dir / "colmap.zip"

            FreeFlowUtils.log(f"Installing COLMAP {version}...")

            if FreeFlowUtils.download_file(url, archive_path):
                if FreeFlowUtils.extract_archive(archive_path, bin_dir):
                    try:
                        archive_path.unlink()
                    except Exception:
                        pass

                    FreeFlowUtils.save_local_version("colmap", version)
                    return True
            return False

        if system == "Linux":
            strategy = FreeFlowUtils._get_colmap_install_strategy()
            FreeFlowUtils.log(
                f"Linux COLMAP install strategy: {strategy} (all local to extension directory)"
            )

            if strategy in {"binary-first", "binary-only"}:
                if FreeFlowUtils._install_colmap_linux_binary(version):
                    return True
                if strategy == "binary-only":
                    return False

            return FreeFlowUtils._install_colmap_linux_mamba(version)

        FreeFlowUtils.log(
            "Auto-Installation of COLMAP is currently supported on Windows and Linux.",
            "WARN",
        )
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
        
        # 2. CHECK COLMAP
        os_name = FreeFlowUtils.get_os()
        if os_name == "Windows":
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
        elif os_name == "Linux":
            if not FreeFlowUtils.get_binary_path("colmap"):
                print("   • COLMAP binary missing. Auto-Installing (local + isolated)...")
                target_colmap_ver = FreeFlowUtils._fetch_github_version(COLMAP_REPO) or COLMAP_DEFAULT_VERSION
                if not FreeFlowUtils.install_colmap(target_colmap_ver):
                    FreeFlowUtils.log(
                        "COLMAP auto-install failed. Set FREEFLOW_COLMAP_LINUX_URL to a Linux binary/archive or use FREEFLOW_COLMAP_INSTALL_STRATEGY=mamba-only",
                        "WARN",
                    )

        # Check FFmpeg
        if not FreeFlowUtils.get_binary_path("ffmpeg"):
             print("   • FFmpeg binary missing. Auto-Installing...")
             FreeFlowUtils.install_ffmpeg()
             
        # Check Python Packages (Fix for shared environment conflicts)
        FreeFlowUtils.install_python_packages()
        
        # 4. CHECK NERFSTUDIO (Splatfacto backend) - Isolated venv INSIDE project directory
        FreeFlowUtils.check_and_install_nerfstudio()

    @staticmethod
    def check_and_install_nerfstudio():
        """
        Check for Nerfstudio installation and auto-install if missing.
        
        Nerfstudio is installed in an isolated venv INSIDE the FreeFlow project:
        ComfyUi-FreeFlowGS/.nerfstudio_venv/
        
        This ensures NO system-wide changes and NO conflicts with ComfyUI.
        
        Works on: Windows, macOS, Linux (requires CUDA for GPU acceleration)
        """
        NerfstudioEnvironment = None
        import_error_details = []
        
        # Try multiple import strategies since utils.py can be called from different contexts
        try:
            # Strategy 1: Direct import (when nodes is in sys.path)
            from nodes.modules.nerfstudio_env import NerfstudioEnvironment
            print("   [Nerfstudio Import] Strategy 1 (direct import) succeeded")
        except ImportError as e:
            import_error_details.append(f"Strategy 1: {e}")
        
        if NerfstudioEnvironment is None:
            try:
                # Strategy 2: Add parent directory to path
                import sys
                from pathlib import Path
                utils_dir = Path(__file__).parent
                nodes_path = utils_dir / "nodes"
                print(f"   [Nerfstudio Import] Strategy 2: utils_dir={utils_dir}, nodes_path={nodes_path}, exists={nodes_path.exists()}")
                if nodes_path.exists() and str(utils_dir) not in sys.path:
                    sys.path.insert(0, str(utils_dir))
                from nodes.modules.nerfstudio_env import NerfstudioEnvironment
                print("   [Nerfstudio Import] Strategy 2 (path modification) succeeded")
            except ImportError as e:
                import_error_details.append(f"Strategy 2: {e}")
        
        if NerfstudioEnvironment is None:
            try:
                # Strategy 3: Relative import from package root
                from .nodes.modules.nerfstudio_env import NerfstudioEnvironment
                print("   [Nerfstudio Import] Strategy 3 (relative import) succeeded")
            except ImportError as e:
                import_error_details.append(f"Strategy 3: {e}")
        
        if NerfstudioEnvironment is None:
            print("   • Nerfstudio: Could not import NerfstudioEnvironment module")
            for detail in import_error_details:
                print(f"     {detail}")
            return False
        
        try:
            
            # Check if already installed
            if NerfstudioEnvironment.is_installed():
                version = NerfstudioEnvironment.get_version()
                
                # Check for updates (like Brush does)
                try:
                    new_version = NerfstudioEnvironment.check_for_update()
                    if new_version:
                        print(f"   • Nerfstudio: Update available ({version} -> {new_version})")
                        print("     Auto-updating Nerfstudio...")
                        
                        def progress_callback(message: str, progress: float):
                            pct = int(progress * 100)
                            print(f"     [{pct:3d}%] {message}")
                        
                        if NerfstudioEnvironment.upgrade(progress_callback):
                            version = NerfstudioEnvironment.get_version()
                            print(f"   • Nerfstudio: Updated to v{version}")
                        else:
                            print(f"   • Nerfstudio: Update failed, using v{version}")
                    else:
                        print(f"   • Nerfstudio: Installed (v{version}) - up to date")
                except Exception as e:
                    # Update check failed, but installation is fine
                    print(f"   • Nerfstudio: Installed (v{version})")
                
                return True
            
            # Check for existing installation elsewhere (conda, system PATH)
            existing = NerfstudioEnvironment.detect_existing()
            if existing:
                source, path = existing
                print(f"   • Nerfstudio: Found existing {source} installation at {path}")
                print(f"     (FreeFlow uses isolated venv inside project directory)")
            
            # Check if we should auto-install
            # Note: Nerfstudio installation can take 10-30 minutes, so we show a clear message
            print("   • Nerfstudio: Not installed in FreeFlow venv.")
            
            # Check prerequisites first
            cuda_version = NerfstudioEnvironment._detect_cuda_version()
            if cuda_version:
                print(f"     CUDA {cuda_version} detected - GPU acceleration available")
            else:
                print("     WARNING: No CUDA detected - Splatfacto requires CUDA for reasonable performance")
                print("     Training will be VERY slow on CPU. Consider using Brush engine instead.")
            
            # Auto-install with progress reporting
            print("     Auto-installing Nerfstudio (this may take 10-30 minutes)...")
            print(f"     Installation location: {NerfstudioEnvironment.VENV_PATH}")
            
            def progress_callback(message: str, progress: float):
                """Progress callback for installation."""
                pct = int(progress * 100)
                print(f"     [{pct:3d}%] {message}")
            
            success = NerfstudioEnvironment.create_venv(progress_callback)
            
            if success:
                version = NerfstudioEnvironment.get_version()
                print(f"   • Nerfstudio: Installed successfully (v{version})")
                return True
            else:
                print("   • Nerfstudio: Installation FAILED")
                print("     Splatfacto engine will not be available.")
                print("     You can retry manually: python -c \"from nodes.modules.nerfstudio_env import NerfstudioEnvironment; NerfstudioEnvironment.create_venv()\"")
                return False
                
        except ImportError as e:
            print(f"   • Nerfstudio: Module import error: {e}")
            return False
        except Exception as e:
            print(f"   • Nerfstudio: Auto-install error: {e}")
            import traceback
            traceback.print_exc()
            return False

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
            local_env_colmap = FreeFlowUtils._get_colmap_env_path() / ("Scripts/colmap.exe" if system == "Windows" else "bin/colmap")

            if system == "Windows":
                # Prefer local direct binary, then local isolated env.
                possible_paths = [
                    bin_dir / "colmap.exe",
                    local_env_colmap,
                ]
                for path in possible_paths:
                    if path.exists():
                        return path
                
                 # Check recursively (often inside COLMAP-3.10-windows-cuda folder)
                for item in bin_dir.rglob("colmap.exe"):
                    return item
                    
            elif system == "Darwin":
                # Prefer local isolated install first.
                local_paths = [
                    bin_dir / "colmap",
                    local_env_colmap,
                ]
                for path in local_paths:
                    if path.exists():
                        return path

                # Check brew install location (Apple Silicon)
                path = Path("/opt/homebrew/bin/colmap")
                if path.exists(): return path
                
                # Check brew install location (Intel)
                path = Path("/usr/local/bin/colmap")
                if path.exists(): return path
                
                # Check system path
                which_colmap = shutil.which("colmap")
                if which_colmap:
                    return Path(which_colmap)
                  
                # Final check for .app bundle (if user installed GUI app manually)
                path = Path("/Applications/COLMAP.app/Contents/MacOS/colmap")
                if path.exists(): return path
            else:
                # Linux: prefer local extension installs.
                local_paths = [
                    bin_dir / "colmap",
                    local_env_colmap,
                ]
                for path in local_paths:
                    if path.exists():
                        return path

                for item in bin_dir.rglob("colmap"):
                    if item.is_file():
                        return item

                # Fallback to system path for compatibility.
                which_colmap = shutil.which("colmap")
                if which_colmap:
                    return Path(which_colmap)

        elif binary_name == "ffmpeg":
             # Try system ffmpeg first
             which_ffmpeg = shutil.which("ffmpeg")
             if which_ffmpeg:
                 return Path(which_ffmpeg)
             # Fallback to local
             path = bin_dir / "ffmpeg"
             if path.exists(): return path
             path = bin_dir / "ffmpeg.exe"
             if path.exists(): return path
             
             print(f"DEBUG: FFmpeg not found in PATH or {bin_dir}")
             return None
             
        return None
