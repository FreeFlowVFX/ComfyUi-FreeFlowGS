import os
import torch
import numpy as np
from PIL import Image
import io
import requests
import json
import folder_paths
import base64
import time
import imageio
import torchaudio
import re
import comfy.utils

# Try importing the Google GenAI SDK
try:
    from google import genai
    from google.genai import types
    from google.genai import errors
    SDK_AVAILABLE = True
except ImportError:
    SDK_AVAILABLE = False
    print("TimNodes Warning: google-genai SDK not found. Some features may be limited.")

# --- Configuration & Helpers ---

def get_api_key(node_key=None):
    if node_key and node_key.strip(): return node_key.strip()
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f).get("GOOGLE_API_KEY", "")
    except: pass
    return os.environ.get("GOOGLE_API_KEY", "")

def tensor_to_base64(image_tensor):
    if image_tensor is None: return None
    if len(image_tensor.shape) > 3: image_tensor = image_tensor[0]
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    img = Image.fromarray(img_np)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

def batch_tensor_to_pil_list(image_tensor):
    pil_images = []
    if image_tensor is None: return pil_images
    
    # Standardize to numpy
    # ComfyUI Tensors are usually [B, H, W, C] or [B, H, W] or [H, W, C] or [H, W]
    if isinstance(image_tensor, torch.Tensor):
        img_np = image_tensor.cpu().numpy()
    else:
        img_np = image_tensor
        
    if len(img_np.shape) == 4:
        # [B, H, W, C]
        for i in range(img_np.shape[0]):
             item = (img_np[i] * 255).astype(np.uint8)
             if item.shape[-1] == 4: # RGBA
                 pil_images.append(Image.fromarray(item[:, :, :3], mode="RGB"))
             elif item.shape[-1] == 1: # Greyscale channel
                 pil_images.append(Image.fromarray(item.squeeze(-1), mode="L"))
             else: # RGB
                 pil_images.append(Image.fromarray(item, mode="RGB"))
                 
    elif len(img_np.shape) == 3:
        # Ambiguity: [B, H, W] (Mask Batch) OR [H, W, C] (Single Image)
        # Heuristic: If last dim is 1, 3, or 4, treat as Channels. Else Batch of Masks.
        c = img_np.shape[-1]
        
        if c in [1, 3, 4]:
            # Treat as Single Image [H, W, C]
            item = (img_np * 255).astype(np.uint8)
            if c == 4: pil_images.append(Image.fromarray(item[:, :, :3], mode="RGB"))
            elif c == 1: pil_images.append(Image.fromarray(item.squeeze(-1), mode="L"))
            else: pil_images.append(Image.fromarray(item, mode="RGB"))
        else:
            # Treat as Batch of Masks [B, H, W]
            for i in range(img_np.shape[0]):
                 item = (img_np[i] * 255).astype(np.uint8)
                 # Item is [H, W] -> Mode L
                 pil_images.append(Image.fromarray(item, mode="L"))
                 
    elif len(img_np.shape) == 2:
        # [H, W] -> Single Mask
        item = (img_np * 255).astype(np.uint8)
        pil_images.append(Image.fromarray(item, mode="L"))
        
    return pil_images

def tensor_to_pil(image_tensor):
    if image_tensor is None: return None
    if len(image_tensor.shape) > 3: image_tensor = image_tensor[0]
    img_np = (image_tensor.cpu().numpy() * 255).astype(np.uint8)
    
    if img_np.shape[-1] == 4:
        img_np = img_np[:, :, :3]
        return Image.fromarray(img_np, mode="RGB")
    elif img_np.shape[-1] == 1:
        return Image.fromarray(img_np.squeeze(-1), mode="L").convert("RGB")
    else:
        return Image.fromarray(img_np, mode="RGB")

def _get_gemini_safety_settings(filter_level):
    if not SDK_AVAILABLE: return None
    
    # Map ComfyUI selection to SDK constants
    threshold = types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE # Default
    
    if filter_level == "block_most":
        threshold = types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE
    elif filter_level == "block_some":
        threshold = types.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
    elif filter_level == "block_few":
        threshold = types.HarmBlockThreshold.BLOCK_ONLY_HIGH
    elif filter_level == "block_none":
        threshold = types.HarmBlockThreshold.BLOCK_NONE
        
    # Apply to all categories
    categories = [
        types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        types.HarmCategory.HARM_CATEGORY_HARASSMENT
    ]
    
    return [types.SafetySetting(category=cat, threshold=threshold) for cat in categories]

# --- Dynamic Model Fetching (KEEPING CURRENT TO SUPPORT IMAGEN 4) ---

DEFAULT_VEO_MODELS = [
    "veo-2.0-generate-preview",
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview"
]
DEFAULT_IMAGEN_MODELS = [
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-ultra-generate-001",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image-preview",
    "imagen-3.0-generate-001",
    "imagen-3.0-fast-generate-001",
]

def get_project_id(node_pid=None):
    if node_pid and node_pid.strip(): return node_pid.strip()
    try:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f).get("GOOGLE_PROJECT_ID", "")
    except: pass
    return os.environ.get("GOOGLE_PROJECT_ID", "")

# --- Dynamic Model Fetching ---

DEFAULT_VEO_MODELS = [
    "veo-2.0-generate-preview",
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview"
]
DEFAULT_IMAGEN_MODELS = [
    "imagen-4.0-generate-001",
    "imagen-4.0-fast-generate-001",
    "imagen-4.0-ultra-generate-001",
    "gemini-3-pro-image-preview",
    "gemini-2.5-flash-image-preview",
    "imagen-3.0-generate-001",
    "imagen-3.0-fast-generate-001",
]
DEFAULT_VERTEX_LOCATIONS = [
    "us-central1", "europe-west4", "asia-southeast1", "us-east4", "us-west1", "northamerica-northeast1"
]

def fetch_vertex_info(project_id):
    """
    Attempts to fetch available Vertex AI locations and models using gcloud CLI if available.
    """
    locs = []
    models = []
    
    if not project_id: return locs, models
    
    # Try fetching Locations via gcloud
    try:
        import subprocess
        # Check locations (this is a generic call to see available compute regions for AI Platform)
        # However, listing actual Veo availability is harder. We will list common regions.
        # Ideally: gcloud ai locations list --project=... --format="value(name)"
        # But this requires 'aiplatform.locations.list' permission.
        cmd = ["gcloud", "ai", "locations", "list", f"--project={project_id}", "--format=value(name)"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            found = [l.strip() for l in result.stdout.split('\n') if l.strip()]
            if found:
                 # Filter to known likely GenAI regions to avoid clutter, or trust gcloud?
                 # gcloud returns all regions (us-central1, us-east1, etc). Not all have Veo.
                 # We'll merge with defaults to ensure coverage.
                 locs = sorted(list(set(found)))
    except: pass

    # Try fetching Models via gcloud (Model Garden / Publisher Models?)
    # gcloud ai models list --region=us-central1 ... lists user models.
    # To list Publisher models: gcloud ai models list --project=... (complex)
    # We will skip model fetching via gcloud for now as it's messy.
    
    return locs, models

def fetch_models():
    # 1. Fetch from Gemini API (AI Studio)
    key = get_api_key()
    veo_studio = []
    imagen_studio = []
    
    if key:
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={key}"
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"].replace("models/", "") for m in data.get("models", [])]
                
                veo_studio = [m for m in models if "veo" in m]
                
                for m in models:
                    if "image" not in m: continue
                    if "experimental" in m or "embedding" in m or "aqa" in m or "tts" in m or "audio" in m or "lite" in m or "computer-use" in m: continue
                    if "gemini-3" in m or "gemini-2.5" in m or "imagen-3" in m or "imagen-4" in m:
                        imagen_studio.append(m)
        except: pass

    # 2. Fetch/Merge from Vertex Info (optional)
    pid = get_project_id()
    vertex_locs, vertex_models = fetch_vertex_info(pid)
    
    # Combine
    final_veo = sorted(list(set(DEFAULT_VEO_MODELS + veo_studio)))
    final_imagen = sorted(list(set(DEFAULT_IMAGEN_MODELS + imagen_studio)))
    final_locs = sorted(list(set(DEFAULT_VERTEX_LOCATIONS + vertex_locs)))
    
    return final_veo, final_imagen, final_locs

VEO_MODELS, IMAGEN_MODELS, VERTEX_LOCATIONS = fetch_models()

# Filter for models that support editing (Imagen only, Gemini usually implies generation only)
EDITING_MODELS = [m for m in IMAGEN_MODELS if "imagen" in m.lower()]
# Always include known working fallback models for editing (Imagen 3)
# logic: prepend them if missing, don't rely on list being empty
FORCED_MODELS = ["imagen-3.0-generate-001", "imagen-3.0-capability-001"]
for fm in reversed(FORCED_MODELS):
    if fm not in EDITING_MODELS:
        EDITING_MODELS.insert(0, fm)

# --- Main Nodes ---

# [BACKUP] ComfyG_NanoBatch
class ComfyG_NanoBatch:
    def __init__(self): pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",),
            },
            "optional": {
                # Dynamic inputs will be added by JS
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("batch_images",)
    FUNCTION = "batch_images"
    CATEGORY = "TimNodes/Google"
    
    def batch_images(self, **kwargs):
        # Collect all images from kwargs
        images = []
        
        # Natural Sort: image_1, image_2, ... image_10
        def natural_keys(text):
            parts = text.split('_')
            return int(parts[-1]) if parts[-1].isdigit() else 0
            
        sorted_keys = sorted([k for k in kwargs.keys() if k.startswith("image_")], key=natural_keys)
        print(f"TimNodes Batcher: Processing {len(sorted_keys)} inputs: {sorted_keys}")
        
        for k in sorted_keys:
            v = kwargs[k]
            if isinstance(v, torch.Tensor):
                images.append(v)
        
        if not images:
            return (torch.zeros((1, 64, 64, 3)),)

        # Resize all to match the first image
        ref_image = images[0]
        ref_h, ref_w = ref_image.shape[1], ref_image.shape[2]
        processed_images = []
        
        print(f"TimNodes Batcher: Reference Size: {ref_w}x{ref_h}")
        
        for i, img in enumerate(images):
            # Handle batch dimension if input is already a batch
            if len(img.shape) == 4:
                 for j in range(img.shape[0]):
                     sub_img = img[j].unsqueeze(0) # [1, H, W, C]
                     if sub_img.shape[1] != ref_h or sub_img.shape[2] != ref_w:
                         print(f"TimNodes Batcher: Resizing input {i} (sub {j}) from {sub_img.shape[1]}x{sub_img.shape[2]} to {ref_h}x{ref_w}")
                         sub_img = comfy.utils.common_upscale(sub_img.movedim(-1, 1), ref_w, ref_h, "bilinear", "center").movedim(1, -1)
                     processed_images.append(sub_img)
            else:
                # Should not happen in ComfyUI usually, but handle [H, W, C]
                sub_img = img.unsqueeze(0)
                if sub_img.shape[1] != ref_h or sub_img.shape[2] != ref_w:
                     print(f"TimNodes Batcher: Resizing input {i} from {sub_img.shape[1]}x{sub_img.shape[2]} to {ref_h}x{ref_w}")
                     sub_img = comfy.utils.common_upscale(sub_img.movedim(-1, 1), ref_w, ref_h, "bilinear", "center").movedim(1, -1)
                processed_images.append(sub_img)
                
        if not processed_images:
            return (torch.zeros((1, 64, 64, 3)),)
            
        # Return concatenated, clamped, and contiguous tensor
        return (torch.cat(processed_images, dim=0).contiguous().clamp(0.0, 1.0),)


# [CURRENT] ComfyG_NanoBatchMask (NEW)
class ComfyG_NanoBatchMask:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
            },
            "optional": {
                # Dynamic inputs will be added by JS
            }
        }

    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("batch_masks",)
    FUNCTION = "batch_masks"
    CATEGORY = "TimNodes/Google"

    def batch_masks(self, **kwargs):
        masks = []
        # Natural sort keys
        def natural_keys(text):
            parts = text.split('_')
            return int(parts[-1]) if parts[-1].isdigit() else 0
            
        sorted_keys = sorted([k for k in kwargs.keys() if k.startswith("mask_")], key=natural_keys)
        print(f"TimNodes Mask Batcher: Processing {len(sorted_keys)} inputs: {sorted_keys}")

        curr_res = None

        for k in sorted_keys:
            v = kwargs[k]
             # ComfyUI masks are often [H, W] or [1, H, W]
             # We want [1, H, W] for concatenation
            if isinstance(v, torch.Tensor):
                if len(v.shape) == 2:
                    v = v.unsqueeze(0) # [H, W] -> [1, H, W]
                elif len(v.shape) == 3:
                     # Could be [B, H, W] - handle each
                     pass
                
                # Check resolution
                if not masks:
                     curr_res = (v.shape[-2], v.shape[-1])
                
                # Resize if needed
                if v.shape[-2] != curr_res[0] or v.shape[-1] != curr_res[1]:
                     # Resize mask using nearest neighbor to preserve hard edges
                     # Format for interpolate needs [B, C, H, W] -> [B, 1, H, W]
                     v_in = v.unsqueeze(1) 
                     v_out = torch.nn.functional.interpolate(v_in, size=curr_res, mode="nearest")
                     v = v_out.squeeze(1)

                masks.append(v)

        if not masks:
            return (torch.zeros((1, 64, 64)),)
            
        return (torch.cat(masks, dim=0),)


# [BACKUP] ComfyG_Veo
class ComfyG_Veo:
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.get_output_directory(), "temp_TimNodes_Veo")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic shot of..."}),
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "model": (VEO_MODELS, {"default": VEO_MODELS[0] if VEO_MODELS else ""}),
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "16:9"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "duration": (["4", "6", "8"], {"default": "6"}),
                "fps": (["24", "30", "60"], {"default": "24"}),

                "person_generation": (["dont_allow", "allow"], {"default": "allow"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "first_frame": ("IMAGE",), 
                "last_frame": ("IMAGE",), 
                "ingredients": ("IMAGE",),
                "api_key": ("STRING", {"multiline": False, "default": ""}),
            }
        }
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path")
    FUNCTION = "generate_video"
    CATEGORY = "TimNodes/Google"

    def generate_video(self, prompt, negative_prompt, model, aspect_ratio, duration, fps, person_generation, enhance_prompt, generate_audio, seed, resolution="720p", api_key="", first_frame=None, last_frame=None, ingredients=None):
        key = get_api_key(api_key)
        if not key: raise ValueError("API Key is missing.")

        print(f"TimNodes Veo: Generating with '{model}' (SDK)...")
        client = genai.Client(api_key=key)

        # --- HELPERS ---
        def to_sdk_image(tensor):
            if tensor is None: return None
            # Handle Batch vs Single
            img_np = tensor.cpu().numpy()
            if len(img_np.shape) == 4: img_np = img_np[0] # Take first from batch [B,H,W,C]
            img_np = img_np.squeeze() # [H,W,C]
            
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            with io.BytesIO() as bio:
                img_pil.save(bio, format="PNG")
                return types.Image(image_bytes=bio.getvalue(), mime_type="image/png")

        def tensor_to_reference_images(tensor):
            # Convert [B,H,W,C] tensor to list of VideoGenerationReferenceImage
            if tensor is None: return None
            refs = []
            img_np = tensor.cpu().numpy() # [B,H,W,C] usually
            if len(img_np.shape) == 3: img_np = img_np[None, ...] # unsqueeze batch if [H,W,C]
            
            for i in range(img_np.shape[0]):
                slice_np = img_np[i]
                img_pil = Image.fromarray((slice_np * 255).astype(np.uint8))
                with io.BytesIO() as bio:
                    img_pil.save(bio, format="PNG")
                    img_data = types.Image(image_bytes=bio.getvalue(), mime_type="image/png")
                    try:
                        # Try struct
                        ref = types.VideoGenerationReferenceImage(image=img_data)
                    except:
                        # Fallback dict
                        ref = {"image": img_data}
                    refs.append(ref)
            return refs

        # --- PREPARE INPUTS ---
        img_start = None
        img_end = None
        ref_imgs = None
        
        if ingredients is not None:
            print("TimNodes: Mode = Ingredients (User requested Override)")
            ref_imgs = tensor_to_reference_images(ingredients)
        else:
            img_start = to_sdk_image(first_frame)
            img_end = to_sdk_image(last_frame)
            if img_start: print("TimNodes: Mode = Image-to-Video")
            if img_end: print("TimNodes: Mode = Interpolation")

        # --- PREPARE CONFIG ---
        config_args = {
            "aspect_ratio": aspect_ratio,
        }
        
        if resolution and resolution != "default": config_args["resolution"] = resolution
        
        # FPS logic
        if fps: 
            try: config_args["fps"] = int(fps) 
            except: pass
        
        # Duration logic
        is_preview = "preview" in model.lower()
        if duration and not is_preview:
             try: config_args["duration_seconds"] = int(duration)
             except: pass
             
        if img_end and not ref_imgs: config_args["last_frame"] = img_end
        if ref_imgs: config_args["reference_images"] = ref_imgs
        if negative_prompt: config_args["negative_prompt"] = negative_prompt
        if enhance_prompt: config_args["enhance_prompt"] = True
        
        # Audio
        if generate_audio: config_args["generate_audio"] = True
        
        # Person Generation (Map allow -> allow_adult)
        if person_generation:
            if person_generation == "allow":
                config_args["person_generation"] = "allow_adult"
            else:
                config_args["person_generation"] = person_generation
                
        if seed is not None and seed > 0: config_args["seed"] = seed

        # --- ATTEMPT GENERATION (WITH RETRY) ---
        max_retries = 10
        current_config = config_args.copy()
        operation = None
        
        for attempt in range(max_retries):
            try:
                vid_config = types.GenerateVideosConfig(**current_config)
                # print(f"TimNodes: Params: {list(current_config.keys())}") # Reduce spam?
                
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=img_start,
                    config=vid_config
                )
                break # Success
                
            except Exception as e:
                err_msg = str(e)
                # Detect "Not Supported" logic
                rejected_key = None
                if "seed" in err_msg and "not supported" in err_msg: rejected_key = "seed"
                elif "fps" in err_msg and "not supported" in err_msg: rejected_key = "fps"
                elif "resolution" in err_msg and "not supported" in err_msg: rejected_key = "resolution"
                elif "person_generation" in err_msg: rejected_key = "person_generation"
                elif "enhance_prompt" in err_msg or "enhancePrompt" in err_msg: rejected_key = "enhance_prompt"
                elif "generate_audio" in err_msg: rejected_key = "generate_audio"

                if rejected_key and rejected_key in current_config:
                    print(f"TimNodes Warning: Removing rejected '{rejected_key}'...")
                    current_config.pop(rejected_key)
                    if attempt == max_retries - 1: raise e
                else:
                    raise Exception(f"Veo Request Failed: {err_msg}")

        # --- POLLING ---
        if not operation: raise Exception("Failed to start operation.")
        print(f"TimNodes Veo: Polling {operation.name}...")
        
        while not operation.done:
            time.sleep(5)
            operation = client.operations.get(operation)
        
        # --- PROCESS RESULT (ROBUST) ---
        if hasattr(operation, "error") and operation.error:
             raise Exception(f"Veo API Operation Error: {operation.error}")

        result = operation.result
        if result is None:
             debug_info = f"Status: {operation.status}" if hasattr(operation, 'status') else "Unknown"
             raise ValueError(f"API returned NO Result. Operation Status: {debug_info}.")

        if not hasattr(result, "generated_videos") or not result.generated_videos:
             print("TimNodes Error: No Videos.")
             rai_reason = ""
             if hasattr(result, "rai_media_filtered_reasons") and result.rai_media_filtered_reasons:
                 rai_reason = f"Safety Filter (RAI): {result.rai_media_filtered_reasons}"
             elif hasattr(result, "rai_media_filtered_count") and result.rai_media_filtered_count > 0:
                 rai_reason = f"Safety Filter (RAI): {result.rai_media_filtered_count} items filtered."
             
             error_msg = f"No videos generated. {rai_reason}" if rai_reason else "No videos generated (Safety/Quota)."
             raise ValueError(error_msg)
             
        vid_uri = result.generated_videos[0].video.uri
        mock_res = {"videoUri": vid_uri}
        return self._process_video(mock_res, str(result), key)
    def _process_video(self, result, full_log, api_key):
        video_uri = None
        if isinstance(result, dict):
            if "generateVideoResponse" in result:
                samples = result["generateVideoResponse"].get("generatedSamples", [])
                if samples and "video" in samples[0]:
                    vid = samples[0]["video"]
                    video_uri = vid.get("uri") or vid.get("videoUri")
            elif "generated_videos" in result:
                vid_obj = result["generated_videos"][0]
                if "video" in vid_obj:
                    video_uri = vid_obj["video"].get("uri") or vid_obj["video"].get("videoUri")
            elif "videoUri" in result:
                video_uri = result["videoUri"]
        
        if not video_uri:
            match = re.search(r'"uri":\s*"(https://[^"]+)"', full_log)
            if match: video_uri = match.group(1)
            
        if not video_uri:
            raise Exception("Could not find video URI.")

        dl_resp = requests.get(video_uri, headers={"x-goog-api-key": api_key})
        if dl_resp.status_code != 200:
            raise Exception(f"Download Failed ({dl_resp.status_code})")
            
        filename = f"veo_{int(time.time())}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "wb") as f: f.write(dl_resp.content)

        try:
            reader = imageio.get_reader(filepath, 'ffmpeg')
            frames = [im for im in reader]
            frames_np = np.array(frames).astype(np.float32) / 255.0
            images_tensor = torch.from_numpy(frames_np)
        except:
            images_tensor = torch.zeros((1, 64, 64, 3))

        audio_output = None
        try:
            # Robust Audio Extraction: MP4 -> WAV (ffmpeg) -> Tensor
            import imageio_ffmpeg
            import subprocess
            import soundfile as sf
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            wav_path = os.path.splitext(filepath)[0] + ".wav"
            
            # Extract audio to WAV
            # -y: overwrite, -vn: no video, -acodec pcm_s16le: standard wav
            cmd = [ffmpeg_exe, "-y", "-i", filepath, "-vn", "-acodec", "pcm_s16le", wav_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(wav_path):
                data, sample_rate = sf.read(wav_path)
                # soundfile returns [frames, channels], torch expects [channels, frames]
                waveform = torch.from_numpy(data).float()
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0) # Mono to [1, frames]
                else:
                    waveform = waveform.t() # [frames, channels] -> [channels, frames]
                
                audio_output = {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}
                
                # Cleanup
                try: os.remove(wav_path)
                except: pass
            else:
                print("TimNodes Warning: No audio track found or extraction failed.")

        except Exception as e:
            print(f"TimNodes Warning: Audio extraction failed: {e}")
            pass

        return (images_tensor, audio_output, filepath)


# [BACKUP] ComfyG_NanoBanana
class ComfyG_NanoBanana:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": "A futuristic banana..."}),
                "model": (IMAGEN_MODELS, {"default": IMAGEN_MODELS[0] if IMAGEN_MODELS else ""}),
                "aspect_ratio": (["default", "1:1", "16:9", "9:16", "3:4", "4:3"], {"default": "default"}),
                "resolution": (["default", "1K", "2K", "4K"], {"default": "default"}),
                "response_modalities": (["IMAGE", "TEXT", "IMAGE+TEXT"], {"default": "IMAGE+TEXT"}),
                "number_of_images": ("INT", {"default": 1, "min": 1, "max": 4}),
                "safety_filter": (["block_most", "block_some", "block_few", "block_none"], {"default": "block_none"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 4294967295}),
            },
            "optional": {
                "api_key": ("STRING", {"multiline": False, "default": ""}),
                "input_images": ("IMAGE",),
                "files_input": ("STRING", {"forceInput": True}),
                "compress_images": ("BOOLEAN", {"default": True, "tooltip": "Resize to 1024px and compress to JPEG (80) to prevent API errors with large batches."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "text_output")
    FUNCTION = "generate_image"
    CATEGORY = "TimNodes/Google"

    def generate_image(self, prompt, model, aspect_ratio, resolution, response_modalities, number_of_images, safety_filter, seed, compress_images=True, api_key="", input_images=None, files_input=""):
        key = get_api_key(api_key)
        if not key: raise ValueError("API Key is missing.")

        print(f"TimNodes NanoBanana: Generating with {model}...")
        
        # --- SDK IMPLEMENTATION FOR GEMINI & IMAGEN 3/4 ---
        is_modern_model = "gemini" in model.lower() or "imagen-3" in model.lower() or "imagen-4" in model.lower()
        if is_modern_model and SDK_AVAILABLE:
            try:
                client = genai.Client(api_key=key)
                
                # Map modalities
                modalities = ["TEXT", "IMAGE"] # Default
                if response_modalities == "IMAGE": modalities = ["IMAGE"]
                elif response_modalities == "TEXT": modalities = ["TEXT"]

                # --- SPLIT LOGIC: SIMPLE (2.5) vs ADVANCED (3.0/4.0) ---
                use_simple_content = "gemini-2.5" in model or "flash" in model
                
                if use_simple_content:
                    # === PATH A: SIMPLE CONTENT (Gemini 2.5) - STATELESS ===
                    # Uses client.models.generate_content() with a Simple List [str, PIL.Image...]
                    print(f"TimNodes: Using Simple Content Path for {model} (Stateless)")
                    
                    simple_contents = [prompt]
                    
                    # 1. Inline Images
                    if input_images is not None:
                        # DEBUG LOGGING
                        print(f"TimNodes DEBUG: Received input_images tensor shape: {input_images.shape}")
                        pil_images = batch_tensor_to_pil_list(input_images)
                        print(f"TimNodes DEBUG: Converted to {len(pil_images)} PIL images.")
                        
                        if compress_images:
                            print(f"TimNodes: Compressing {len(pil_images)} images for API stability...")
                            for img in pil_images:
                                # Resize (Max 1024)
                                max_size = 1024
                                if img.width > max_size or img.height > max_size:
                                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                                # Compress (JPEG 80)
                                buffered = io.BytesIO()
                                img.save(buffered, format="JPEG", quality=80, optimize=True)
                                simple_contents.append(Image.open(buffered))
                        else:
                            print(f"TimNodes: Sending {len(pil_images)} images in Full Quality.")
                            simple_contents.extend(pil_images)
                        
                        print(f"TimNodes DEBUG: Final contents length: {len(simple_contents)}")
                        
                    # 2. Files Input (Load as PIL)
                    if files_input and files_input.strip():
                        file_paths = [f.strip() for f in files_input.split('\n') if f.strip()]
                        for fp in file_paths:
                            fp = fp.strip('"').strip("'")
                            if os.path.exists(fp):
                                try:
                                    img = Image.open(fp)
                                    if img.mode != "RGB": img = img.convert("RGB")
                                    simple_contents.append(img)
                                    print(f"TimNodes: Loaded file '{fp}' as image.")
                                except Exception as e:
                                    print(f"TimNodes Error loading file {fp}: {e}")
                    
                    # Generate (Stateless)
                    try:
                        # CONFIG SANITIZATION:
                        img_cfg = None
                        img_cfg = None
                        if "IMAGE" in modalities:
                             # Map "default" to None to let API decide
                             ar_param = aspect_ratio if aspect_ratio != "default" else None
                             
                             # Map Resolution Strings to Values
                             res_param = resolution if resolution != "default" else None
                             print(f"TimNodes DEBUG: Config -> Aspect: {ar_param}, Res: {res_param} (Type: {type(res_param)})")
                             
                             img_cfg = types.ImageConfig(aspect_ratio=ar_param, image_size=res_param)

                        config = types.GenerateContentConfig(
                            response_modalities=modalities,
                            image_config=img_cfg,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            safety_settings=_get_gemini_safety_settings(safety_filter)
                        )
                        response = client.models.generate_content(model=model, contents=simple_contents, config=config)
                        if not response.candidates: raise ValueError("No candidates returned.")
                    except Exception as e:
                        print(f"TimNodes Warning: Simple Content failed ({e}). Retrying with BLOCK_NONE...")
                        fallback_config = types.GenerateContentConfig(
                            safety_settings=_get_gemini_safety_settings("block_none")
                        )
                        response = client.models.generate_content(model=model, contents=simple_contents, config=fallback_config)

                else:
                    # === PATH B: MODERN CONTENT (Gemini 3.0 / Imagen 4.0) - OPTIMIZED ===
                    print(f"TimNodes: Using Modern Content Path for {model}")
                    
                    parts = [types.Part.from_text(text=prompt)]
                    
                    # 1. Inline Images
                    if input_images is not None:
                        pil_images = batch_tensor_to_pil_list(input_images)
                        for img in pil_images:
                            if compress_images:
                                # Resize (Max 1024)
                                max_size = 1024
                                if img.width > max_size or img.height > max_size:
                                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                                # Compress (JPEG 80)
                                buffered = io.BytesIO()
                                img.save(buffered, format="JPEG", quality=80, optimize=True)
                                img_bytes = buffered.getvalue()
                                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"))
                            else:
                                # Full Quality (PNG)
                                buffered = io.BytesIO()
                                img.save(buffered, format="PNG")
                                img_bytes = buffered.getvalue()
                                parts.append(types.Part.from_bytes(data=img_bytes, mime_type="image/png"))
                        
                        msg = "Compressed" if compress_images else "Full Quality"
                        print(f"TimNodes: Added {len(pil_images)} images ({msg}) to parts.")

                    # 2. Files Input
                    if files_input and files_input.strip():
                        file_paths = [f.strip() for f in files_input.split('\n') if f.strip()]
                        for fp in file_paths:
                            fp = fp.strip('"').strip("'")
                            if os.path.exists(fp):
                                try:
                                    print(f"TimNodes: Uploading file '{fp}'...")
                                    uploaded_file = client.files.upload(path=fp)
                                    parts.append(types.Part.from_uri(file_uri=uploaded_file.uri, mime_type=uploaded_file.mime_type))
                                except Exception as e:
                                    print(f"TimNodes Error uploading file {fp}: {e}")

                    contents = [types.Content(role="user", parts=parts)]
                    
                    # Generate (Content)
                    try:
                        # CONFIG SANITIZATION
                        img_cfg = None
                        img_cfg = None
                        if "IMAGE" in modalities:
                             # Map "default" to None to let API decide
                             ar_param = aspect_ratio if aspect_ratio != "default" else None
                             
                             # Map Resolution Strings to Values
                             res_param = resolution if resolution != "default" else None
                             print(f"TimNodes DEBUG: Config -> Aspect: {ar_param}, Res: {res_param} (Type: {type(res_param)})")
                             
                             img_cfg = types.ImageConfig(aspect_ratio=ar_param, image_size=res_param)

                        config = types.GenerateContentConfig(
                            response_modalities=modalities,
                            image_config=img_cfg,
                            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                            safety_settings=_get_gemini_safety_settings(safety_filter)
                        )
                        response = client.models.generate_content(model=model, contents=contents, config=config)
                        if not response.candidates: raise ValueError("No candidates returned.")
                    except Exception as e:
                        # Smart Fallback
                        err_str = str(e).upper()
                        if "BLOCKEDREASON.SAFETY" in err_str: raise e
                        
                        print(f"TimNodes Warning: Generation failed ({e}). Retrying with BLOCK_NONE...")
                        fallback_config = types.GenerateContentConfig(
                            safety_settings=_get_gemini_safety_settings("block_none")
                        )
                        response = client.models.generate_content(model=model, contents=contents, config=fallback_config)

                # --- PROCESS RESPONSE (Common) ---
                final_images = []
                final_text = ""
                
                if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                    final_text += f"\n[SAFETY FEEDBACK]: {response.prompt_feedback}\n"
                
                if response.candidates:
                    cand = response.candidates[0]
                    if hasattr(cand, 'finish_reason'):
                         final_text += f"\n[FINISH REASON]: {cand.finish_reason}\n"

                    if cand.content and cand.content.parts:
                        for part in cand.content.parts:
                            if part.text: final_text += part.text
                            if part.inline_data:
                                try:
                                    img_data = part.inline_data.data
                                    img = Image.open(io.BytesIO(img_data))
                                    if img.mode != "RGB": img = img.convert("RGB")
                                    img_t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
                                    final_images.append(img_t)
                                except Exception as e:
                                    print(f"TimNodes Error processing image part: {e}")
                else:
                    print("TimNodes Debug: No candidates returned.")
                
                if not final_images:
                    error_msg = "Gemini blocked the image generation."
                    if "SAFETY" in final_text or "block" in final_text.lower():
                         error_msg += f" Reason: Safety Violation. {final_text}"
                    else:
                         error_msg += f" Details: {final_text}"
                    raise ValueError(error_msg)
                
                if len(final_images) > 1: return (torch.cat(final_images, dim=0), final_text)
                return (final_images[0], final_text)

            except Exception as e:
                print(f"TimNodes Exception: {e}")
                raise e
        
        # --- FALLBACK / REST API IMPLEMENTATION ---
        else:
            headers = {"Content-Type": "application/json", "x-goog-api-key": key}
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:predict"
            
            instance = {"prompt": prompt}
            has_image = False
            if input_images is not None:
                instance["image"] = {"bytesBase64Encoded": tensor_to_base64(input_images), "mimeType": "image/png"}
                has_image = True

            payload = {
                "instances": [instance],
                "parameters": {
                    "sampleCount": number_of_images,
                    "aspectRatio": aspect_ratio,
                    "safetySetting": safety_filter, 
                }
            }

            try:
                response = requests.post(url, headers=headers, json=payload)
                
                if response.status_code == 400 and has_image:
                     instance.pop("image")
                     payload["instances"] = [instance]
                     response = requests.post(url, headers=headers, json=payload)

                if response.status_code != 200:
                    raise Exception(f"Imagen Error ({response.status_code}): {response.text}")
                resp_json = response.json()
                images = []
                for p in resp_json.get("predictions", []):
                    if "bytesBase64Encoded" in p: b64 = p["bytesBase64Encoded"]
                    elif "image" in p: b64 = p["image"]["bytesBase64Encoded"]
                    else: continue
                    img = Image.open(io.BytesIO(base64.b64decode(b64)))
                    images.append(torch.from_numpy(np.array(img.convert("RGB")).astype(np.float32) / 255.0))
                
                if not images: raise Exception("No images found.")
                if len(images) > 1: return (torch.stack(images), "Success")
                return (images[0].unsqueeze(0), "Success")

            except Exception as e:
                raise Exception(f"Imagen Error: {e}")


# [BACKUP] ComfyG_Model_Scanner
class ComfyG_Model_Scanner:
    def __init__(self): pass
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"api_key": ("STRING", {"multiline": False, "default": ""})}}
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model_list",)
    FUNCTION = "scan_models"
    CATEGORY = "Google2Comfy"

    def scan_models(self, api_key):
        if not api_key: return ("Please enter API Key",)
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
            resp = requests.get(url)
            data = resp.json()
            models = [m["name"].replace("models/", "") for m in data.get("models", [])]
            return ("\n".join(models),)
        except Exception as e: return (f"Scan Failed: {e}",)

# [BACKUP] ComfyG_FolderScanner
class ComfyG_FolderScanner:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "folder_path": ("STRING", {"default": "/path/to/images", "multiline": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_paths",)
    FUNCTION = "scan_folder"
    CATEGORY = "Google2Comfy"

    def scan_folder(self, folder_path):
        if not os.path.isdir(folder_path):
            return ("",)
            
        valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        files = []
        try:
            for f in sorted(os.listdir(folder_path)):
                name, ext = os.path.splitext(f)
                if ext.lower() in valid_exts:
                    files.append(os.path.join(folder_path, f))
        except Exception as e:
            print(f"TimNodes Scanner Error: {e}")
            
        return ("\n".join(files),)

# [CURRENT] ComfyG_ProBanana (NEW)
class ComfyG_ProBanana:
    def __init__(self): pass

    @classmethod
    def INPUT_TYPES(s):
        # Dynamically fetch models if possible, else fallback
        # Always put PREFERRED editing models at top
        model_list = list(EDITING_MODELS) # Copy default list
        
        return {
            "required": {
                "reference_image": ("IMAGE",),
                "prompt": ("STRING", {"multiline": True, "default": "Describe the edit..."}),
                
                # Model selection
                "model": (model_list, {"default": model_list[0] if model_list else "gemini-1.5-pro-001"}), 
                "vertex_project_id": ("STRING", {"default": "gen-lang-client-0367144707"}),
                "vertex_location": (["us-central1", "europe-west4", "asia-northeast1", "us-west1"], {"default": "us-central1"}),
                
                # Edit Config
                # Edit Config
                "edit_mode": (["EDIT_MODE_DEFAULT", "EDIT_MODE_INPAINT_INSERTION", "EDIT_MODE_INPAINT_REMOVAL", "EDIT_MODE_BACKGROUND_SWAP", "EDIT_MODE_CONTROLLED_EDITING", "EDIT_MODE_STYLE"], {"default": "EDIT_MODE_DEFAULT"}),
                "guidance_scale": ("FLOAT", {"default": 60.0, "min": 0.0, "max": 500.0, "step": 0.5}),
                "safety_filter": (["block_most", "block_some", "block_few", "block_none"], {"default": "block_some"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK",),
                "control_image": ("IMAGE",),
                "control_type": (["CONTROL_TYPE_DEFAULT", "CONTROL_TYPE_CANNY", "CONTROL_TYPE_SCRIBBLE", "CONTROL_TYPE_FACE_MESH", "CONTROL_TYPE_SEGMENTATION"],),
                "style_reference": ("IMAGE",),
                "subject_reference": ("IMAGE",),
                "subject_description": ("STRING", {"multiline": False}),
                "negative_prompt": ("STRING", {"multiline": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("edited_image",)
    FUNCTION = "edit_image"
    CATEGORY = "Google2Comfy"

    def edit_image(self, reference_image, prompt, model, vertex_project_id, vertex_location, edit_mode, guidance_scale, safety_filter, seed, 
                   mask=None, control_image=None, control_type="CONTROL_TYPE_DEFAULT", style_reference=None, subject_reference=None, subject_description=None, negative_prompt=None):
        
        if not SDK_AVAILABLE:
            raise ImportError("Google GenAI SDK not installed. Run 'pip install google-genai'")

        # Validate Location
        if vertex_location != "us-central1":
            print(f"TimNodes Warning: Vertex AI Image Editing is primarily supported in 'us-central1'. You selected '{vertex_location}'. Attempting anyway...")

        # Initialize Vertex Client
        print(f"TimNodes ProBanana: Initializing Vertex AI Client (Project: {vertex_project_id}, Loc: {vertex_location})")
        try:
            client = genai.Client(vertexai=True, project=vertex_project_id, location=vertex_location)
        except Exception as e:
            raise ValueError(f"Failed to initialize Vertex AI Client: {e}. Ensure ADC is set up: 'gcloud auth application-default login'")

        print(f"TimNodes ProBanana: Editing with {model} in {edit_mode}...")
        
        # Prepare Inputs
        batch_size = reference_image.shape[0]
        # FIX: Ensure we have a list of PIL images
        ref_pils = batch_tensor_to_pil_list(reference_image)
        
        mask_pils = [None]*batch_size
        if mask is not None:
             # Masks handling: ensure robust shape
             if len(mask.shape) == 2: mask = mask.unsqueeze(0) # [H,W] -> [1,H,W]
             if len(mask.shape) == 3: 
                if mask.shape[0] != batch_size and mask.shape[0] == 1:
                    mask = mask.repeat(batch_size, 1, 1) # Broadcast
             mask_pils = batch_tensor_to_pil_list(mask) # Now robust
             if len(mask_pils) == 1 and batch_size > 1: mask_pils = mask_pils * batch_size # List broadcast
        
        ctrl_pils = batch_tensor_to_pil_list(control_image)
        if len(ctrl_pils) == 1 and batch_size > 1: ctrl_pils = ctrl_pils * batch_size
        
        style_pils = batch_tensor_to_pil_list(style_reference)
        if len(style_pils) == 1 and batch_size > 1: style_pils = style_pils * batch_size
        
        sub_pils = batch_tensor_to_pil_list(subject_reference)
        if len(sub_pils) == 1 and batch_size > 1: sub_pils = sub_pils * batch_size

        final_images = []
        out_tensors = []

        try:
            # Check Edit Mode requirements
            is_inpaint = edit_mode in ["EDIT_MODE_INPAINT_INSERTION", "EDIT_MODE_INPAINT_REMOVAL", "EDIT_MODE_BACKGROUND_SWAP"]
            if is_inpaint and mask is None:
                if edit_mode != "EDIT_MODE_BACKGROUND_SWAP": 
                    pass 

            # Map Edit Mode string to Enum
            em_enum = None
            if edit_mode != "EDIT_MODE_DEFAULT":
                em_enum = getattr(types.EditMode, edit_mode, None)

            # Helper to create SDK Image type
            def get_image_wrapper(pil_img, fmt="PNG"):
                if pil_img is None: return None
                with io.BytesIO() as bio:
                    pil_img.save(bio, format=fmt)
                    # FIX: SDK uses camelCase arguments in constructor!
                    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
                    return types.Image(imageBytes=bio.getvalue(), mimeType=mime)

            for i in range(batch_size):
                curr_ref = ref_pils[i]
                curr_mask = mask_pils[i] if i < len(mask_pils) else None
                curr_control = ctrl_pils[i] if i < len(ctrl_pils) else None
                curr_style = style_pils[i] if i < len(style_pils) else None
                curr_subject = sub_pils[i] if i < len(sub_pils) else None
                
                # Check control image requirement
                if edit_mode == "EDIT_MODE_CONTROLLED_EDITING" and curr_control is None:
                     raise ValueError(f"Mode '{edit_mode}' requires a CONTROL_IMAGE input.")

                # Build Config Args
                config_args = {
                    "guidance_scale": guidance_scale,
                    "safety_filter_level": _get_gemini_safety_settings(safety_filter)[0].threshold,
                    "seed": seed + i,
                    "negative_prompt": negative_prompt if negative_prompt else None,
                    "number_of_images": 1
                }
                if em_enum: config_args["edit_mode"] = em_enum
                
                ra_config = types.EditImageConfig(**config_args)
                
                # Helper for Padding
                def pad_to_square(img, fill_color=(0,0,0)):
                    w, h = img.size
                    if w == h: return img, (0, 0, w, h)
                    max_dim = max(w, h)
                    new_img = Image.new(img.mode, (max_dim, max_dim), fill_color)
                    left = (max_dim - w) // 2
                    top = (max_dim - h) // 2
                    new_img.paste(img, (left, top))
                    return new_img, (left, top, w, h)

                # Construct Reference List
                ref_id_counter = 1
                reference_imgs_list = []
                
                # PRE-CHECK: Padding Requirement
                # If we have (Base + Mask + Style/Subject) -> That is 3 refs.
                # If image is non-square, API fails.
                # So we must pad everything if we anticipate > 2 refs and image is non-square.
                active_ref_count = 1 # Base
                if curr_mask is not None: active_ref_count += 1
                if curr_control is not None: active_ref_count += 1
                if curr_style is not None: active_ref_count += 1 # Style or Fallback Subject
                if curr_subject is not None: active_ref_count += 1
                
                needs_padding = False
                original_crop_box = None
                
                if active_ref_count > 2 and curr_ref.size[0] != curr_ref.size[1]:
                    print(f"TimNodes Info: Automatically padding {curr_ref.size} image to Square to support {active_ref_count} references.")
                    needs_padding = True
                    curr_ref, original_crop_box = pad_to_square(curr_ref, (0,0,0))
                    if curr_mask: curr_mask, _ = pad_to_square(curr_mask, 0)
                    if curr_control: curr_control, _ = pad_to_square(curr_control, (0,0,0))
                    # Do not pad Style/Subject references as they are not spatially aligned to the canvas
                    # ...unless they are meant to be spatially aligned? 
                    # Subject/Style refs are usually thematic, not spatial. 
                    # BUT Control images ARE spatial. Mask IS spatial. 
                    # So we only pad spatial inputs: Base, Mask, Control.
                
                # 1. Base Image (Raw)
                base_wrapper = get_image_wrapper(curr_ref, fmt="JPEG") 
                if base_wrapper is None or not base_wrapper.image_bytes:
                    raise ValueError(f"Base Reference Image is invalid or has no bytes.")
                print(f"TimNodes DEBUG: Base Image Bytes: {len(base_wrapper.image_bytes)}")
                reference_imgs_list.append(types.RawReferenceImage(reference_id=ref_id_counter, reference_image=base_wrapper))
                ref_id_counter += 1
                
                # 2. Mask
                if curr_mask is not None:
                    if curr_mask.size != curr_ref.size:
                        curr_mask = curr_mask.resize(curr_ref.size, Image.NEAREST)
                    # Always send mask if provided, regardless of explicit 'is_inpaint' mode,
                    # as API may support it in Default mode too.
                    mask_wrapper = get_image_wrapper(curr_mask)
                    print(f"TimNodes DEBUG: Mask Image Bytes: {len(mask_wrapper.image_bytes)}")
                    mask_config = types.MaskReferenceConfig(mask_mode=types.MaskReferenceMode.MASK_MODE_USER_PROVIDED)
                    reference_imgs_list.append(types.MaskReferenceImage(reference_id=ref_id_counter, reference_image=mask_wrapper, config=mask_config))
                    ref_id_counter += 1
                
                # 3. Control
                if edit_mode == "EDIT_MODE_CONTROLLED_EDITING" and curr_control is not None:
                     if curr_control.size != curr_ref.size:
                        curr_control = curr_control.resize(curr_ref.size, Image.BILINEAR)
                     ctrl_wrapper = get_image_wrapper(curr_control)
                     print(f"TimNodes DEBUG: Control Image Bytes: {len(ctrl_wrapper.image_bytes)}")
                     ctrl_type_enum = getattr(types.ControlReferenceType, control_type, types.ControlReferenceType.CONTROL_TYPE_DEFAULT)
                     reference_imgs_list.append(types.ControlReferenceImage(reference_id=ref_id_counter, reference_image=ctrl_wrapper, config=types.ControlReferenceConfig(control_type=ctrl_type_enum)))
                     ref_id_counter += 1

                # 4. Style (ENABLED)
                # 4. Style (ENABLED)
                # 4. Style Handling (with Fallback)
                if curr_style is not None:
                     style_wrapper = get_image_wrapper(curr_style, fmt="JPEG")
                     
                     if edit_mode == "EDIT_MODE_STYLE":
                         # User explicitly chose Style Mode - try standard Style Ref
                         print(f"TimNodes DEBUG: Using Standard Style Reference (Mode: {edit_mode})")
                         reference_imgs_list.append(types.StyleReferenceImage(reference_id=ref_id_counter, reference_image=style_wrapper))
                     else:
                         # Fallback: Map to SubjectReference for broad compatibility in other modes
                         print(f"TimNodes Info: Using Subject Reference Fallback for Style Input (Mode: {edit_mode})")
                         # Use provided description or generic one
                         style_desc = subject_description if subject_description else "Transfer the visual style of this reference image"
                         
                         reference_imgs_list.append(types.SubjectReferenceImage(
                             reference_id=ref_id_counter, 
                             reference_image=style_wrapper,
                             config=types.SubjectReferenceConfig(subject_description=style_desc)
                        ))
                     
                     ref_id_counter += 1
                
                # 5. Subject
                if curr_subject is not None:
                     sub_wrapper = get_image_wrapper(curr_subject)
                     print(f"TimNodes DEBUG: Subject Image Bytes: {len(sub_wrapper.image_bytes)}")
                     reference_imgs_list.append(types.SubjectReferenceImage(reference_id=ref_id_counter, reference_image=sub_wrapper, config=types.SubjectReferenceConfig(subject_description=subject_description)))
                     ref_id_counter += 1

                # Call API
                print(f"TimNodes ProBanana: Processing batch item {i+1}/{batch_size}")
                print(f"TimNodes DEBUG Payload: Mode={edit_mode}, Refs={[type(x) for x in reference_imgs_list]}")

                try:
                    response = client.models.edit_image(
                        model=model,
                        prompt=prompt,
                        reference_images=reference_imgs_list,
                        config=ra_config
                    )
                    
                    if not response.generated_images:
                        raise ValueError("No images generated by Vertex AI.")
                    
                    img_obj = response.generated_images[0].image
                    if not img_obj or not img_obj.image_bytes:
                         raise ValueError("Received response but image content is empty.")
                         
                    img_data = img_obj.image_bytes
                    ret_image = Image.open(io.BytesIO(img_data))
                    ret_image = Image.open(io.BytesIO(img_data))
                    ret_image = ret_image.convert("RGB")
                    
                    # Un-Pad if needed
                    if needs_padding and original_crop_box:
                        l, t, w, h = original_crop_box
                        # Center crop back to original dims
                        ret_image = ret_image.crop((l, t, l+w, t+h))
                        
                    ret_image = np.array(ret_image).astype(np.float32) / 255.0
                    ret_image = torch.from_numpy(ret_image)[None,]
                    out_tensors.append(ret_image)

                except Exception as e:
                    print(f"TimNodes Detailed Error: {e}")
                    # Hint for Style or Control Mode mismatch
                    if "INVALID_ARGUMENT" in str(e):
                         if style_reference is not None and edit_mode != "EDIT_MODE_STYLE":
                              print("TimNodes Hint: You provided a Style Reference but are not in 'EDIT_MODE_STYLE'. This usually causes errors.")
                         if control_image is not None and edit_mode != "EDIT_MODE_CONTROLLED_EDITING":
                              print("TimNodes Hint: You provided a Control Image but are not in 'EDIT_MODE_CONTROLLED_EDITING'.")
                    
                    if "INVALID_ARGUMENT" in str(e) and "generate-001" in model:
                        print(f"TimNodes Hint: You are using '{model}'. Try switching to 'imagen-3.0-capability-001'.")
                    raise e

            if len(out_tensors) > 1:
                return (torch.cat(out_tensors, dim=0),)
            return (out_tensors[0],)

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise ValueError(f"ProBanana Failed: {e}\n(Tip: Check ComfyUI Console for 'TimNodes DEBUG Payload' details.)")






# [NEW] ComfyG_VeoVertexAPI
class ComfyG_VeoVertexAPI(ComfyG_Veo):
    def __init__(self):
        self.output_dir = os.path.join(folder_paths.get_output_directory(), "temp_TimNodes_VeoVertex")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "project_id": ("STRING", {"multiline": False, "default": get_project_id()}),
                "location": (VERTEX_LOCATIONS, {"default": "us-central1"}),
                "prompt": ("STRING", {"multiline": True, "default": "A cinematic shot of..."}),
                
                # We use the global VEO_MODELS list from the file scope
                "model": (VEO_MODELS, {"default": VEO_MODELS[0] if VEO_MODELS else "veo-2.0-generate-preview"}),
                
                "aspect_ratio": (["16:9", "9:16", "1:1", "4:3", "3:4", "21:9"], {"default": "16:9"}),
                "resolution": (["720p", "1080p"], {"default": "720p"}),
                "duration": (["4", "6", "8"], {"default": "6"}),
                "fps": (["24", "30", "60"], {"default": "24"}),
                "person_generation": (["dont_allow", "allow"], {"default": "allow"}),
                "enhance_prompt": ("BOOLEAN", {"default": False}),
                "generate_audio": ("BOOLEAN", {"default": False}),
                "safety_filter": (["block_most", "block_some", "block_few", "block_none"], {"default": "block_none"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1}),
            },
            "optional": {
                "negative_prompt": ("STRING", {"multiline": True, "default": ""}),
                "image_start": ("IMAGE",),
                "image_end": ("IMAGE",),
                
                # Extended Ingredients for Vertex
                "asset_references": ("IMAGE",), 
                "style_references": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "AUDIO", "STRING")
    RETURN_NAMES = ("images", "audio", "video_path")
    FUNCTION = "generate_video_vertex"
    CATEGORY = "TimNodes/Google"

    def generate_video_vertex(self, project_id, location, prompt, model, aspect_ratio, resolution, duration, fps, person_generation, enhance_prompt, generate_audio, safety_filter, seed, negative_prompt="", image_start=None, image_end=None, asset_references=None, style_references=None):
        if not project_id:
             # Try fallback to global config if user cleared the field
             project_id = get_project_id()
             
        if not project_id:
             raise ValueError("Project ID is required for Vertex API usage.")
        
        if not SDK_AVAILABLE:
            raise ImportError("Google GenAI SDK (google-genai) not installed or found.")

        print(f"TimNodes VeoVertex: Initializing Client (Project: {project_id}, Location: {location})...")
        
        # Initialize Vertex Client
        client = genai.Client(vertexai=True, project=project_id, location=location)

        # --- LOCAL HELPERS ---
        def to_sdk_image(tensor):
            if tensor is None: return None
            img_np = tensor.cpu().numpy()
            if len(img_np.shape) == 4: img_np = img_np[0]
            img_np = img_np.squeeze()
            
            # Clamp and Convert
            img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            
            with io.BytesIO() as bio:
                img_pil.save(bio, format="PNG")
                return types.Image(image_bytes=bio.getvalue(), mime_type="image/png")

        def tensor_to_reference_list(tensor, ref_type="ASSET"):
            if tensor is None: return []
            refs = []
            img_np = tensor.cpu().numpy()
            if len(img_np.shape) == 3: img_np = img_np[None, ...] # [B,H,W,C]
            
            for i in range(img_np.shape[0]):
                slice_np = np.clip(img_np[i] * 255, 0, 255).astype(np.uint8)
                img_pil = Image.fromarray(slice_np)
                with io.BytesIO() as bio:
                    img_pil.save(bio, format="PNG")
                    img_data = types.Image(image_bytes=bio.getvalue(), mime_type="image/png")
                    
                    # Try to construct robust reference
                    ref_obj = None
                    try:
                         # Attempt 1: Using dict with type hint if supported by wrapper
                         ref_dict = {"image": img_data}
                         if ref_type == "STYLE":
                             ref_dict["reference_type"] = "STYLE" 
                         else:
                             ref_dict["reference_type"] = "ASSET"
                         
                         ref_obj = types.VideoGenerationReferenceImage(**ref_dict)
                    except:
                         # Fallback: Just Image
                         print(f"TimNodes Warning: Could not set reference_type '{ref_type}', defaulting.")
                         try:
                             ref_obj = types.VideoGenerationReferenceImage(image=img_data)
                         except:
                             ref_obj = {"image": img_data}
                             
                    refs.append(ref_obj)
            return refs

        # --- PREPARE ---
        img_s = to_sdk_image(image_start)
        img_e = to_sdk_image(image_end)
        
        all_refs = []
        all_refs.extend(tensor_to_reference_list(asset_references, "ASSET"))
        all_refs.extend(tensor_to_reference_list(style_references, "STYLE"))
        
        # --- CONFIG ---
        config_args = {
            "aspect_ratio": aspect_ratio,
        }
        
        if resolution and resolution != "default": config_args["resolution"] = resolution
        
        try: config_args["fps"] = int(fps) 
        except: pass
        
        is_preview = "preview" in model.lower()
        if duration and not is_preview:
             try: config_args["duration_seconds"] = int(duration)
             except: pass
             
        if img_e: config_args["last_frame"] = img_e
        if all_refs: config_args["reference_images"] = all_refs
        if negative_prompt: config_args["negative_prompt"] = negative_prompt
        if enhance_prompt: config_args["enhance_prompt"] = True
        if generate_audio: config_args["generate_audio"] = True
        
        if person_generation:
             config_args["person_generation"] = "allow_adult" if person_generation == "allow" else "dont_allow"

        # Note: Veo (GenerateVideosConfig) does not support 'safety_settings' despite initial docs.
        # Removing to fix ValidationError.
        
        if seed > 0: config_args["seed"] = seed

        # --- EXECUTE ---
        print(f"TimNodes VeoVertex: Sending request... Refs: {len(all_refs)}")
        
        max_retries = 5
        operation = None
        current_config = config_args.copy()
        
        for attempt in range(max_retries):
            try:
                vid_config = types.GenerateVideosConfig(**current_config)
                
                operation = client.models.generate_videos(
                    model=model,
                    prompt=prompt,
                    image=img_s,
                    config=vid_config
                )
                break
            except Exception as e:
                err = str(e)
                print(f"TimNodes VeoVertex Error (Attempt {attempt+1}): {err}")
                
                # Check for generic "not supported" to strip optionals aggressively
                is_not_supported = "not supported" in err.lower() or "failed_precondition" in err.lower()

                # Cleanup params
                if "enhance_prompt" in err or is_not_supported: 
                    if "enhance_prompt" in current_config:
                        print("TimNodes info: Stripping 'enhance_prompt'...")
                        current_config.pop("enhance_prompt", None)
                        continue
                        
                if "audio" in err or is_not_supported:
                    if "generate_audio" in current_config:
                        print("TimNodes info: Stripping 'generate_audio'...")
                        current_config.pop("generate_audio", None)
                        continue

                # Negative prompt sometimes causes issues with interpolation
                if is_not_supported and "negative_prompt" in current_config:
                     print("TimNodes info: Stripping 'negative_prompt'...")
                     current_config.pop("negative_prompt", None)
                     continue
                     
                # Reference images might conflict with strict interpolation
                if is_not_supported and "reference_images" in current_config and img_e:
                     print("TimNodes info: Stripping 'reference_images' due to Start/End frame conflict...")
                     current_config.pop("reference_images", None)
                     continue

                # Specific checks
                if "fps" in err: current_config.pop("fps", None)
                elif "seed" in err: current_config.pop("seed", None)
                elif "person_generation" in err: current_config.pop("person_generation", None)
                elif "safety_settings" in err: current_config.pop("safety_settings", None)
                
                if attempt == max_retries -1: raise e
                
        if not operation: raise ValueError("Failed to start operation.")
        
        print(f"TimNodes VeoVertex: Operation '{operation.name}' started. Polling...")
        
        while not operation.done:
            time.sleep(5)
            operation = client.operations.get(operation)
            
        if hasattr(operation, "error") and operation.error:
             raise Exception(f"VeoVertex Operation Error: {operation.error}")

        result = operation.result
        if not result: raise ValueError("No result returned.")
        
        # --- DOWNLOAD ---
        vid_uri = None
        video_bytes = None
        
        # Check for bytes first (No URI needed)
        try:
             if hasattr(result, "generated_videos") and result.generated_videos:
                  vid = result.generated_videos[0].video
                  if hasattr(vid, "video_bytes") and vid.video_bytes:
                       video_bytes = vid.video_bytes
        except: pass
        
        if not video_bytes:
            # Robust URI Extraction
            try:
                 # Standard Veo Path
                 if hasattr(result, "generated_videos") and result.generated_videos:
                     vid_uri = result.generated_videos[0].video.uri
            except: pass
            
            if not vid_uri:
                 try:
                     # Check 'video_uri' directly on result
                     if hasattr(result, "video_uri"): vid_uri = result.video_uri
                 except: pass
                 
            if not vid_uri:
                 try:
                     # Try dictionary access if result is subscriptable
                     if "generated_videos" in result:
                          vid_uri = result["generated_videos"][0]["video"]["uri"]
                 except: pass
                 
            if not vid_uri:
                 # DEBUG: Print structure if failed
                 print(f"TimNodes Debug: Failed to extract URI. Result Object: {result}")
                 # Try searching string representation for gs:// or https://
                 import re
                 match = re.search(r'(gs://\S+|https://\S+)', str(result))
                 if match:
                     vid_uri = match.group(1).strip("',\"")
                     print(f"TimNodes Debug: Found URI via Regex: {vid_uri}")

        if not vid_uri and not video_bytes:
             # Check for Safety/RAI blocks
             rai_reason = None
             try:
                 if hasattr(result, "rai_media_filtered_reasons") and result.rai_media_filtered_reasons:
                      rai_reason = result.rai_media_filtered_reasons
                 elif hasattr(result, "rai_media_filtered_count") and result.rai_media_filtered_count and result.rai_media_filtered_count > 0:
                      rai_reason = f"{result.rai_media_filtered_count} items filtered by safety policy. (Possible Support Code: 17301594 - Person/Child Policy)"
             except: pass
                  
             if rai_reason:
                  raise ValueError(f"Generation Blocked by Safety Filters (RAI): {rai_reason}")
             
             raise ValueError(f"No video URI or Byte Data found in result. (Type: {type(result)})")
        
        final_path = os.path.join(self.output_dir, f"veo_vertex_{int(time.time())}.mp4")
        
        if video_bytes:
             print(f"TimNodes Debug: Writing {len(video_bytes)} bytes directly to file.")
             with open(final_path, "wb") as f:
                 f.write(video_bytes)
        elif vid_uri:
             print(f"TimNodes Debug: Video URI: {vid_uri}")
             
             if vid_uri.startswith("gs://"):
                 print("TimNodes: GCS URI detected. Attempting download via gcloud storage...")
                 try:
                     import subprocess
                     cmd = ["gcloud", "storage", "cp", vid_uri, final_path]
                     subprocess.run(cmd, check=True)
                 except Exception as e:
                     raise Exception(f"Failed to download GCS URI ({vid_uri}). Ensure gcloud is installed/authed. Error: {e}")
             else:
                 # HTTP Download
                 resp = requests.get(vid_uri)
                 if resp.status_code == 403:
                     print("TimNodes Warning: 403 Forbidden. This URL likely requires authentication.")
                 
                 if resp.status_code == 200:
                     with open(final_path, "wb") as f:
                         f.write(resp.content)
                 else:
                     pass

        if not os.path.exists(final_path):
             raise Exception("Video download failed.")

        # --- PROCESS FRAMES ---
        images_tensor = torch.zeros((1, 64, 64, 3))
        try:
            reader = imageio.get_reader(final_path, 'ffmpeg')
            frames = [im for im in reader]
            frames_np = np.array(frames).astype(np.float32) / 255.0
            images_tensor = torch.from_numpy(frames_np)
        except Exception as e:
            print(f"Error reading video frames: {e}")
            
        # --- AUDIO ---
        audio_output = None
        try:
            import imageio_ffmpeg
            import subprocess
            import soundfile as sf
            
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            wav_path = os.path.splitext(final_path)[0] + ".wav"
            cmd = [ffmpeg_exe, "-y", "-i", final_path, "-vn", "-acodec", "pcm_s16le", wav_path]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if os.path.exists(wav_path):
                data, sr = sf.read(wav_path)
                waveform = torch.from_numpy(data).float()
                if waveform.ndim == 1: waveform = waveform.unsqueeze(0)
                else: waveform = waveform.t()
                audio_output = {"waveform": waveform.unsqueeze(0), "sample_rate": sr}
                try: os.remove(wav_path)
                except: pass
        except: pass
        
        return (images_tensor, audio_output, final_path)


NODE_CLASS_MAPPINGS = {
    "ComfyG_Veo": ComfyG_Veo,
    "ComfyG_VeoVertexAPI": ComfyG_VeoVertexAPI,
    "ComfyG_NanoBanana": ComfyG_NanoBanana,
    "ComfyG_Model_Scanner": ComfyG_Model_Scanner,
    "ComfyG_NanoBatch": ComfyG_NanoBatch,
    "ComfyG_FolderScanner": ComfyG_FolderScanner,
    "ComfyG_NanoBatchMask": ComfyG_NanoBatchMask,
    "ComfyG_ProBanana": ComfyG_ProBanana,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyG_Veo": "TimNodes Veo",
    "ComfyG_VeoVertexAPI": "TimNodes VeoVertexAPI",
    "ComfyG_NanoBanana": "TimNodes NanoBanana",
    "ComfyG_Model_Scanner": "TimNodes Model Scanner",
    "ComfyG_NanoBatch": "TimNodes NanoBatch",
    "ComfyG_FolderScanner": "TimNodes Folder Scanner",
    "ComfyG_NanoBatchMask": "TimNodes NanoBatchMask",
    "ComfyG_ProBanana": "TimNodes ProBanana",
}
