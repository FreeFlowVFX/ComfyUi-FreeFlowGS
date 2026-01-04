from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import folder_paths
import random
import subprocess
import shutil
import os
import hashlib

class FreeFlow_SmartGridMonitor:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "multicam_feed": ("MULTICAM_DICT",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 120}),
                "grid_resolution": (["1080p", "1440p", "4K"],),
                "output_grid_image": ("BOOLEAN", {"default": False, "label_on": "Output Image Batch (VHS Compatible)", "label_off": "Video Path Only"}),
                "cam_filter": ("STRING", {"default": "*", "multiline": False, "placeholder": "Filter cameras (e.g. *, cam01, cam02)"}),
                "show_labels": ("BOOLEAN", {"default": True, "label_on": "Show Camera Names", "label_off": "Clean Feed"}),
                "label_size": ("INT", {"default": 30, "min": 10, "max": 200}),
                "label_position": (["Top Left", "Top Center", "Top Right", "Bottom Left", "Bottom Center", "Bottom Right"],),
                "frame_range": ("STRING", {"default": "*", "multiline": False, "placeholder": "e.g. 0-100, 0,10,20, or * (All)"}),
            },
            "optional": {
                "audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("video_path", "grid_images")
    FUNCTION = "monitor"
    OUTPUT_NODE = True
    CATEGORY = "FreeFlow"

    def monitor(self, multicam_feed, fps, grid_resolution, output_grid_image, cam_filter="*", show_labels=True, label_size=30, label_position="Top Left", frame_range="*", audio=None):
        print("DEBUG: Generating Smart Grid Video Preview...")
        from ..utils import FreeFlowUtils 
        
        # 0. Camera Filtering logic
        all_cameras = sorted(multicam_feed.keys())
        if cam_filter and cam_filter.strip() != "*":
            filter_list = [c.strip() for c in cam_filter.split(",") if c.strip()]
            cameras = []
            for cam in all_cameras:
                for f in filter_list:
                    if f in cam:
                        cameras.append(cam)
                        break
            cameras = sorted(list(set(cameras)))
            if not cameras:
                print(f"WARN: Filter '{cam_filter}' matched 0 cameras. Using all.")
                cameras = all_cameras
        else:
            cameras = all_cameras

        num_cams = len(cameras)
        empty_image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)

        if num_cams == 0:
            print("ERROR: No cameras found in feed.")
            return {"ui": {"gifs": []}, "result": ("", empty_image)}

        # 1. Resolution mapping
        res_map = {
            "1080p": (1920, 1080),
            "1440p": (2560, 1440),
            "4K": (3840, 2160)
        }
        target_w, target_h = res_map.get(grid_resolution, (1920, 1080))

        # 2. Frame Range Parsing
        # Determine total possible frames from input
        total_available_frames = 0
        for cam in cameras:
            total_available_frames = max(total_available_frames, len(multicam_feed[cam]))
            
        indices_to_process = []
        
        fr = frame_range.strip()
        # Default behavior: If '*' or empty, USE EVERYTHING found in feed.
        if fr == "*" or fr.lower() == "all" or not fr:
             indices_to_process = list(range(total_available_frames))
        else:
            # Parse list/range
            try:
                parts = fr.split(",")
                for p in parts:
                    p = p.strip()
                    if "-" in p:
                        start, end = map(int, p.split("-"))
                        # Clamp end
                        end = min(end, total_available_frames)
                        indices_to_process.extend(range(start, end)) 
                    else:
                        idx = int(p)
                        if idx < total_available_frames:
                            indices_to_process.append(idx)
            except ValueError:
                print(f"WARN: Invalid frame range '{frame_range}'. Using first 100 frames as failsafe.")
                indices_to_process = list(range(min(100, total_available_frames)))
                
        # Sort and deduplicate
        indices_to_process = sorted(list(set(indices_to_process)))
        
        if not indices_to_process:
             print("WARN: No frames selected. Defaulting to first 10.")
             indices_to_process = list(range(min(10, total_available_frames)))

        # 2a. Update Cache Hash with Frame Range
        hasher = hashlib.md5()
        hasher.update(str(cameras).encode('utf-8'))
        hasher.update(str(fps).encode('utf-8'))
        hasher.update(str(grid_resolution).encode('utf-8'))
        hasher.update(str(show_labels).encode('utf-8'))
        hasher.update(str(label_size).encode('utf-8'))
        hasher.update(str(label_position).encode('utf-8'))
        hasher.update(str(indices_to_process).encode('utf-8')) # CRITICAL: Hash the ACTUAL indices
        if cameras and multicam_feed[cameras[0]]:
             hasher.update(str(multicam_feed[cameras[0]][0]).encode('utf-8'))
        
        cache_id = hasher.hexdigest()
        cache_dir = Path(folder_paths.get_temp_directory()) / "freeflow_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cached_video_path = cache_dir / f"grid_{cache_id}.mp4"
        
        # Check cache if we are NOT outputting images (images always need re-gen to populate output tensor, unless we cache tensor, but RAM...)
        if cached_video_path.exists() and not output_grid_image:
             print(f"DEBUG: Using cached grid video: {cached_video_path}")
             preview = {"filename": cached_video_path.name, "subfolder": "freeflow_cache", "type": "temp", "format": "video/mp4", "frame_rate": fps}
             return {"ui": {"gifs": [preview]}, "result": (str(cached_video_path), empty_image)}

        # 3. Grid Layout
        cols = int(np.ceil(np.sqrt(num_cams)))
        rows = int(np.ceil(num_cams / cols))
        cell_w = target_w // cols
        cell_h = target_h // rows
        
        print(f"DEBUG: Grid {cols}x{rows}, {num_cams} cameras, {len(indices_to_process)} frames selected")

        # 4. Preparation
        temp_dir = Path(folder_paths.get_temp_directory()) / f"freeflow_proc_{random.randint(10000, 99999)}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = temp_dir / "frames"
        frames_dir.mkdir()
        
        tensor_frames = []
        
        # Font setup (System Fonts)
        font = None
        system_fonts = [
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/SFNS.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        
        for fpath in system_fonts:
            if os.path.exists(fpath):
                try:
                    font = ImageFont.truetype(fpath, size=label_size)
                    break 
                except: continue
        
        if font is None:
            try:
                font = ImageFont.truetype("arial.ttf", size=label_size)
            except:
                font = ImageFont.load_default() # Fallback

        # 5. Composite Generation
        for i, frame_idx in enumerate(indices_to_process):
            # i is sequential index for output file (frame_000000.png)
            # frame_idx is actual source frame index (e.g. 10, 15, 20)
            
            canvas = Image.new("RGB", (target_w, target_h), (0, 0, 0))
            draw = ImageDraw.Draw(canvas)
            
            for cam_idx, cam in enumerate(cameras):
                frames = multicam_feed[cam]
                
                # Calculate position
                r = cam_idx // cols
                c = cam_idx % cols
                x = c * cell_w
                y = r * cell_h
                
                if frame_idx < len(frames):
                    try:
                        img = Image.open(frames[frame_idx])
                        img = ImageOps.exif_transpose(img)
                        img = img.convert("RGB")
                        img = img.resize((cell_w, cell_h), Image.Resampling.LANCZOS)
                        canvas.paste(img, (x, y))
                    except Exception as e:
                        pass
                
                # Draw Label
                if show_labels:
                    label_text = f"{cam}"
                    
                    # Calculate Text Position
                    try:
                        bbox = draw.textbbox((0, 0), label_text, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]
                    except:
                        text_w, text_h = draw.textsize(label_text, font=font)

                    margin = 10
                    
                    if "Top" in label_position:
                        text_y = y + margin
                    else: # Bottom
                        text_y = y + cell_h - text_h - margin * 2
                    
                    if "Left" in label_position:
                        text_x = x + margin
                    elif "Right" in label_position:
                        text_x = x + cell_w - text_w - margin
                    else: # Center
                        text_x = x + (cell_w - text_w) // 2
                    
                    # Draw Background Box
                    padding = 5
                    bg_box = [text_x - padding, text_y - padding, text_x + text_w + padding, text_y + text_h + padding]
                    draw.rectangle(bg_box, fill="black", outline="white", width=1)
                    
                    draw.text((text_x, text_y), label_text, font=font, fill="white")
            
            # Save sequential frame
            frame_path = frames_dir / f"frame_{i:06d}.png"
            canvas.save(frame_path)
            
            if output_grid_image:
                 img_array = np.array(canvas).astype(np.float32) / 255.0
                 tensor_frames.append(torch.from_numpy(img_array))

            if i % max(1, len(indices_to_process) // 10) == 0:
                print(f"   Progress: {i}/{len(indices_to_process)} frames")

        # 6. Encoding
        ffmpeg_bin = FreeFlowUtils.get_binary_path("ffmpeg")
        if not ffmpeg_bin:
             print("ERROR: ffmpeg not found! Returning static.")
             return {"ui": {"gifs": []}, "result": ("", empty_image)}

        output_video = temp_dir / "grid_preview.mp4"
        
        ffmpeg_cmd = [
            str(ffmpeg_bin), "-y",
            "-framerate", str(fps),
            "-start_number", "0",  # Frames start from 0
            "-i", str(frames_dir / "frame_%06d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
            str(output_video)
        ]
        
        print(f"DEBUG: Running ffmpeg: {' '.join(ffmpeg_cmd)}")
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            print(f"DEBUG: ffmpeg completed successfully, output: {output_video}")
            shutil.copy2(output_video, cached_video_path)
            shutil.rmtree(temp_dir, ignore_errors=True)
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg Error: {e}")
            print(f"FFmpeg STDERR: {e.stderr.decode('utf-8', errors='ignore')}")
            print(f"FFmpeg STDOUT: {e.stdout.decode('utf-8', errors='ignore')}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return {"ui": {"gifs": []}, "result": ("", empty_image)}

        preview = {
            "filename": cached_video_path.name,
            "subfolder": "freeflow_cache",
            "type": "temp",
            "format": "video/mp4",
            "frame_rate": fps,
        }
        
        final_tensor = empty_image
        if output_grid_image and tensor_frames:
             final_tensor = torch.stack(tensor_frames)

        return {"ui": {"gifs": [preview]}, "result": (str(cached_video_path), final_tensor)}
