
import sys
import os
import shutil
import json
import datetime
import re
import torch
import numpy as np
import itertools
import subprocess
from PIL import Image, ExifTags
from PIL.PngImagePlugin import PngInfo

# Add VHS path to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) # Google2Comfy/
custom_nodes_root = os.path.dirname(project_root) # ComfyUI/custom_nodes/

# List of common folder names for Video Helper Suite
possible_vhs_dirs = ["comfyui-videohelpersuite", "ComfyUI-VideoHelperSuite", "VideoHelperSuite"]
vhs_path = None

for d in possible_vhs_dirs:
    potential_path = os.path.join(custom_nodes_root, d)
    if os.path.exists(potential_path):
        vhs_path = potential_path
        break

if vhs_path:
    if vhs_path not in sys.path:
        print(f"VideoCombineVHStim: using VHS at {vhs_path}")
        sys.path.append(vhs_path)
else:
    # Fallback to local if not found externally (just in case user hasn't deleted it yet or path logic fails)
    local_vhs = os.path.join(project_root, "comfyui-videohelpersuite")
    if os.path.exists(local_vhs):
        print(f"VideoCombineVHStim: External VHS not found, using local copy at {local_vhs}")
        if local_vhs not in sys.path:
            sys.path.append(local_vhs)
    else:
        print("VideoCombineVHStim Warning: comfyui-videohelpersuite not found in custom_nodes or locally. Import errors may occur.")

try:
    import folder_paths
except ImportError:
    # Fallback to prevent import error during development/testing
    folder_paths = None

from comfy.utils import ProgressBar

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
    ENCODE_ARGS
)
from videohelpersuite.logger import logger

class VideoCombineVHStim(VideoCombine):
    @classmethod
    def INPUT_TYPES(s):
        ffmpeg_formats, format_widgets = get_video_formats()
        format_widgets["image/webp"] = [['lossless', "BOOLEAN", {'default': True}]]
        
        # We need to manually construct the dictionary to ensure everything is included
        # Copying from original VHS node structure
        return {
            "required": {
                "images": (imageOrLatent,),
                "frame_rate": (
                    floatOrInt,
                    {"default": 8, "min": 1, "step": 1},
                ),
                "loop_count": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "filename_prefix": ("STRING", {"default": "AnimateDiff"}),
                "format": (["image/gif", "image/webp"] + ffmpeg_formats, {"formats": format_widgets}),
                "pingpong": ("BOOLEAN", {"default": False}),
                "save_output": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "audio": ("AUDIO",),
                "meta_batch": ("VHS_BatchManager",),
                "vae": ("VAE",),
                "temporary_file": ("STRING", {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "unique_id": "UNIQUE_ID",
                "manual_format_widgets": "MANUAL_FORMAT_WIDGETS",
            },
        }

    RETURN_TYPES = ("VHS_FILENAMES",)
    RETURN_NAMES = ("Filenames",)
    OUTPUT_NODE = True
    CATEGORY = "TimNodes/VHS"
    FUNCTION = "combine_video"

    async def combine_video(
        self,
        frame_rate: int,
        loop_count: int,
        images=None,
        latents=None,
        filename_prefix="AnimateDiff",
        format="image/gif",
        pingpong=False,
        save_output=True,
        prompt=None,
        extra_pnginfo=None,
        audio=None,
        unique_id=None,
        manual_format_widgets=None,
        meta_batch=None,
        vae=None,
        temporary_file="",
        **kwargs
    ):
        # --- Custom System File Path Logic (Simplified) ---
        use_system_path = False
        
        # Check if user put path in filename_prefix
        target_path = ""
        target_filename = filename_prefix
        
        if os.path.isabs(filename_prefix):
            # User put absolute path in filename_prefix
            target_path = os.path.dirname(filename_prefix)
            target_filename = os.path.basename(filename_prefix)
            use_system_path = True
            
        if use_system_path:
            # Ensure safe path
            try:
                if not os.path.exists(target_path):
                    os.makedirs(target_path, exist_ok=True)
                
                # Check for existing file and increment counter to avoid overwrite
                # target_filename is "MyVideo" (prefix)
                # The format dict (which we don't fully parse here yet) adds extension later?
                # Actually, in combine_video, we don't know the exact extension UNTIL we parse 'format'.
                # But 'format' is passed as argument.
                # However, the standard `get_save_image_path` handles this validation internally.
                # Since we are creating logic BEFORE ffmpeg runs, we need to guess the extension or resolve collision AFTER file creation?
                # No, we must providing the full output path to ffmpeg. 
                # Let's verify extension from 'format' (user selected widget).
                
                # Simple heuristic:
                # If 'format' is list from widgets, it's [extension, format_name]
                # If 'format' is string, it's "video/h264-mp4" etc.
                
                # Wait, VideoCombine logic:
                # format argument is checked later:
                # video_format = get_format_config(format, ffmpeg_formats)
                # We need to replicate that briefly to know the extension if we want to check collision NOW.
                
                # BUT: The 'filename' passed to ffmpeg usually INCLUDES pattern for counter OR is fixed.
                # If we pass "MyVideo", ffmpeg creates "MyVideo.mp4".
                # If "MyVideo.mp4" exists, we want "MyVideo_0001.mp4".
                
                # Let's Implement a simple "Find next available prefix"
                # We simply look for files starting with prefix in the dir.
                
                # Actually, standard ComfyUI behavior: prefix_00001.
                # We will replicate this.
                
                counter = 1
                unique_filename = f"{target_filename}_{counter:05}"
                # We can't check full existence without extension, which depends on format.
                # Compromise: We will search for ANY file with that base name? 
                # No, that's too aggressive.
                
                # Better approach:
                # Inspect the user 'format' arg to guess extension?
                # Or just construct the base name with counter and let it be.
                # For robustness, let's look for the highest counter in the dir matching the prefix.
                
                # Basic implementation matching ComfyUI's standard 'get_save_image_path' somewhat:
                existing_counters = []
                for f in os.listdir(target_path):
                    if f.startswith(target_filename):
                        # check if it matches pattern: prefix_XXXXX.ext
                        # remove extension
                        base = os.path.splitext(f)[0]
                        suffix = base[len(target_filename):]
                        if suffix.startswith("_") and suffix[1:].isdigit():
                            try:
                                existing_counters.append(int(suffix[1:]))
                            except:
                                pass
                
                if existing_counters:
                    existing_counters.sort()
                    counter = existing_counters[-1] + 1
                
                filename = f"{target_filename}_{counter:05}"
                full_output_folder = target_path
                subfolder = ""     
            except Exception as e:
                logger.error(f"Failed to create system path: {target_path}. Error: {e}")
                raise Exception(f"Failed to create system path: {target_path}")
        
        if not use_system_path:
            # Standard ComfyUI Logic
            if folder_paths is None:
                raise ImportError("folder_paths not found. Are you running inside ComfyUI?")
            output_dir = (
                folder_paths.get_output_directory()
                if save_output
                else folder_paths.get_temp_directory()
            )
            (
                full_output_folder,
                filename,
                counter,
                subfolder,
                _,
            ) = folder_paths.get_save_image_path(filename_prefix, output_dir)

    # --- Custom Save/Preview Logic for Temporary File ---
        if temporary_file and os.path.exists(temporary_file):
            final_file_path = temporary_file
            
            # Use extension from temp file if possible, else default to mp4
            temp_ext = os.path.splitext(temporary_file)[1]
            if not temp_ext: temp_ext = ".mp4"
            
            if use_system_path:
                # Copy to desired system path with COLLISION CHECK
                try:
                    # Re-evaluate filename collision based on ACTUAL extension of temp file
                    # The block above guessed purely on prefix. Here we know the extension.
                    # Let's perform a specific check for this file copy.
                    
                    copy_counter = 1
                    # Loop until free
                    while True:
                         check_name = f"{target_filename}_{copy_counter:05}{temp_ext}"
                         check_path = os.path.join(full_output_folder, check_name)
                         if not os.path.exists(check_path):
                             dest_path = check_path
                             break
                         copy_counter += 1
                    
                    shutil.copy2(temporary_file, dest_path)
                    final_file_path = dest_path
                except Exception as e:
                    logger.error(f"Failed to copy temp file to system path: {e}")
            
            # For preview:
            output_dir_base = folder_paths.get_output_directory()
            tim_temp = os.path.join(output_dir_base, "temp_TimNodes")
            
            preview_filename = os.path.basename(temporary_file)
            preview_subfolder = "temp_TimNodes"
            
            # Force copy to temp_TimNodes to ensure visibility
            if not os.path.exists(tim_temp): os.makedirs(tim_temp, exist_ok=True)
            dest_preview = os.path.join(tim_temp, preview_filename)
            
            # Always copy/overwrite to ensure fresh preview logic
            try:
                shutil.copy2(temporary_file, dest_preview)
            except Exception as e:
                logger.warn(f"Preview copy failed: {e}")

            preview = {}
            preview['filename'] = preview_filename
            preview['subfolder'] = preview_subfolder
            preview['type'] = 'output'
            preview['format'] = format
            preview['frame_rate'] = frame_rate
            preview['fullpath'] = final_file_path
            
            return {"ui": {"gifs": [preview]}, "result": ((save_output, [final_file_path]),)}

        output_files = []

        if images is None:
            return ((save_output, []),)
        if vae is not None:
            if isinstance(images, dict):
                images = images['samples']
            else:
                vae = None
                
        if isinstance(images, torch.Tensor) and images.size(0) == 0:
            return ((save_output, []),)
            
        num_frames = len(images)
        pbar = ProgressBar(num_frames)
        if vae is not None:
            downscale_ratio = getattr(vae, "downscale_ratio", 8)
            width = images.size(-1)*downscale_ratio
            height = images.size(-2)*downscale_ratio
            frames_per_batch = (1920 * 1080 * 16) // (width * height) or 1
            #Python 3.12 adds an itertools.batched, but it's easily replicated for legacy support
            def batched(it, n):
                while batch := tuple(itertools.islice(it, n)):
                    yield batch
            def batched_encode(images, vae, frames_per_batch):
                for batch in batched(iter(images), frames_per_batch):
                    image_batch = torch.from_numpy(np.array(batch))
                    yield from vae.decode(image_batch)
            images = batched_encode(images, vae, frames_per_batch)
            first_image = next(images)
            #repush first_image
            images = itertools.chain([first_image], images)
            #A single image has 3 dimensions. Discard higher dimensions
            while len(first_image.shape) > 3:
                first_image = first_image[0]
        else:
            first_image = images[0]
            images = iter(images)
        
        
        # Path logic moved to top


        metadata = PngInfo()
        video_metadata = {}
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
            video_metadata["prompt"] = json.dumps(prompt)
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
                video_metadata[x] = extra_pnginfo[x]
            extra_options = extra_pnginfo.get('workflow', {}).get('extra', {})
        else:
            extra_options = {}
        metadata.add_text("CreationTime", datetime.datetime.now().isoformat(" ")[:19])

        if meta_batch is not None and unique_id in meta_batch.outputs:
            (counter, output_process) = meta_batch.outputs[unique_id]
        else:
            # comfy counter workaround
            max_counter = 0

            # Loop through the existing files
            matcher = re.compile(f"{re.escape(filename)}_(\\d+)\\D*\\..+", re.IGNORECASE)
            
            if os.path.exists(full_output_folder):
                for existing_file in os.listdir(full_output_folder):
                    # Check if the file matches the expected format
                    match = matcher.fullmatch(existing_file)
                    if match:
                        # Extract the numeric portion of the filename
                        file_counter = int(match.group(1))
                        # Update the maximum counter value if necessary
                        if file_counter > max_counter:
                            max_counter = file_counter

            # Increment the counter by 1 to get the next available value
            counter = max_counter + 1
            output_process = None

        # save first frame as png to keep metadata
        first_image_file = f"{filename}_{counter:05}.png"
        file_path = os.path.join(full_output_folder, first_image_file)
        if extra_options.get('VHS_MetadataImage', True) != False:
            Image.fromarray(tensor_to_bytes(first_image)).save(
                file_path,
                pnginfo=metadata,
                compress_level=4,
            )
        output_files.append(file_path)

        format_type, format_ext = format.split("/")
        if format_type == "image":
            if meta_batch is not None:
                raise Exception("Pillow('image/') formats are not compatible with batched output")
            image_kwargs = {}
            if format_ext == "gif":
                image_kwargs['disposal'] = 2
            if format_ext == "webp":
                #Save timestamp information
                exif = Image.Exif()
                exif[ExifTags.IFD.Exif] = {36867: datetime.datetime.now().isoformat(" ")[:19]}
                image_kwargs['exif'] = exif
                image_kwargs['lossless'] = kwargs.get("lossless", True)
            file = f"{filename}_{counter:05}.{format_ext}"
            file_path = os.path.join(full_output_folder, file)
            if pingpong:
                images = to_pingpong(images)
            def frames_gen(images):
                for i in images:
                    pbar.update(1)
                    yield Image.fromarray(tensor_to_bytes(i))
            frames = frames_gen(images)
            # Use pillow directly to save an animated image
            next(frames).save(
                file_path,
                format=format_ext.upper(),
                save_all=True,
                append_images=frames,
                duration=round(1000 / frame_rate),
                loop=loop_count,
                compress_level=4,
                **image_kwargs
            )
            output_files.append(file_path)
        else:
            # Use ffmpeg to save a video
            if ffmpeg_path is None:
                raise ProcessLookupError(f"ffmpeg is required for video outputs and could not be found.\nIn order to use video outputs, you must either:\n- Install imageio-ffmpeg with pip,\n- Place a ffmpeg executable in {os.path.abspath('')}, or\n- Install ffmpeg and add it to the system path.")

            if manual_format_widgets is not None:
                logger.warn("Format args can now be passed directly. The manual_format_widgets argument is now deprecated")
                kwargs.update(manual_format_widgets)

            has_alpha = first_image.shape[-1] == 4
            kwargs["has_alpha"] = has_alpha
            video_format = apply_format_widgets(format_ext, kwargs)
            dim_alignment = video_format.get("dim_alignment", 2)
            if (first_image.shape[1] % dim_alignment) or (first_image.shape[0] % dim_alignment):
                #output frames must be padded
                to_pad = (-first_image.shape[1] % dim_alignment,
                          -first_image.shape[0] % dim_alignment)
                padding = (to_pad[0]//2, to_pad[0] - to_pad[0]//2,
                           to_pad[1]//2, to_pad[1] - to_pad[1]//2)
                padfunc = torch.nn.ReplicationPad2d(padding)
                def pad(image):
                    image = image.permute((2,0,1))#HWC to CHW
                    padded = padfunc(image.to(dtype=torch.float32))
                    return padded.permute((1,2,0))
                images = map(pad, images)
                dimensions = (-first_image.shape[1] % dim_alignment + first_image.shape[1],
                              -first_image.shape[0] % dim_alignment + first_image.shape[0])
                logger.warn("Output images were not of valid resolution and have had padding applied")
            else:
                dimensions = (first_image.shape[1], first_image.shape[0])
            if pingpong:
                if meta_batch is not None:
                    logger.error("pingpong is incompatible with batched output")
                images = to_pingpong(images)
                if num_frames > 2:
                    num_frames += num_frames -2
                    pbar.total = num_frames
            if loop_count > 0:
                loop_args = ["-vf", "loop=loop=" + str(loop_count)+":size=" + str(num_frames)]
            else:
                loop_args = []
            if video_format.get('input_color_depth', '8bit') == '16bit':
                images = map(tensor_to_shorts, images)
                if has_alpha:
                    i_pix_fmt = 'rgba64'
                else:
                    i_pix_fmt = 'rgb48'
            else:
                images = map(tensor_to_bytes, images)
                if has_alpha:
                    i_pix_fmt = 'rgba'
                else:
                    i_pix_fmt = 'rgb24'
            file = f"{filename}_{counter:05}.{video_format['extension']}"
            file_path = os.path.join(full_output_folder, file)
            bitrate_arg = []
            bitrate = video_format.get('bitrate')
            if bitrate is not None:
                bitrate_arg = ["-b:v", str(bitrate) + "M" if video_format.get('megabit') == 'True' else str(bitrate) + "K"]
            args = [ffmpeg_path, "-v", "error", "-f", "rawvideo", "-pix_fmt", i_pix_fmt,
                    # The image data is in an undefined generic RGB color space, which in practice means sRGB.
                    # sRGB has the same primaries and matrix as BT.709, but a different transfer function (gamma),
                    # called by the sRGB standard name IEC 61966-2-1. However, video hosting platforms like YouTube
                    # standardize on full BT.709 and will convert the colors accordingly. This last minute change
                    # in colors can be confusing to users. We can counter it by lying about the transfer function
                    # on a per format basis, i.e. for video we will lie to FFmpeg that it is already BT.709. Also,
                    # because the input data is in RGB (not YUV) it is more efficient (fewer scale filter invocations)
                    # to specify the input color space as RGB and then later, if the format actually wants YUV,
                    # to convert it to BT.709 YUV via FFmpeg's -vf "scale=out_color_matrix=bt709".
                    "-color_range", "pc", "-colorspace", "rgb", "-color_primaries", "bt709",
                    "-color_trc", video_format.get("fake_trc", "iec61966-2-1"),
                    "-s", f"{dimensions[0]}x{dimensions[1]}", "-r", str(frame_rate), "-i", "-"] \
                    + loop_args

            images = map(lambda x: x.tobytes(), images)
            env=os.environ.copy()
            if  "environment" in video_format:
                env.update(video_format["environment"])

            if "pre_pass" in video_format:
                if meta_batch is not None:
                    #Performing a prepass requires keeping access to all frames.
                    #Potential solutions include keeping just output frames in
                    #memory or using 3 passes with intermediate file, but
                    #very long gifs probably shouldn't be encouraged
                    raise Exception("Formats which require a pre_pass are incompatible with Batch Manager.")
                images = [b''.join(images)]
                os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
                in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
                pre_pass_args = args[:in_args_len] + video_format['pre_pass']
                merge_filter_args(pre_pass_args)
                try:
                    subprocess.run(pre_pass_args, input=images[0], env=env,
                                   capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occurred in the ffmpeg prepass:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
            if "inputs_main_pass" in video_format:
                in_args_len = args.index("-i") + 2 # The index after ["-i", "-"]
                args = args[:in_args_len] + video_format['inputs_main_pass'] + args[in_args_len:]

            if output_process is None:
                if 'gifski_pass' in video_format:
                    format = 'image/gif'
                    output_process = gifski_process(args, dimensions, frame_rate, video_format, file_path, env)
                    audio = None
                else:
                    args += video_format['main_pass'] + bitrate_arg
                    merge_filter_args(args)
                    output_process = ffmpeg_process(args, video_format, video_metadata, file_path, env)
                #Proceed to first yield
                output_process.send(None)
                if meta_batch is not None:
                    meta_batch.outputs[unique_id] = (counter, output_process)

            for image in images:
                pbar.update(1)
                output_process.send(image)
            if meta_batch is not None:
                requeue_workflow((meta_batch.unique_id, not meta_batch.has_closed_inputs))
            if meta_batch is None or meta_batch.has_closed_inputs:
                #Close pipe and wait for termination.
                try:
                    total_frames_output = output_process.send(None)
                    output_process.send(None)
                except StopIteration:
                    pass
                if meta_batch is not None:
                    meta_batch.outputs.pop(unique_id)
                    if len(meta_batch.outputs) == 0:
                        meta_batch.reset()
            else:
                #batch is unfinished
                #TODO: Check if empty output breaks other custom nodes
                return {"ui": {"unfinished_batch": [True]}, "result": ((save_output, []),)}

            output_files.append(file_path)


            a_waveform = None
            if audio is not None:
                try:
                    #safely check if audio produced by VHS_LoadVideo actually exists
                    a_waveform = audio['waveform']
                except:
                    pass
            if a_waveform is not None:
                # Create audio file if input was provided
                output_file_with_audio = f"{filename}_{counter:05}-audio.{video_format['extension']}"
                output_file_with_audio_path = os.path.join(full_output_folder, output_file_with_audio)
                if "audio_pass" not in video_format:
                    logger.warn("Selected video format does not have explicit audio support")
                    video_format["audio_pass"] = ["-c:a", "libopus"]


                # FFmpeg command with audio re-encoding
                #TODO: expose audio quality options if format widgets makes it in
                #Reconsider forcing apad/shortest
                channels = audio['waveform'].size(1)
                min_audio_dur = total_frames_output / frame_rate + 1
                if video_format.get('trim_to_audio', 'False') != 'False':
                    apad = []
                else:
                    apad = ["-af", "apad=whole_dur="+str(min_audio_dur)]
                mux_args = [ffmpeg_path, "-v", "error", "-n", "-i", file_path,
                            "-ar", str(audio['sample_rate']), "-ac", str(channels),
                            "-f", "f32le", "-i", "-", "-c:v", "copy"] \
                            + video_format["audio_pass"] \
                            + apad + ["-shortest", output_file_with_audio_path]

                audio_data = audio['waveform'].squeeze(0).transpose(0,1) \
                        .numpy().tobytes()
                merge_filter_args(mux_args, '-af')
                try:
                    res = subprocess.run(mux_args, input=audio_data,
                                         env=env, capture_output=True, check=True)
                except subprocess.CalledProcessError as e:
                    raise Exception("An error occured in the ffmpeg subprocess:\n" \
                            + e.stderr.decode(*ENCODE_ARGS))
                if res.stderr:
                    print(res.stderr.decode(*ENCODE_ARGS), end="", file=sys.stderr)
                output_files.append(output_file_with_audio_path)
                #Return this file with audio to the webui.
                #It will be muted unless opened or saved with right click
                file = output_file_with_audio
        if extra_options.get('VHS_KeepIntermediate', True) == False:
            for intermediate in output_files[1:-1]:
                if os.path.exists(intermediate):
                    os.remove(intermediate)
        preview = {
                "filename": file,
                "subfolder": subfolder,
                "type": "output" if save_output else "temp",
                "format": format,
                "frame_rate": frame_rate,
                "workflow": first_image_file,
                "fullpath": output_files[-1],
            }
        
        
        # Custom Save/Preview Logic for Temporary File moved to top


        # CASE B: Standard Generation (No temp file)
        # Check output_files from previous block
        if output_files:
             generated_file = output_files[-1]
             
             # If using system path, we already saved it there (via file_path logic in ffmpeg block? No wait)
             # In original code: file_path = os.path.join(full_output_folder, file)
             # If use_system_path was True, full_output_folder IS the system path.
             # So the file IS at the system path.
             
             # Now for Preview:
             # If use_system_path is True, file is outside. Copy to temp_TimNodes.
             # If use_system_path is False, file is in output. Preview is fine.
             
             if use_system_path:
                 output_dir_base = folder_paths.get_output_directory()
                 tim_temp = os.path.join(output_dir_base, "temp_TimNodes")
                 if not os.path.exists(tim_temp): os.makedirs(tim_temp, exist_ok=True)
                 
                 preview_filename = f"tmp_vhs_{os.path.basename(generated_file)}"
                 dest_preview = os.path.join(tim_temp, preview_filename)
                 shutil.copy2(generated_file, dest_preview)
                 
                 preview['type'] = 'output' # Serve from output/temp_TimNodes
                 preview['subfolder'] = 'temp_TimNodes'
                 preview['filename'] = preview_filename
                 
        if num_frames == 1 and 'png' in format and '%03d' in file:
            preview['format'] = 'image/png'
            preview['filename'] = file.replace('%03d', '001')
        return {"ui": {"gifs": [preview]}, "result": ((save_output, output_files),)}

NODE_CLASS_MAPPINGS = {
    "VideoCombineVHStim": VideoCombineVHStim,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "VideoCombineVHStim": "Video Combine VHS tim",
}
