# ComfyUI-FreeFlowGS: 4D Gaussian Splatting

**High-Fidelity 4D Gaussian Splatting Suite for ComfyUI**

ComfyUI-FreeFlowGS is a custom node suite designed to facilitate the reconstruction of 4D Gaussian Splats from multi-view video footage directly within ComfyUI. 

Designed for technical artists and researchers, this suite implements a sequential "warm-start" pipeline that prioritizes temporal consistency. The system is built upon the **Brush (WebGPU)** and **COLMAP** engines, offering native cross-platform support for **Windows (CUDA)**, **macOS (Metal)**, and **Linux (Vulkan)**.

## 🔬 Core Methodology

Standard 4D reconstruction often treats video frames as independent solved states, leading to inconsistent point counts and visual jitter. FreeFlowGS addresses this through three specific modes:

- **Cinema Mode (Fixed Topology)**: Locks the point cloud structure after the initial frame. By solving the geometry on Frame 0 and restricting point birth/death in subsequent frames, the system treats the splats as a deformable mesh. This ensures consistent point indices across the timeline, significantly reducing high-frequency flickering.
- **Adaptive Sequential Flow**: Implements a warm-start logic where Frame $N$ initializes Frame $N+1$. This maintains the stability of static elements (backgrounds) while allowing the optimizer to focus gradients on moving areas.
- **Hybrid 3D-4D Separation**: Capable of isolating the static background (solved once) from the dynamic foreground.

## 🚀 Installation

1.  **Clone the Repository**:
    Navigate to your ComfyUI `custom_nodes` directory and run:
    ```bash
    git clone https://github.com/FreeFlowVFX/ComfyUi-FreeFlowGS.git
    ```

2.  **Install Python Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Binaries**: 
    The nodes will attempt to auto-install dependencies (Brush, FFmpeg) on first run. 
    *Note: macOS users should install COLMAP via `brew install colmap`.*

---

## 📚 Node Reference & Usage

### 🎥 FreeFlow Multi-Camera Loader
Scans a directory for subfolders containing image sequences. It automatically detects valid camera folders and synchronizes frames.
- **Input**: `directory_path` (Absolute path to root capture folder)
- **Output**: `MULTICAM_DICT` (Dictionary of camera names to image paths)

### ⚓ FreeFlow COLMAP Anchor
Runs Structure-from-Motion (SfM) on Frame 0 to create a sparse point cloud. This "anchor" ensures all subsequent frames share a common coordinate system and scale.
- **Inputs**: `multicam_feed`
- **Parameters**: 
  - `quality`: Preset (Low/Medium/High/Extreme).
  - `camera_model`: `OPENCV` (standard) or `PINHOLE`.
  - `matching_method`: `Exhaustive` (Accurate), `Sequential` (Fast for video), `VocabTree` (Fast for large datasets).

### 🧠 FreeFlow 4D Adaptive Engine
The core training node that evolves the Gaussian Splats over time using the input video feed and the COLMAP anchor.
- **Inputs**: `multicam_feed`, `colmap_anchor`
- **Key Parameters**:
  - `topology_mode`:
    - **Fixed (Cinema-Smooth)**: Locks topology after Frame 0. Essential for smooth video.
    - **Dynamic (Default-Flicker)**: Adds/removes points every frame.
  - `visualize_training`: Spawns a native window or saves preview images to `output/` to monitor progress.
  - `init_from_sparse`: Initializes Frame 0 from the COLMAP point cloud.

### 📺 FreeFlow Smart Grid Monitor
Creates a professional "Video Wall" grid of all camera feeds to verify synchronization.
- **Inputs**: `multicam_feed`
- **Parameters**: `grid_resolution`, `cam_filter`, `show_labels`.

### 🌊 FreeFlow Post-Process Smoother
Applies temporal smoothing (Savitzky-Golay filter) to an existing `.ply` sequence to further reduce high-frequency jitter.
- **Inputs**: `ply_sequence`
- **Parameters**: `window_size`, `poly_order`.

### 📂 FreeFlow PLY Sequence Loader
Loads a sequence of `.ply` files for playback or post-processing.
- **Parameters**: `directory_path`, `frame_range`.

### 🧊 FreeFlow COLMAP Visualizer
Visualizes the sparse point cloud and camera positions in 3D space directly in the ComfyUI browser.
- **Inputs**: `colmap_data`

### 🎮 FreeFlow Interactive Player
Web-based 3D player with depth sorting and camera controls to view the resulting 4D splats.

---

## 🛠️ Troubleshooting

- **Flickering**: Ensure `topology_mode` is set to `Fixed (Cinema-Smooth)`. Dynamic topology inherently causes popping artifacts.
- **Training Stuck**: If using "Spawn Native GUI", close the Brush window to resume process, or switch to "Save Preview Images".
- **Missing Binaries**: Check the ComfyUI console logs. You may need to install `colmap` or `ffmpeg` manually if auto-install fails.

## License
GNU General Public License v3.0
