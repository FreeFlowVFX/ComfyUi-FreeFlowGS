# ComfyUI-FreeFlowGS
**High-Fidelity 4D Gaussian Splatting Suite for ComfyUI**

ComfyUI-FreeFlowGS is a custom node suite designed to facilitate the reconstruction of 4D Gaussian Splats from multi-view video footage directly within ComfyUI.

Designed for technical artists and researchers, this suite implements a sequential "warm-start" pipeline that prioritizes temporal consistency. By separating the reconstruction of static environments from dynamic subjects and offering control over point topology, FreeFlowGS aims to mitigate the flickering and popping artifacts common in standard dynamic splatting workflows.

The system is built upon the **Brush (WebGPU)** and **COLMAP** engines, offering native cross-platform support for **Windows (CUDA)**, **macOS (Metal)**, and **Linux (Vulkan)**.

---

## 🔬 Core Methodology

Standard 4D reconstruction often treats video frames as independent solved states, leading to inconsistent point counts and visual jitter. FreeFlowGS addresses this through three specific modes of operation:

### 1. Cinema Mode (Fixed Topology)
Locks the point cloud structure after the initial frame. By solving the geometry on Frame 0 and restricting point birth/death in subsequent frames, the system treats the splats as a deformable mesh. This ensures consistent point indices across the timeline, significantly reducing high-frequency flickering.

### 2. Adaptive Sequential Flow
Implements a warm-start logic where Frame $N$ initializes Frame $N+1$. This maintains the stability of static elements (backgrounds) while allowing the optimizer to focus gradients on moving areas.

### 3. Hybrid 3D-4D Separation
Capable of isolating the static background (solved once) from the dynamic foreground, ensuring that non-moving parts of the scene remain computationally cheap and visually stable.

---

## ✨ Key Features

### 🎥 Production-Grade 4D Pipeline
- **Sequential Solving**: Automates the "load-train-save-init" loop required for temporally consistent video reconstruction.
- **Cross-Platform Engine**: Leverages the brush CLI with WebGPU support, allowing native execution on Apple Silicon (M-Series) without emulation, alongside standard NVIDIA CUDA support.

### 📐 "Glass-Box" Photogrammetry
- **Native COLMAP Integration**: Exposes the full COLMAP CLI parameters (SIFT extraction, matching, mapping) within ComfyUI, rather than hiding them behind presets.
- **In-Node Visualization**: Includes a custom 3D viewer node that parses and renders COLMAP sparse point clouds and camera frustums, allowing users to verify alignment quality before committing to training.

### 📹 Multi-View Data Management
- **Scalable Loader**: Designed to handle synchronized footage from 15+ cameras.
- **Smart Grid Monitor**: A dedicated custom UI widget for verifying temporal synchronization across all inputs, featuring global scrubbing and a "Motion Mask" view to visualize frame differences.

### 📦 Optimization & Storage
- **Tiered Compression**: Includes a dedicated post-processing node for managing file sizes:
    - **Production**: FP16 quantization (approx. 50% size reduction, visually lossless).
    - **Web Delivery**: `.spz` format support (approx. 90% size reduction) for integration with modern web viewers.

---

## 🔄 Supported Workflows
- **Multi-View Video to 4D Splat**: Full pipeline from raw footage to `.ply` sequence.
- **Stop-Motion Reconstruction**: Specialized handling for frame-by-frame captured sequences.
- **Hybrid Static/Dynamic Composing**: (Experimental) Mask-based separation of actors and environments.

---

## 💻 Requirements

- **ComfyUI**: Latest version (Python 3.10+ recommended).
- **Hardware**:
    - **Windows**: NVIDIA GPU (CUDA) or Vulkan-compatible GPU.
    - **macOS**: Apple Silicon (M1/M2/M3/M4) via Metal.
    - **Linux**: Vulkan-compatible GPU.
- **Dependencies**: `brush` and `colmap` binaries (Automatic setup included in the `__init__.py`).

## Installation

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
    The nodes will attempt to auto-install dependencies on first run. 
    *Note: macOS users should install COLMAP via `brew install colmap`.*
