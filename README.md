# ComfyUI-FreeFlowGS

Production-focused 4D Gaussian Splatting tools for ComfyUI.

FreeFlowGS is a multi-camera, sequential warm-start pipeline designed for temporally stable splat sequences. It supports multiple training backends and includes fixed-topology workflows for downstream smoothing, interpolation, and editorial use.

## What It Delivers

- Frame-to-frame warm start with backend-specific initialization
- Fixed-topology mode for stable point count and order
- Dynamic mode for unconstrained growth/adaptation
- Distributed anchor workflow across multiple machines
- Motion-aware masking for temporal stability
- Realign/smooth post-process outputs in separate folders

## Core Method (Temporal Prediction, Compression-Inspired)

FreeFlowGS uses a temporal strategy inspired by prediction ideas common in video compression:

- A strong reference state (anchor frame)
- Predictive continuation to subsequent frames (warm start)
- Motion-aware update gating (masking)

This is a conceptual analogy to I/P-frame style thinking, not an implementation of AVC/HEVC algorithms (no block residual coding, quantization, or entropy coding). The practical goal is similar: preserve stable regions and spend optimization where motion actually exists.

## Backend Matrix

| Backend | Label in UI | Best Use | Platform Reality | Notes |
|---|---|---|---|---|
| Brush | `Brush` | Fast iteration and broad support | Windows, macOS, Linux | Uses PLY -> points warm start |
| Splatfacto (Nerfstudio) | `Splatfacto (Pro)` | Highest controllability and quality | CUDA/NVIDIA only | Full checkpoint warm start (all Gaussian attributes) |
| OpenSplat | `OpenSplat (Mac/CPU)` | CPU/OpenCL fallback path | macOS/CPU-friendly path | Shared-parameter path, simpler setup |

Notes:
- Splatfacto depends on `gsplat`, which currently requires CUDA. In practice this means **no native Splatfacto training on macOS**.
- The node UI hides unavailable backends by platform/capability checks.

## ⚙️ Installation

1. Clone into ComfyUI custom nodes:

```bash
git clone https://github.com/FreeFlowVFX/ComfyUi-FreeFlowGS.git
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

3. First run (ComfyUI auto-install flow):
- FreeFlow performs dependency checks and attempts auto-install for supported components when the nodes initialize.
- This includes Nerfstudio environment setup where supported and backend-specific dependencies.
- On macOS, install COLMAP manually if needed:

```bash
brew install colmap
```

### Windows note (Splatfacto)

- For Windows CUDA setups, the project supports pre-built `gsplat` wheel installation during Nerfstudio setup to avoid local compiler friction.
- This improves first-run reliability for `Splatfacto (Pro)` on Windows.

## 🎬 Recommended Production Workflow

### Single Machine (Fixed Topology)

1. Build COLMAP anchor
2. Run `FreeFlow GS Engine (4D)` with:
   - `topology_mode = Fixed (Stable)`
   - optional `initial_iterations_override` + `initial_quality_preset`
3. Optional postprocess:
   - `realigned/` for topology alignment output
   - `smoothed/` for temporal smoothing output

### Multi-Machine (Distributed)

One machine acts as producer, others as consumers.

- Producer:
  - `distributed_anchor = true`
  - choose `distributed_anchor_frame`
  - writes `Distributed_Anchor/anchor_frame_XXXX.*`

- Consumers:
  - same shared output root
  - same `distributed_anchor_frame`
  - start from distributed anchor source automatically

Splatfacto uses checkpoint-handoff semantics for distributed continuity; Brush/OpenSplat use anchor PLY handoff.

## 🧠 GS Engine: Key Behaviors

### Fixed Topology

- Topology is locked after initialization phase
- Designed for temporal consistency and post smoothing
- Splatfacto fixed export uses checkpoint-direct path to preserve point count consistency

### Warm Start

- Brush/OpenSplat: prior PLY converted to points for next frame initialization
- Splatfacto: previous checkpoint injected (full attribute warm start), with optimizer/callback rebind safety

### Warmup Frames

`warmup_frames` is pre-roll before selected range.

- It uses frames before your first selected frame if they exist
- It does not consume your selected main range
- Anchor frame is never treated as warmup

## 🛰️ Masking: Why It Exists and How It Works

Temporal drift often appears when static regions continue receiving unnecessary updates. Masking reduces that by focusing updates on moving content.

### Parameters

- `masking_method`: `None`, `Optical Flow (Robust)`, `Simple Diff (Fast)`
- `motion_sensitivity`: 0.0 to 1.0, step 0.001 (3-decimal precision)
  - Higher values detect subtler motion (larger moving region)
  - For "exclude only fully static", start around `0.80-0.90`

### Splatfacto Mask Modes

- `Mask Only (Current)`
  - Writes `masks/` in frame work dir
  - Uses nerfstudio `--masks-path`

- `Blend Static From Previous (Recommended)`
  - Still computes optical-flow masks
  - Applies masks directly to create blended training images
  - No `masks/` folder is expected in this mode
  - Useful for reducing visible seam artifacts in static zones

### Debug Outputs

Enable `save_mask_debug_images` (Splatfacto) to write inspection artifacts to:

`output/MaskDebug/<frame_xxxx_work>/`

Includes mask snapshots and, in blend mode, current/previous/blended image outputs.

## 🧬 FLAME Integration Status (In Progress)

FreeFlow includes a `FreeFlow FlameTracker` node and mesh-guidance hooks, but the full FLAME-driven "stable underlying mesh driver" workflow is still in progress.

Current status:

- `FreeFlow FlameTracker` supports:
  - `MediaPipe-3D (Fast-Robust)`
  - `FLAME-Fit (Pro-Experimental)`
- `FreeFlow GS Engine (4D)` accepts optional `guidance_mesh` input for mesh-guided initialization.

Roadmap direction:

- strengthen FLAME-guided temporal consistency in fixed-topology runs
- improve mesh-to-splat coupling for cleaner long-shot stability
- keep guidance optional so non-FLAME workflows remain simple and fast

In short: FLAME integration is real and usable for experiments today, with ongoing work to make it a robust production mesh driver.

## 📁 Output Layout

Typical output structure:

```text
<output_root>/
  <prefix>_frame_XXXX.ply
  Distributed_Anchor/
    anchor_frame_XXXX.ply
    anchor_frame_XXXX.json
  MaskDebug/
    frame_XXXX_work/
      *_mask.png
      *_current.png
      *_previous.png
      *_blended.png
  realigned/
    <prefix>_frame_XXXX.ply
  smoothed/
    <prefix>_frame_XXXX.ply
```

## 🧩 Node Inventory

- FreeFlow Multi-Camera Loader
- FreeFlow COLMAP Anchor
- FreeFlow GS Engine (4D) [primary production node]
- FreeFlow 4D Adaptive Engine [compatibility/alternative path]
- FreeFlow Smart Grid Monitor
- FreeFlow Post-Process Smoother
- FreeFlow PLY Sequence Loader
- FreeFlow 3D Visualizer / COLMAP Visualizers
- FreeFlow 4D Player
- FreeFlow FlameTracker (FLAME path is experimental/in-progress)
- FreeFlow Mesh Loader

## 🛠️ Troubleshooting

- Workflow load error like `topo.includes is not a function`
  - Pull latest updates and reload browser; compatibility guards were added for legacy widget values.

- No `masks/` folder in Splatfacto blend mode
  - Expected. Blend mode uses computed masks internally and trains on blended images.

- Static regions still unstable
  - Use `Optical Flow (Robust)`
  - Raise `motion_sensitivity` toward `0.85-0.90`
  - Use `Blend Static From Previous (Recommended)`
  - Enable `save_mask_debug_images` and inspect coverage

- Splatfacto missing on macOS
  - Expected today: Splatfacto requires CUDA/gsplat.
  - Use Brush or OpenSplat on macOS, or run Splatfacto on a Windows/Linux CUDA machine.

- Preview controls not visible
  - `preview_interval`, `preview_camera_filter`, `eval_camera_index` appear only in `Save Preview Images` mode.

## 📚 References

- ITU-T H.264/AVC (motion-compensated prediction concepts): https://www.itu.int/rec/T-REC-H.264
- ITU-T H.265/HEVC (temporal prediction concepts): https://www.itu.int/rec/T-REC-H.265
- Farneback optical flow (conceptual basis for robust motion estimation in this pipeline)

## License

GNU General Public License v3.0
