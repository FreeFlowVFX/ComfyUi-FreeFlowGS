# Brush CLI Flags Reference

**Binary:** `brush.exe` / `brush_linux` / `brush_macos`
**Source:** ArthurBrussee/brush

## Usage

```
brush.exe [OPTIONS] [PATH_OR_URL]
```

## Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--total-steps` | 30000 | Total number of steps to train for |
| `--lr-mean` | 2e-5 | Start learning rate for the mean (position) parameters |
| `--lr-mean-end` | 2e-7 | End learning rate for the mean parameters |
| `--mean-noise-weight` | 50.0 | How much noise to add to mean params of low opacity gaussians |
| `--lr-coeffs-dc` | 2e-3 | Learning rate for the base SH (RGB) coefficients |
| `--lr-coeffs-sh-scale` | 20.0 | How much to divide LR by for higher SH orders |
| `--lr-opac` | 0.012 | Learning rate for the opacity parameter |
| `--lr-scale` | 7e-3 | Learning rate for the scale parameters |
| `--lr-scale-end` | 5e-3 | End learning rate for the scale parameters |
| `--lr-rotation` | 2e-3 | Learning rate for the rotation parameters |
| `--ssim-weight` | 0.2 | Weight of SSIM loss (compared to L1 loss) |
| `--opac-decay` | 0.004 | Factor of opacity decay (gradual opacity reduction) |
| `--scale-decay` | 0.002 | Factor of scale decay |
| `--aux-loss-time` | 0.9 | How long to apply aux losses (1 = full training) |

## Refine (Densification) Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-splats` | 10000000 | Max number of splats (upper bound) |
| `--refine-every` | 200 | Frequency of refinement where gaussians are replaced/densified |
| `--growth-grad-threshold` | 0.003 | Threshold to control splat growth. Lower = faster growth |
| `--growth-select-fraction` | 0.2 | Fraction of splats that grow when deemed necessary. Higher = more aggressive |
| `--growth-stop-iter` | 15000 | Iteration after which splat growth stops |
| `--match-alpha-weight` | 0.1 | Weight of L1 loss on alpha if input has transparency |
| `--lpips-loss-weight` | 0.0 | Perceptual loss weight (LPIPS) |

## Model Options

| Flag | Default | Description |
|------|---------|-------------|
| `--sh-degree` | 3 | Spherical harmonics degree of splats (0-3) |

## Dataset Options

| Flag | Default | Description |
|------|---------|-------------|
| `--max-frames` | (all) | Max number of frames to load |
| `--max-resolution` | 1920 | Max resolution of images to load |
| `--eval-split-every` | (none) | Create eval dataset by selecting every nth image |
| `--subsample-frames` | (none) | Load only every nth frame |
| `--subsample-points` | (none) | Load only every nth point from initial SfM data |

## Process Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | 42 | Random seed |
| `--start-iter` | 0 | Iteration to resume from |
| `--eval-every` | 1000 | Evaluate every this many steps |
| `--eval-save-to-disk` | false | Save rendered eval images to disk |
| `--export-every` | 5000 | Export PLY every this many steps |
| `--export-path` | . | Location for exported files |
| `--export-name` | export_{iter}.ply | Filename of exported PLY |
| `--with-viewer` | false | Spawn a viewer to visualize training |

## Rerun.io Options

| Flag | Default | Description |
|------|---------|-------------|
| `--rerun-enabled` | false | Enable rerun.io logging |
| `--rerun-log-train-stats-every` | 50 | How often to log training stats |
| `--rerun-log-splats-every` | (none) | How often to log full splat cloud (heavy) |
| `--rerun-max-img-size` | 512 | Max size of images logged to rerun |

---

## Important Note: opacity_reset_interval

The original 3DGS paper and Nerfstudio implementation use `opacity_reset_interval` to periodically 
reset opacity values and cull faint gaussians. **Brush does NOT have this flag.**

Brush uses a different approach:
- `--opac-decay` (default 0.004) - Gradually reduces opacity over training time
- Gaussians below MIN_OPACITY (1/255) are automatically pruned during refinement

The old FreeFlow code had `--opacity_reset_interval` in the UI but it was **never working** 
because Brush silently ignores unknown flags. This parameter has been removed.

---

## FreeFlow UI Parameter Mapping

| FreeFlow UI Parameter | Brush Flag | FreeFlow Default | Notes |
|-----------------------|------------|------------------|-------|
| `iterations` | `--total-steps` | 30000 | ✅ Implemented |
| `splat_count` | `--max-splats` | 500000 | ✅ Implemented |
| `learning_rate` | `--lr-mean` | 0.00002 | ✅ Implemented |
| `sh_degree` | `--sh-degree` | 3 | ✅ Implemented |
| `densification_interval` | `--refine-every` | 200 | ✅ Implemented |
| `densify_grad_threshold` | `--growth-grad-threshold` | 0.00004 | ✅ Lower than Brush default for more detail |
| `growth_select_fraction` | `--growth-select-fraction` | 0.1 | ✅ Lower than Brush default (0.2) for stability |
| `feature_lr` | `--lr-coeffs-dc` | 0.0025 | ✅ Implemented |
| `gaussian_lr` | `--lr-scale` | 0.00016 | ✅ **Much lower than Brush default (0.007) for 4D stability** |
| `opacity_lr` | `--lr-opac` | 0.01 | ✅ Implemented |
| `scale_loss_weight` | `--scale-loss-weight` | 1e-8 | ✅ Implemented (not in Brush defaults, but supported) |

### Why FreeFlow Uses Different Defaults

FreeFlow is optimized for **4D Gaussian Splat sequences** where temporal stability matters:

- **`gaussian_lr` = 0.00016** (vs Brush 0.007): Very low scale learning rate prevents splats from 
  changing size rapidly between frames, which would cause flickering/wobbling.

- **`densify_grad_threshold` = 0.00004** (vs Brush 0.003): Lower threshold creates more detailed 
  reconstructions by adding points more aggressively.

- **`growth_select_fraction` = 0.1** (vs Brush 0.2): More conservative growth for stability.

---

## Fixed Topology Mode Flags

When using Fixed (Cinema-Smooth) topology mode for frames > 0:
- `--growth-stop-iter 0` - Stops all splat growth immediately
- `--refine-every 999999` - Effectively disables refinement

---

## Recommended Settings for 4D

### Quick Preview
```
--total-steps 2000
--max-splats 100000
--lr-mean 0.00005
--lr-scale 0.00016
--sh-degree 2
```

### Production Dynamic Mode
```
--total-steps 6000
--max-splats 500000
--lr-mean 0.00002
--lr-scale 0.00016
--refine-every 300
--growth-grad-threshold 0.00008
--growth-select-fraction 0.05
--sh-degree 3
```

### Production Fixed Mode (Frame 0)
Train first frame with Dynamic settings, then lock topology for remaining frames.
