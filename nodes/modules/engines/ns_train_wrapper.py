"""
FreeFlow ns-train Wrapper
Handles cross-frame warm start and PyTorch 2.6 compatibility.

CROSS-FRAME WARM START (Option C: Checkpoint Points as Initialization)
----------------------------------------------------------------------
The key insight: Gaussian count changes during densification (50k→200k+).
If we just load checkpoint weights into a fresh model (initialized from
COLMAP sparse), shapes mismatch and nothing loads.

FIX: Before nerfstudio starts, we extract the point cloud FROM the
checkpoint and write it as points3D.txt. Nerfstudio then initializes
the model with the SAME Gaussian count as the checkpoint. When we
inject the checkpoint weights post-setup, ALL shapes match perfectly —
preserving positions, scales, rotations, opacities, and SH coefficients.

Flow:
  1. Parse --ff-load-weights <checkpoint_dir> (our custom flag)
  2. Load checkpoint, extract _model.means (positions) and colors
  3. Find --data <path> in args → write to <path>/sparse/0/points3D.txt
  4. Nerfstudio initializes model from these points (correct Gaussian count)
  5. After setup, inject ALL _model.* weights (shapes now match!)
  6. Training starts from step 0 on the NEW frame's images

PYTORCH 2.6 FIX
----------------
Patches save_checkpoint to handle optimizer state_dict KeyError when
Gaussian count changes via densification.

Usage: python ns_train_wrapper.py --ff-load-weights <ckpt_dir> <ns-train args...>
"""
import sys
import os


def main():
    # --- Parse our custom arg (before nerfstudio sees it) ---
    checkpoint_dir = None
    data_path = None
    clean_args = []
    i = 1  # Skip sys.argv[0] (this script)
    while i < len(sys.argv):
        if sys.argv[i] == "--ff-load-weights":
            checkpoint_dir = sys.argv[i + 1]
            i += 2
        else:
            # Also capture --data path for seeding points3D.txt
            if sys.argv[i] == "--data" and i + 1 < len(sys.argv):
                data_path = sys.argv[i + 1]
            clean_args.append(sys.argv[i])
            i += 1
    
    # Set sys.argv for nerfstudio
    sys.argv = ["ns-train"] + clean_args
    
    import torch
    import nerfstudio.engine.trainer as trainer_module
    
    # --- STEP 1: Seed points3D.txt from checkpoint ---
    _ff_model_state = None
    
    if checkpoint_dir:
        from pathlib import Path
        ckpt_dir = Path(checkpoint_dir)

        def _extract_model_state(pipeline_state):
            model_state = {}
            if not isinstance(pipeline_state, dict):
                return model_state
            for key, value in pipeline_state.items():
                clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
                if clean_key.startswith("_model.") or clean_key.startswith("model."):
                    model_state[clean_key] = value
            return model_state

        def _select_means_key(model_state):
            preferred_keys = [
                "_model.gauss_params.means",
                "model.gauss_params.means",
                "_model.means",
                "model.means",
            ]
            for key in preferred_keys:
                if key in model_state:
                    return key
            for key in model_state:
                if key.endswith(".gauss_params.means") or key.endswith(".means"):
                    return key
            return None

        def _validate_means_tensor(means_tensor):
            if not isinstance(means_tensor, torch.Tensor):
                return None, None, f"means is not a torch.Tensor (got {type(means_tensor).__name__})"
            try:
                means_cpu = means_tensor.detach().cpu().float()
            except Exception as ex:
                return None, None, f"failed to move means to CPU ({ex})"

            if means_cpu.ndim != 2:
                return None, None, f"unexpected means rank {means_cpu.ndim} (expected 2)"
            if means_cpu.shape[0] <= 0:
                return None, None, "means has zero rows"
            if means_cpu.shape[1] < 3:
                return None, None, f"unexpected means shape {tuple(means_cpu.shape)} (expected Nx3+)"

            xyz = means_cpu[:, :3]
            finite_mask = torch.isfinite(xyz).all(dim=1)
            finite_count = int(finite_mask.sum().item())
            if finite_count <= 0:
                return None, None, "no finite xyz rows in means tensor"
            if finite_count != int(xyz.shape[0]):
                return None, None, f"means has {int(xyz.shape[0]) - finite_count} non-finite rows"

            return xyz, finite_mask, None
        
        # Find candidate checkpoints (newest first), validate, and pick first good one.
        ckpt_files = sorted(ckpt_dir.glob("step-*.ckpt"))
        if not ckpt_files:
            print(f"[FreeFlow] WARNING: No checkpoint files in {ckpt_dir}")
        else:
            selected_ckpt_path = None
            selected_model_state = None
            selected_means_xyz = None
            selected_finite_mask = None
            selected_colors_tensor = None

            for ckpt_path in reversed(ckpt_files):
                print(f"[FreeFlow] Checking checkpoint candidate: {ckpt_path.name}")

                try:
                    loaded_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                except Exception as ex:
                    print(f"[FreeFlow] WARNING: Failed to load {ckpt_path.name} ({type(ex).__name__}: {ex})")
                    continue

                model_state = _extract_model_state(loaded_state.get("pipeline", {}))
                if not model_state:
                    print(f"[FreeFlow] WARNING: {ckpt_path.name} has no model tensors in pipeline state")
                    continue

                means_key = _select_means_key(model_state)
                if not means_key:
                    print(f"[FreeFlow] WARNING: {ckpt_path.name} has no means tensor key")
                    continue

                means_xyz, finite_mask, means_error = _validate_means_tensor(model_state.get(means_key))
                if means_error:
                    print(f"[FreeFlow] WARNING: Skipping {ckpt_path.name}: {means_error}")
                    continue

                colors_tensor = None
                for key, value in model_state.items():
                    if "features_dc" in key and isinstance(value, torch.Tensor):
                        colors_tensor = value
                        break

                selected_ckpt_path = ckpt_path
                selected_model_state = model_state
                selected_means_xyz = means_xyz
                selected_finite_mask = finite_mask
                selected_colors_tensor = colors_tensor
                print(f"[FreeFlow] Using checkpoint: {ckpt_path}")
                print(f"[FreeFlow] Extracted {len(model_state)} model weight tensors")
                break

            if selected_ckpt_path and data_path:
                if selected_means_xyz is None or selected_finite_mask is None or selected_model_state is None:
                    print("[FreeFlow] WARNING: Internal checkpoint-selection state incomplete; falling back to sparse init")
                else:
                    num_gaussians = selected_means_xyz.shape[0]
                    valid_points = int(selected_finite_mask.sum().item())
                    skipped_invalid = int(num_gaussians - valid_points)
                    print(f"[FreeFlow] Checkpoint has {num_gaussians} Gaussians ({valid_points} finite xyz rows)")

                    # Write points3D.txt with checkpoint positions
                    # CRITICAL: Delete points3D.bin if it exists — nerfstudio checks .bin FIRST
                    # and would read the original COLMAP sparse instead of our checkpoint seed.
                    points3d_path = Path(data_path) / "sparse" / "0" / "points3D.txt"
                    points3d_path.parent.mkdir(parents=True, exist_ok=True)

                    bin_file = points3d_path.parent / "points3D.bin"
                    if bin_file.exists():
                        bin_file.unlink()
                        print("[FreeFlow] Deleted points3D.bin (forcing nerfstudio to read our .txt)")

                    SH_C0 = 0.28209479177387814

                    with open(points3d_path, "w") as f:
                        f.write("# 3D point list with one line of data per point:\n")
                        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                        f.write(f"# Number of points: {valid_points}, mean track length: 0\n")

                        point_id = 1
                        for idx in range(num_gaussians):
                            if not bool(selected_finite_mask[idx].item()):
                                continue

                            x, y, z = selected_means_xyz[idx].tolist()

                            # Approximate RGB from SH DC coefficient
                            r, g, b = 128, 128, 128
                            if (
                                isinstance(selected_colors_tensor, torch.Tensor)
                                and selected_colors_tensor.ndim >= 2
                                and idx < selected_colors_tensor.shape[0]
                            ):
                                try:
                                    dc = selected_colors_tensor[idx]
                                    if dc.dim() > 1:
                                        dc = dc.squeeze(0)
                                    if dc.numel() >= 3:
                                        dc_rgb = dc[:3].detach().cpu().float()
                                        if bool(torch.isfinite(dc_rgb).all().item()):
                                            rf = dc_rgb[0].item() * SH_C0 + 0.5
                                            gf = dc_rgb[1].item() * SH_C0 + 0.5
                                            bf = dc_rgb[2].item() * SH_C0 + 0.5
                                            r = min(255, max(0, int(rf * 255)))
                                            g = min(255, max(0, int(gf * 255)))
                                            b = min(255, max(0, int(bf * 255)))
                                except Exception:
                                    pass

                            f.write(f"{point_id} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")
                            point_id += 1

                    print(
                        f"[FreeFlow] Seeded points3D.txt with {valid_points} points from {selected_ckpt_path.name}"
                        + (f" (skipped {skipped_invalid} invalid)" if skipped_invalid > 0 else "")
                    )
                    print(f"[FreeFlow]   → {points3d_path}")

                    # Store model state for injection after pipeline setup
                    _ff_model_state = selected_model_state
            elif selected_ckpt_path and not data_path:
                print("[FreeFlow] WARNING: --data path not found in args, cannot seed points3D.txt")
                print("[FreeFlow] WARNING: Falling back to sparse init (warm-start injection skipped)")
            else:
                print(f"[FreeFlow] WARNING: No valid checkpoint found in {ckpt_dir}")
                print("[FreeFlow] WARNING: Falling back to sparse init for this frame")
    
    # --- STEP 2: Patch Trainer.setup to inject weights post-creation ---
    if _ff_model_state:
        _original_setup = trainer_module.Trainer.setup
        
        def _patched_setup(self, test_mode="val"):
            _original_setup(self, test_mode)
            
            # CRITICAL: Reset step counter to 0 to ensure fresh learning rate schedule
            # If we continue from checkpoint's step, LR would be near zero (no updates)
            if hasattr(self, 'step') and self.step > 0:
                print(f"[FreeFlow] Resetting step counter from {self.step} to 0 for fresh training")
                self.step = 0
                # Reset global step in optimizers if they track it
                if hasattr(self, 'optimizers') and self.optimizers:
                    for opt_name, optimizer in self.optimizers.optimizers.items():
                        if hasattr(optimizer, '_step_count'):
                            optimizer._step_count = 0
                # Reset learning rate schedulers to step 0
                if hasattr(self, 'optimizers') and self.optimizers and hasattr(self.optimizers, 'schedulers'):
                    for sched_name, scheduler in self.optimizers.schedulers.items():
                        if hasattr(scheduler, '_step_count'):
                            scheduler._step_count = 0
                        if hasattr(scheduler, 'last_epoch'):
                            scheduler.last_epoch = 0
                        print(f"[FreeFlow] Reset scheduler '{sched_name}' to step 0")
            
            # Now the pipeline is created with our seeded points (correct Gaussian count).
            # Inject the checkpoint model weights — shapes should match!
            try:
                current_state = self.pipeline.state_dict()
                old_means_id = None
                try:
                    old_means_id = id(self.pipeline.model.means)
                except Exception:
                    pass
                
                loaded = 0
                skipped_shape = 0
                skipped_missing = 0
                
                compatible_state = {}
                for key, value in _ff_model_state.items():
                    if key in current_state:
                        if value.shape == current_state[key].shape:
                            compatible_state[key] = value
                            loaded += 1
                        else:
                            skipped_shape += 1
                            if skipped_shape <= 5:
                                print(f"[FreeFlow]   Shape mismatch: {key} "
                                      f"(ckpt: {value.shape} vs model: {current_state[key].shape})")
                    else:
                        skipped_missing += 1
                
                if compatible_state:
                    self.pipeline.load_state_dict(compatible_state, strict=False)

                    # IMPORTANT: Splatfacto load_state_dict can recreate parameter tensors
                    # (means/scales/opacities/etc). Rebuild optimizers + callbacks so they
                    # point to the new Parameter objects instead of stale references.
                    try:
                        from nerfstudio.engine.callbacks import TrainingCallbackAttributes

                        self.optimizers = self.setup_optimizers()
                        self.callbacks = self.pipeline.get_training_callbacks(
                            TrainingCallbackAttributes(
                                optimizers=self.optimizers,
                                grad_scaler=self.grad_scaler,
                                pipeline=self.pipeline,
                                trainer=self,
                            )
                        )

                        new_means = self.pipeline.model.means
                        new_means_id = id(new_means)
                        means_in_optimizer = False
                        bound_groups = []
                        for group_name, opt in self.optimizers.optimizers.items():
                            for pg in opt.param_groups:
                                if any(p is new_means for p in pg.get('params', [])):
                                    means_in_optimizer = True
                                    bound_groups.append(group_name)
                                    break

                        print(f"[FreeFlow] Optimizers rebound after warm start injection")
                        if old_means_id is not None:
                            print(f"[FreeFlow] Param IDs: means old={old_means_id} new={new_means_id}")
                        print(f"[FreeFlow] Means bound in optimizer groups: {bound_groups if bound_groups else 'NONE'}")
                        if not means_in_optimizer:
                            print("[FreeFlow] WARNING: means parameter is NOT bound to any optimizer group")
                    except Exception as rebind_ex:
                        print(f"[FreeFlow] WARNING: Failed to rebuild optimizers/callbacks ({rebind_ex})")
                
                print(f"[FreeFlow] Weight injection: {loaded} loaded, "
                      f"{skipped_shape} shape mismatches, {skipped_missing} missing")
                
                if loaded > 0 and skipped_shape == 0:
                    print(f"[FreeFlow] ✅ ALL model weights loaded successfully — full warm start!")
                elif loaded > 0:
                    print(f"[FreeFlow] ⚠️ Partial warm start ({skipped_shape} shape mismatches)")
                else:
                    print(f"[FreeFlow] ❌ No weights loaded — training from scratch")
                
                # Diagnostic: print model state fingerprint + check trainability
                try:
                    means = self.pipeline.model.means
                    print(f"[FreeFlow] 📊 BEFORE training: means range=({means.data.min():.4f}, {means.data.max():.4f}), "
                          f"mean={means.data.mean():.4f}, count={means.shape[0]}, "
                          f"requires_grad={means.requires_grad}")
                    # List ALL optimizer groups and their param counts
                    if hasattr(self, 'optimizers') and hasattr(self.optimizers, 'optimizers'):
                        for name, opt in self.optimizers.optimizers.items():
                            n_params = sum(len(pg['params']) for pg in opt.param_groups)
                            lr = opt.param_groups[0].get('lr', 'n/a') if opt.param_groups else 'n/a'
                            print(f"[FreeFlow] 📊 Optimizer group '{name}': {n_params} params, lr={lr}")
                    else:
                        print(f"[FreeFlow] 📊 WARNING: No optimizers found on trainer!")
                    
                    # Register gradient hook on means — fires DIRECTLY during backprop
                    _grad_log_count = [0]
                    def _means_grad_hook(grad):
                        _grad_log_count[0] += 1
                        if _grad_log_count[0] <= 3 or _grad_log_count[0] % 1000 == 0:
                            print(f"[FreeFlow] 📊 GRAD step {_grad_log_count[0]}: "
                                  f"grad_mean={grad.mean():.8f}, grad_norm={grad.norm():.4f}, "
                                  f"means_mean={means.data.mean():.6f}")
                            import sys; sys.stdout.flush()
                    means.register_hook(_means_grad_hook)
                    print(f"[FreeFlow] 📊 Gradient hook registered on means parameter")
                    
                    # Check if model has fixed topology flags
                    if hasattr(self.pipeline.model.config, 'stop_split_at'):
                        stop_split = self.pipeline.model.config.stop_split_at
                        print(f"[FreeFlow] 📊 Topology config: stop_split_at={stop_split}")
                        if stop_split == 0:
                            print(f"[FreeFlow] ⚠️ FIXED TOPOLOGY MODE: Densification disabled (Gaussians won't split/clone)")
                    if hasattr(self.pipeline.model.config, 'refine_every'):
                        refine_every = self.pipeline.model.config.refine_every
                        print(f"[FreeFlow] 📊 Refinement interval: refine_every={refine_every}")
                        if refine_every >= 999999:
                            print(f"[FreeFlow] ⚠️ REFINEMENT DISABLED: No densification will occur")
                    
                except Exception as ex:
                    print(f"[FreeFlow] 📊 BEFORE diagnostic error: {ex}")
                    
            except Exception as e:
                print(f"[FreeFlow] WARNING: Weight injection failed ({e})")
        
        trainer_module.Trainer.setup = _patched_setup
    
    # --- STEP 3: Fix PyTorch 2.6 save_checkpoint crash ---
    _original_save_checkpoint = trainer_module.Trainer.save_checkpoint

    def _patched_save_checkpoint(self, step):
        print(f"[FreeFlow] 💾 save_checkpoint called at step {step}")
        import sys; sys.stdout.flush()
        
        try:
            return _original_save_checkpoint(self, step)
        except (KeyError, RuntimeError) as e:
            print(f"\n[FreeFlow] Optimizer state save failed ({type(e).__name__}: {e})")
            print(f"[FreeFlow] Saving checkpoint WITHOUT optimizer state...")
            
            ckpt_dir = self.checkpoint_dir
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"step-{step:09d}.ckpt"
            
            if hasattr(self.pipeline, "module"):
                pipeline_state = self.pipeline.module.state_dict()
            else:
                pipeline_state = self.pipeline.state_dict()
            
            save_dict = {
                "step": step,
                "pipeline": pipeline_state,
                "optimizers": {},
            }
            
            try:
                save_dict["schedulers"] = {
                    k: v.state_dict() for k, v in self.optimizers.schedulers.items()
                }
            except Exception:
                save_dict["schedulers"] = {}
            
            torch.save(save_dict, ckpt_path)
            print(f"[FreeFlow] Checkpoint saved: {ckpt_path}")

    trainer_module.Trainer.save_checkpoint = _patched_save_checkpoint
    
    # --- RUN ns-train ---
    from nerfstudio.scripts.train import entrypoint
    entrypoint()


if __name__ == "__main__":
    main()
