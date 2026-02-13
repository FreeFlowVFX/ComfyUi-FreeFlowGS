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
        
        # Find latest checkpoint
        ckpt_files = sorted(ckpt_dir.glob("step-*.ckpt"))
        if not ckpt_files:
            print(f"[FreeFlow] WARNING: No checkpoint files in {ckpt_dir}")
        else:
            ckpt_path = ckpt_files[-1]
            print(f"[FreeFlow] Loading checkpoint: {ckpt_path}")
            
            loaded_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            pipeline_state = loaded_state.get("pipeline", {})
            
            # Extract model weights (skip datamanager)
            model_state = {}
            for key, value in pipeline_state.items():
                clean_key = key.replace("module.", "", 1) if key.startswith("module.") else key
                if clean_key.startswith("_model.") or clean_key.startswith("model."):
                    model_state[clean_key] = value
            
            print(f"[FreeFlow] Extracted {len(model_state)} model weight tensors")
            
            # Find the means (positions) tensor
            means_key = None
            for k in model_state:
                if k.endswith(".means") or k.endswith(".gauss_params.means"):
                    means_key = k
                    break
            
            if means_key and data_path:
                means = model_state[means_key]
                num_gaussians = means.shape[0]
                print(f"[FreeFlow] Checkpoint has {num_gaussians} Gaussians")
                
                # Find color (features_dc) for approximate RGB
                colors_key = None
                for k in model_state:
                    if "features_dc" in k:
                        colors_key = k
                        break
                
                # Write points3D.txt with checkpoint positions
                # CRITICAL: Delete points3D.bin if it exists — nerfstudio checks .bin FIRST
                # and would read the original COLMAP sparse (50k) instead of our checkpoint (200k)
                points3d_path = Path(data_path) / "sparse" / "0" / "points3D.txt"
                points3d_path.parent.mkdir(parents=True, exist_ok=True)
                
                bin_file = points3d_path.parent / "points3D.bin"
                if bin_file.exists():
                    bin_file.unlink()
                    print(f"[FreeFlow] Deleted points3D.bin (forcing nerfstudio to read our .txt)")
                
                SH_C0 = 0.28209479177387814
                
                with open(points3d_path, "w") as f:
                    f.write("# 3D point list with one line of data per point:\n")
                    f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                    f.write(f"# Number of points: {num_gaussians}, mean track length: 0\n")
                    
                    for idx in range(num_gaussians):
                        x, y, z = means[idx].tolist()
                        
                        # Skip NaN points
                        if x != x or y != y or z != z:
                            continue
                        
                        # Approximate RGB from SH DC coefficient
                        r, g, b = 128, 128, 128
                        if colors_key and colors_key in model_state:
                            dc = model_state[colors_key][idx]
                            # features_dc shape is typically [1, 3] or [3]
                            if dc.dim() > 1:
                                dc = dc.squeeze(0)
                            rf = dc[0].item() * SH_C0 + 0.5
                            gf = dc[1].item() * SH_C0 + 0.5
                            bf = dc[2].item() * SH_C0 + 0.5
                            r = min(255, max(0, int(rf * 255)))
                            g = min(255, max(0, int(gf * 255)))
                            b = min(255, max(0, int(bf * 255)))
                        
                        f.write(f"{idx + 1} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} 0.0\n")
                
                print(f"[FreeFlow] Seeded points3D.txt with {num_gaussians} points from checkpoint")
                print(f"[FreeFlow]   → {points3d_path}")
                
                # Store model state for injection after pipeline setup
                _ff_model_state = model_state
            elif not means_key:
                print(f"[FreeFlow] WARNING: No 'means' tensor found in checkpoint keys:")
                for k in sorted(model_state.keys())[:20]:
                    print(f"[FreeFlow]   {k}: {model_state[k].shape}")
            elif not data_path:
                print(f"[FreeFlow] WARNING: --data path not found in args, cannot seed points3D.txt")
    
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
