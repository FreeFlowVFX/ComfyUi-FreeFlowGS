import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * FreeFlow GS Engine - UI Extension
 * Handles conditional visibility of parameters based on engine selection.
 * 
 * Parameter Groups:
 * - Shared: visualize_training, preview_interval, topology_mode, iterations, sh_degree, etc.
 * - Brush-only: splat_count, learning_rate, densification_interval, etc.
 * - Splatfacto-only: splatfacto_variant, cull_alpha_thresh, etc.
 * - Conditional: preview_camera_filter (Brush + Save Preview), apply_smoothing (Fixed topology)
 */

app.registerExtension({
    name: "FreeFlow.GS_Engine",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FreeFlow_GS_Engine") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Helper to find widget by name
                const findWidget = (name) => this.widgets.find((w) => w.name === name);

                // ═══════════════════════════════════════════════════════════════
                // PARAMETER GROUPS
                // ═══════════════════════════════════════════════════════════════
                
                // Engine selector
                const engineWidget = findWidget("engine_backend");

                // --- Brush-specific params ---
                const brushParams = [
                    "splat_count",
                    "learning_rate", 
                    "densification_interval",
                    "densify_grad_threshold",
                    "growth_select_fraction",
                    "feature_lr",
                    "gaussian_lr",
                    "opacity_lr",
                    "scale_loss_weight",
                    "masking_method",
                    "motion_sensitivity",
                    // Brush-only preview controls
                    "preview_camera_filter",
                    "eval_camera_index"
                ];

                // --- Splatfacto-specific params ---
                const splatfactoParams = [
                    "splatfacto_variant",
                    "cull_alpha_thresh",
                    "splatfacto_densify_grad_thresh",
                    "use_scale_regularization"
                ];

                // --- Visualization params (show when NOT "Off") ---
                const vizWidget = findWidget("visualize_training");
                const previewParams = [
                    "preview_interval"
                    // preview_camera_filter and eval_camera_index are Brush-only, handled separately
                ];

                // --- Topology params (show when "Fixed") ---
                const topoWidget = findWidget("topology_mode");
                const fixedTopoParams = [
                    "apply_smoothing"
                ];

                // --- Distributed params (show when enabled) ---
                const distributedWidget = findWidget("distributed_anchor");
                const distributedParams = [
                    "distributed_anchor_path",
                    "distributed_anchor_frame",
                    "warmup_frames"
                ];

                // Capture all widgets initially
                const allWidgets = [...this.widgets];
                this._freeflow_all_widgets = allWidgets;

                // ═══════════════════════════════════════════════════════════════
                // SERIALIZATION HOOKS
                // ═══════════════════════════════════════════════════════════════
                
                const onSerialize = this.onSerialize;
                this.onSerialize = function (o) {
                    if (onSerialize) onSerialize.apply(this, arguments);
                    if (this._freeflow_all_widgets) {
                        o.widgets_values = this._freeflow_all_widgets.map(w => w.value);
                    }
                };

                // Smart restoration for version mismatches
                const onConfigure = this.onConfigure;
                this.onConfigure = function (o) {
                    if (onConfigure) onConfigure.apply(this, arguments);

                    if (this._freeflow_all_widgets && o.widgets_values && 
                        o.widgets_values.length < this._freeflow_all_widgets.length) {
                        console.log("[FreeFlow GS] Detected version mismatch. Attempting smart restoration...");
                        
                        const saved = o.widgets_values;
                        let savedIdx = 0;
                        const currentVals = {};

                        // Defaults
                        currentVals["engine_backend"] = "Brush (Fast)";
                        currentVals["visualize_training"] = "Off";
                        currentVals["topology_mode"] = "Dynamic (Default-Flicker)";
                        currentVals["distributed_anchor"] = false;

                        for (const w of this._freeflow_all_widgets) {
                            const engine = currentVals["engine_backend"];
                            const isBrush = engine && engine.includes("Brush");
                            const isSplatfacto = engine && (engine.includes("Splatfacto") || engine.includes("Nerfstudio"));
                            const viz = currentVals["visualize_training"];
                            const topo = currentVals["topology_mode"];
                            const dist = currentVals["distributed_anchor"];

                            let visible = true;
                            if (brushParams.includes(w.name)) {
                                visible = isBrush;
                            } else if (splatfactoParams.includes(w.name)) {
                                visible = isSplatfacto;
                            } else if (previewParams.includes(w.name)) {
                                visible = (viz !== "Off");
                            } else if (fixedTopoParams.includes(w.name)) {
                                visible = (topo && topo.includes("Fixed"));
                            } else if (distributedParams.includes(w.name)) {
                                visible = dist;
                            }

                            if (visible && savedIdx < saved.length) {
                                const val = saved[savedIdx++];
                                w.value = val;
                                currentVals[w.name] = val;
                            } else {
                                currentVals[w.name] = w.value;
                            }
                        }
                    }
                };

                // ═══════════════════════════════════════════════════════════════
                // MAIN VISIBILITY UPDATE FUNCTION
                // ═══════════════════════════════════════════════════════════════
                
                const updateVisibility = () => {
                    const engine = engineWidget ? engineWidget.value : "Brush (Fast)";
                    const isBrush = engine && engine.includes("Brush");
                    const isSplatfacto = engine && (engine.includes("Splatfacto") || engine.includes("Nerfstudio"));
                    const isOpenSplat = engine && engine.includes("OpenSplat");

                    const vizMode = vizWidget ? vizWidget.value : "Off";
                    const showPreviewParams = (vizMode !== "Off");
                    const isSavePreview = (vizMode === "Save Preview Images");

                    const topoMode = topoWidget ? topoWidget.value : "";
                    const isFixedTopo = (topoMode && topoMode.includes("Fixed"));

                    const showDistributed = distributedWidget ? distributedWidget.value : false;

                    // Filter widgets based on current state
                    this.widgets = allWidgets.filter(w => {
                        // --- Engine-specific params ---
                        if (brushParams.includes(w.name)) {
                            // Special case: preview_camera_filter and eval_camera_index
                            // Only show if Brush AND Save Preview Images mode
                            if (w.name === "preview_camera_filter" || w.name === "eval_camera_index") {
                                return isBrush && isSavePreview;
                            }
                            return isBrush;
                        }
                        if (splatfactoParams.includes(w.name)) {
                            return isSplatfacto;
                        }
                        
                        // --- Preview interval (shared, but only when viz is not Off) ---
                        if (previewParams.includes(w.name)) {
                            return showPreviewParams;
                        }
                        
                        // --- Topology params (only when Fixed mode) ---
                        if (fixedTopoParams.includes(w.name)) {
                            return isFixedTopo;
                        }
                        
                        // --- Distributed params (only when enabled) ---
                        if (distributedParams.includes(w.name)) {
                            return showDistributed;
                        }
                        
                        // --- OpenSplat currently uses shared params only ---
                        // No OpenSplat-specific params yet
                        
                        return true; // Show all other params
                    });

                    this.setSize([this.size[0], this.computeSize([this.size[0], this.size[1]])[1]]);
                    if (app.canvas) app.canvas.setDirty(true, true);
                };

                // ═══════════════════════════════════════════════════════════════
                // REGISTER CALLBACKS
                // ═══════════════════════════════════════════════════════════════
                
                if (engineWidget) {
                    engineWidget.callback = () => updateVisibility();
                }
                if (vizWidget) {
                    vizWidget.callback = () => updateVisibility();
                }
                if (topoWidget) {
                    topoWidget.callback = () => updateVisibility();
                }
                if (distributedWidget) {
                    distributedWidget.callback = () => updateVisibility();
                }

                // Initial visibility update
                setTimeout(() => updateVisibility(), 100);

                return r;
            };
        }
    },

    // ═══════════════════════════════════════════════════════════════
    // PREVIEW IMAGE DISPLAY
    // ═══════════════════════════════════════════════════════════════
    
    setup() {
        // Global listener for preview updates
        api.addEventListener("freeflow_preview", (event) => {
            const data = event.detail;
            const nodeId = data.node;

            const node = app.graph.getNodeById(nodeId);
            if (!node) return;

            // Only handle GS_Engine nodes
            if (node.type !== "FreeFlow_GS_Engine") return;

            const img = new Image();
            img.src = api.apiURL(`/view?filename=${data.filename}&subfolder=${data.subfolder}&type=${data.type}`);
            img.onload = () => {
                node._freeflow_preview = [img];

                const ratio = img.width / img.height;
                const w = node.size[0] - 20;
                const h = w / ratio;

                const sz = node.computeSize();
                const widgetsHeight = sz[1];
                const requiredHeight = widgetsHeight + h + 20;

                if (node.size[1] < requiredHeight) {
                    node.setSize([node.size[0], requiredHeight]);
                }

                app.graph.setDirtyCanvas(true, true);
            };

            // Custom draw for preview
            if (!node._has_preview_draw) {
                const origDraw = node.onDrawForeground;
                node.onDrawForeground = function (ctx) {
                    if (origDraw) origDraw.apply(this, arguments);

                    if (this._freeflow_preview && this._freeflow_preview.length > 0) {
                        const img = this._freeflow_preview[0];
                        const ratio = img.width / img.height;
                        const w = this.size[0] - 20;
                        const h = w / ratio;
                        const sz = this.computeSize();
                        const y = sz[1] + 10;

                        ctx.drawImage(img, 10, y, w, h);
                    }
                };
                node._has_preview_draw = true;
            }
        });
    }
});
