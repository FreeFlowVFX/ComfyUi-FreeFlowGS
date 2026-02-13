import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

/**
 * FreeFlow GS Engine - UI Extension
 * Handles conditional visibility of parameters based on engine selection.
 * 
 * Parameter Groups:
 * - Shared: spawn_native_gui, preview_interval, topology_mode, iterations, sh_degree, etc.
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

                // Visualization widget (needed for filtering options and visibility)
                const vizWidget = findWidget("visualize_training");

                // --- Preview params (Brush + Save Preview mode only) ---
                const previewParams = [
                    "preview_interval"
                ];

                // --- Brush-specific params ---
                const brushParams = [
                    "visualize_training",
                    "preview_interval",
                    "splat_count",
                    "learning_rate",
                    "densification_interval",
                    "densify_grad_threshold",
                    "growth_select_fraction",
                    "feature_lr",
                    "gaussian_lr",
                    "opacity_lr",
                    "scale_loss_weight",
                    // Brush-only preview controls
                    "preview_camera_filter",
                    "eval_camera_index"
                ];

                // --- Masking params (shared across all engines) ---
                const maskingParams = [
                    "masking_method",
                    "motion_sensitivity"
                ];

                // --- Splatfacto-specific params ---
                const splatfactoParams = [
                    "splatfacto_viewer",
                    "splatfacto_variant",
                    "cull_alpha_thresh",
                    "splatfacto_densify_grad_thresh",
                    "use_scale_regularization",
                    "max_gs_num",
                    "refine_every",
                    "warmup_length",
                    "num_downscales",
                    "cull_screen_size",
                    "split_screen_size",
                    "sh_degree_interval",
                    "background_color"
                ];


                // --- Topology params (show when "Fixed") ---
                const topoWidget = findWidget("topology_mode");
                const fixedTopoParams = [
                    "realign_topology",
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
                            if (maskingParams.includes(w.name)) {
                                visible = true; // Always visible for all engines
                            } else if (brushParams.includes(w.name)) {
                                visible = isBrush;
                            } else if (splatfactoParams.includes(w.name)) {
                                visible = isSplatfacto;
                            } else if (previewParams.includes(w.name)) {
                                visible = isBrush;
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

                const getComboOptions = (widget) => {
                    if (!widget || !widget.options) return [];
                    if (Array.isArray(widget.options)) return [...widget.options];
                    if (Array.isArray(widget.options.values)) return [...widget.options.values];
                    return [];
                };

                const setComboOptions = (widget, values) => {
                    if (!widget) return;
                    if (Array.isArray(widget.options)) {
                        widget.options = [...values];
                    } else if (widget.options && Array.isArray(widget.options.values)) {
                        widget.options.values = [...values];
                    }
                };

                const updateVisibility = () => {
                    try {
                        const engine = engineWidget ? engineWidget.value : "Brush (Fast)";
                        const isBrush = engine && engine.includes("Brush");
                        const isSplatfacto = engine && (engine.includes("Splatfacto") || engine.includes("Nerfstudio"));
                        const isOpenSplat = engine && engine.includes("OpenSplat");

                        // Filter visualization dropdown options based on engine
                        if (vizWidget && vizWidget.options) {
                            if (!vizWidget._originalOptions) {
                                vizWidget._originalOptions = getComboOptions(vizWidget);
                            }

                            let allowedOptions;
                            if (isBrush) {
                                allowedOptions = ["Off", "Save Preview Images", "Spawn Native GUI"];
                            } else if (isSplatfacto) {
                                allowedOptions = ["Off", "Spawn Native GUI"];
                            } else {
                                allowedOptions = ["Off"];
                            }

                            const filtered = vizWidget._originalOptions.filter(opt => allowedOptions.includes(opt));
                            setComboOptions(vizWidget, filtered);

                            if (!allowedOptions.includes(vizWidget.value)) {
                                vizWidget.value = "Off";
                            }
                        }

                        const vizMode = vizWidget ? vizWidget.value : "Off";
                        const isSavePreview = (vizMode === "Save Preview Images");

                        const topoMode = topoWidget ? topoWidget.value : "";
                        const isFixedTopo = (topoMode && topoMode.includes("Fixed"));

                        const showDistributed = distributedWidget ? distributedWidget.value : false;

                        // Filter widgets based on current state
                        this.widgets = allWidgets.filter(w => {
                            // --- Masking params (show for ALL engines) ---
                            if (maskingParams.includes(w.name)) {
                                return true; // Always visible for all engines
                            }

                            // --- Engine-specific params ---
                            if (brushParams.includes(w.name)) {
                                // preview_camera_filter and eval_camera_index only in Save Preview mode
                                if (w.name === "preview_camera_filter" || w.name === "eval_camera_index") {
                                    return isBrush && isSavePreview;
                                }
                                return isBrush;
                            }
                            if (splatfactoParams.includes(w.name)) {
                                return isSplatfacto;
                            }

                            // --- Preview interval (Brush + Save Preview mode only) ---
                            if (previewParams.includes(w.name)) {
                                return isBrush && isSavePreview;
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
                    } catch (err) {
                        console.error("[FreeFlow GS] Visibility update error:", err);
                    }
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
