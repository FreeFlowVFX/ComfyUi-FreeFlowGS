import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

app.registerExtension({
    name: "FreeFlow.AdaptiveEngine",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FreeFlow_AdaptiveEngine") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                // Helper to find widget by name
                const findWidget = (name) => this.widgets.find((w) => w.name === name);

                const vizWidget = findWidget("visualize_training");
                const intervalWidget = findWidget("preview_interval");
                const filterWidget = findWidget("preview_camera_filter");
                const evalCameraWidget = findWidget("eval_camera_index"); // Camera index selector
                const topoWidget = findWidget("topology_mode");
                const maskingMethodWidget = findWidget("masking_method");
                const existingFramesPolicyWidget = findWidget("existing_frames_policy");

                const createAnchorWidget = findWidget("distributed_anchor");
                const anchorPathWidget = findWidget("distributed_anchor_path");
                const anchorFrameWidget = findWidget("distributed_anchor_frame");
                const warmupFramesWidget = findWidget("warmup_frames");

                // Capture all widgets initially to preserve order and references
                const allWidgets = [...this.widgets];
                this._freeflow_all_widgets = allWidgets; // Persist for serialization

                // Hook Serialization to save ALL values, including hidden ones
                // This prevents data loss/shifting when saving with hidden params
                const onSerialize = this.onSerialize;
                this.onSerialize = function (o) {
                    if (onSerialize) onSerialize.apply(this, arguments);

                    if (this._freeflow_all_widgets) {
                        // Overwrite the truncated widgets_values with the full list
                        o.widgets_values = this._freeflow_all_widgets.map(w => w.value);
                    }
                };

                // Safe normalization for old workflows where combo values can be numeric
                const asStr = (v, fallback = "") => {
                    if (v === undefined || v === null) return fallback;
                    return String(v);
                };
                const asBool = (v) => {
                    if (typeof v === "boolean") return v;
                    if (typeof v === "number") return v !== 0;
                    if (typeof v === "string") {
                        const s = v.trim().toLowerCase();
                        return s === "true" || s === "1" || s === "yes";
                    }
                    return !!v;
                };

                // Smart Restoration for Truncated Saves
                const onConfigure = this.onConfigure;
                this.onConfigure = function (o) {
                    if (onConfigure) onConfigure.apply(this, arguments);

                    if (this._freeflow_all_widgets && o.widgets_values && o.widgets_values.length < this._freeflow_all_widgets.length) {
                        console.log("[FreeFlow] Detected version mismatch/truncation. Attempting smart restoration...");
                        const saved = o.widgets_values;
                        const full = [];
                        let savedIdx = 0;
                        const currentVals = {};

                        // Default Controller Values
                        currentVals["visualize_training"] = "Off";
                        currentVals["topology_mode"] = "Dynamic (Default-Flicker)";
                        currentVals["distributed_anchor"] = false;
                        currentVals["masking_method"] = "None (No Masking)";

                        for (const w of this._freeflow_all_widgets) {
                            // Standard Visibility Logic (Must match updateVisibility)
                            const viz = asStr(currentVals["visualize_training"], "Off");
                            const topo = asStr(currentVals["topology_mode"]);
                            const dist = asBool(currentVals["distributed_anchor"]);
                            const framePolicy = asStr(currentVals["existing_frames_policy"], "Continue (Auto-Resume)");
                            const maskingMode = asStr(currentVals["masking_method"], "None (No Masking)");

                            let visible = true;
                            if (w.name === "preview_interval" || w.name === "preview_camera_filter" || w.name === "eval_camera_index") {
                                visible = (viz === "Save Preview Images");
                            } else if (w.name === "motion_sensitivity") {
                                visible = (maskingMode !== "None (No Masking)");
                            } else if (w.name === "apply_smoothing") {
                                visible = topo.includes("Fixed");
                            } else if (w.name === "run_postprocess_if_no_training") {
                                visible = topo.includes("Fixed") && (framePolicy === "Continue (Auto-Resume)");
                            } else if (w.name === "distributed_anchor_path" || w.name === "distributed_anchor_frame" || w.name === "warmup_frames") {
                                visible = dist;
                            }

                            if (visible && savedIdx < saved.length) {
                                let val = saved[savedIdx++];
                                if (["visualize_training", "topology_mode", "masking_method", "existing_frames_policy"].includes(w.name)) {
                                    val = asStr(val, asStr(w.value));
                                } else if (w.name === "distributed_anchor") {
                                    val = asBool(val);
                                }
                                w.value = val;
                                currentVals[w.name] = val;
                                full.push(val);
                            } else {
                                full.push(w.value);
                                currentVals[w.name] = (w.name === "distributed_anchor") ? asBool(w.value) : w.value;
                            }
                        }

                        o.widgets_values = full;
                    }
                };

                // Callback to toggle visibility
                const updateVisibility = () => {
                    // 1. Preview Controls Visibility
                    const mode = asStr(vizWidget ? vizWidget.value : "Off", "Off");
                    const showPreviewControls = (mode === "Save Preview Images");

                    // 2. Smoothing Controls Visibility
                    const topo = asStr(topoWidget ? topoWidget.value : "");
                    const showSmoothing = topo.includes("Fixed");
                    const framePolicy = asStr(existingFramesPolicyWidget ? existingFramesPolicyWidget.value : "Continue (Auto-Resume)", "Continue (Auto-Resume)");
                    const showPostprocessNoTraining = showSmoothing && (framePolicy === "Continue (Auto-Resume)");

                    // 3. Distributed Controls Visibility
                    const showDistributed = asBool(createAnchorWidget ? createAnchorWidget.value : false);

                    // 4. Masking sensitivity visibility
                    const maskingMode = asStr(maskingMethodWidget ? maskingMethodWidget.value : "None (No Masking)", "None (No Masking)");
                    const showMotionSensitivity = (maskingMode !== "None (No Masking)");

                    // Filter widgets based on conditions
                    this.widgets = allWidgets.filter(w => {
                        if (w.name === "preview_interval") return showPreviewControls;
                        if (w.name === "preview_camera_filter") return showPreviewControls;
                        if (w.name === "eval_camera_index") return showPreviewControls;
                        if (w.name === "motion_sensitivity") return showMotionSensitivity;
                        if (w.name === "apply_smoothing") return showSmoothing;
                        if (w.name === "run_postprocess_if_no_training") return showPostprocessNoTraining;
                        if (w.name === "distributed_anchor_path") return showDistributed;
                        if (w.name === "distributed_anchor_frame") return showDistributed;
                        if (w.name === "warmup_frames") return showDistributed;
                        return true; // Show strictly required/other widgets
                    });

                    this.setSize([this.size[0], this.computeSize([this.size[0], this.size[1]])[1]]);

                    // Force refresh
                    if (app.canvas) app.canvas.setDirty(true, true);
                };

                if (vizWidget) {
                    vizWidget.callback = () => {
                        updateVisibility();
                    };
                }

                if (topoWidget) {
                    topoWidget.callback = () => {
                        updateVisibility();
                    };
                }

                if (existingFramesPolicyWidget) {
                    existingFramesPolicyWidget.callback = () => {
                        updateVisibility();
                    };
                }

                if (createAnchorWidget) {
                    createAnchorWidget.callback = () => {
                        updateVisibility();
                    };
                }

                if (maskingMethodWidget) {
                    maskingMethodWidget.callback = () => {
                        updateVisibility();
                    };
                }

                setTimeout(() => updateVisibility(), 100);

                return r;
            };

            // --- Live Preview Listener ---
            // We hook into onExecuted? No, we listen for socket event.
            // Since we can't augment the instance directly from here easily for global events,
            // we register the global listener once, or check inside the node?
            // "api.addEventListener" is global.

            // We'll manage the listener logic here.
            // Ideally should check if node exists.
        }
    },

    setup() {
        // Global listener
        api.addEventListener("freeflow_preview", (event) => {
            const data = event.detail;
            const nodeId = data.node;

            const node = app.graph.getNodeById(nodeId);
            if (!node) return;

            // Check if widget exists
            // We usually want an IMAGE widget.
            // If the node doesn't have one, we add it.

            let imgWidget = node.widgets ? node.widgets.find(w => w.name === "preview_image") : null;

            if (!imgWidget) {
                // Create image widget
                // ComfyUI doesn't expose a simple "Add Image Widget" API easily without internal calls
                // But we can create a custom widget object.
                imgWidget = {
                    name: "preview_image",
                    type: "image",
                    value: data, // {filename, subfolder, type}
                    draw: function (ctx, node, widget_width, y, widget_height) {
                        // Standard drawing logic is handled by 'app.js' if type is "image"?
                        // No, user widgets need implementation unless using builtin.
                        // BUT, we can use the "Preview Image" node's widget?
                        // Easier: Just update the node's "imgs" property if using standard output?
                        // But we didn't output an image to the graph connection.

                        // We can create a DOM widget (like we did for SmartGrid)?
                        // OR we can try to inject a standard widget.

                        // The standard "PreviewImage" node uses `widgets.IMAGE`.
                        // Let's rely on app.graphToDom handling if we set the value correctly?
                        // "image" type widgets are usually rendered by the DOM overlay if they have a value.
                    },
                    computeSize: () => [256, 256], // Placeholder
                };
                // node.addWidget ... wait, manual widget addition is messy.

                // BETTER: Use "app.nodeOutputs" to pretend we have an output?
                // No.

                // EASIEST: Just add a DOM element like SmartGrid.
                // We already have code for that in SmartGrid.
                // Let's create a visual DOM element.
            }

            // Actually, we can reuse the SmartGrid approach:
            // node.onDrawForeground...
            // But we need to load the image first.

            const img = new Image();
            img.src = api.apiURL(`/view?filename=${data.filename}&subfolder=${data.subfolder}&type=${data.type}`);
            img.onload = () => {
                node._freeflow_preview = [img]; // Use private property to avoid default Comfy rendering

                // Resize node to fit image
                const ratio = img.width / img.height;
                const w = node.size[0] - 20;
                const h = w / ratio;

                // Calculate height needed by widgets
                const sz = node.computeSize();
                const widgetsHeight = sz[1];

                const requiredHeight = widgetsHeight + h + 20;

                if (node.size[1] < requiredHeight) {
                    node.setSize([node.size[0], requiredHeight]);
                }

                app.graph.setDirtyCanvas(true, true);
            };

            // Custom draw to render the image
            if (!node._has_preview_draw) {
                const origDraw = node.onDrawForeground;
                node.onDrawForeground = function (ctx) {
                    if (origDraw) origDraw.apply(this, arguments);

                    if (this._freeflow_preview && this._freeflow_preview.length > 0) {
                        const img = this._freeflow_preview[0];

                        // Recalculate dimensions
                        const ratio = img.width / img.height;
                        const w = this.size[0] - 20;
                        const h = w / ratio;

                        // Find dynamic Y position below widgets
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
