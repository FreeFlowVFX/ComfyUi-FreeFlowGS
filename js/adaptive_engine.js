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
                const filterWidget = findWidget("preview_camera_filter"); // Newly added
                const topoWidget = findWidget("topology_mode");
                const smoothWidget = findWidget("apply_smoothing");

                // Callback to toggle visibility
                const updateVisibility = () => {
                    // 1. Preview Controls Visibility
                    const mode = vizWidget.value;
                    const showPreviewControls = (mode === "Save Preview Images");

                    // 2. Smoothing Controls Visibility
                    // Show only if Fixed Topology
                    const topo = topoWidget ? topoWidget.value : "";
                    const showSmoothing = (topo && topo.includes("Fixed"));

                    // Helper to toggle a single widget
                    const toggleWidget = (widget, shouldShow) => {
                        if (!widget) return;
                        if (!shouldShow) {
                            if (widget.type !== "converted-widget") { // Prevent double-hiding logic issues
                                widget.lastType = widget.type;
                                widget.type = "converted-widget";
                                widget.computeSize = () => [0, -4];
                            }
                        } else {
                            if (widget.lastType) {
                                widget.type = widget.lastType;
                                widget.computeSize = undefined;
                                widget.lastType = undefined; // Reset
                            }
                        }
                    };

                    toggleWidget(intervalWidget, showPreviewControls);
                    toggleWidget(filterWidget, showPreviewControls);
                    toggleWidget(smoothWidget, showSmoothing);

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
