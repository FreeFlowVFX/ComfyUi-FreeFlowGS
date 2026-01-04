import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { setWidgetConfig } from "../../extensions/core/widgetInputs.js";
import { applyTextReplacements } from "../../scripts/utils.js";

// --- Helper Functions ---

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existant object");
        return;
    }
    if (property in object && object[property]) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig.apply(this, arguments);
            return callback.apply(this, arguments) ?? r;
        };
    } else {
        object[property] = callback;
    }
}

function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

// --- Preview Logic (Adapted from VHS.core.js) ---

function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        var element = document.createElement("div");
        const previewNode = this;
        var previewWidget = this.addDOMWidget("vhs_preview", "preview", element, {
            serialize: false,
            hideOnZoom: false,
            getValue() {
                return element.value;
            },
            setValue(v) {
                element.value = v;
            },
        });
        previewWidget.computeSize = function (width) {
            if (this.aspectRatio && !this.parentEl.hidden) {
                let height = (previewNode.size[0] - 20) / this.aspectRatio + 10;
                if (!(height > 0)) {
                    height = 0;
                }
                this.computedHeight = height + 10;
                return [width, height];
            }
            return [width, -4];
        }
        // previewWidget.value = { hidden: false, paused: false, params: {} } // Initial state
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "vhs_preview";
        previewWidget.parentEl.style['width'] = "100%";
        element.appendChild(previewWidget.parentEl);

        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = true;
        previewWidget.videoEl.loop = true;
        previewWidget.videoEl.muted = true;
        previewWidget.videoEl.style['width'] = "100%";
        previewWidget.videoEl.onloadedmetadata = function () {
            previewWidget.aspectRatio = this.videoWidth / this.videoHeight;
            fitHeight(previewNode);
        };
        previewWidget.videoEl.hidden = true;

        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style['width'] = "100%";
        previewWidget.imgEl.hidden = true;
        previewWidget.imgEl.onload = () => {
            previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
            fitHeight(previewNode);
        };

        previewWidget.parentEl.appendChild(previewWidget.videoEl);
        previewWidget.parentEl.appendChild(previewWidget.imgEl);

        this.updateParameters = (params) => {
            if (!previewWidget.value) previewWidget.value = { hidden: false, paused: false };
            if (!previewWidget.value.params) previewWidget.value.params = {};

            Object.assign(previewWidget.value.params, params);

            // Update Source
            let urlParams = new URLSearchParams(params);
            let url = api.apiURL('/view?' + urlParams);

            previewWidget.parentEl.hidden = previewWidget.value.hidden;

            // Check format
            let format = params.format;
            if (format?.startsWith('video')) {
                previewWidget.videoEl.src = url;
                previewWidget.videoEl.hidden = false;
                previewWidget.imgEl.hidden = true;
                if (!previewWidget.value.paused) previewWidget.videoEl.play();
            } else if (format?.startsWith('image')) {
                previewWidget.imgEl.src = url;
                previewWidget.videoEl.hidden = true;
                previewWidget.imgEl.hidden = false;
            } else if (params.filename && (params.filename.endsWith(".mp4") || params.filename.endsWith(".gif"))) {
                // Fallback for filename detection
                if (params.filename.endsWith(".gif")) {
                    previewWidget.imgEl.src = url;
                    previewWidget.videoEl.hidden = true;
                    previewWidget.imgEl.hidden = false;
                } else {
                    previewWidget.videoEl.src = url;
                    previewWidget.videoEl.hidden = false;
                    previewWidget.imgEl.hidden = true;
                    if (!previewWidget.value.paused) previewWidget.videoEl.play();
                }
            }
        };
    });
}

// --- Format Widget Logic ---

function addFormatWidgets(nodeType, nodeData) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        var formatWidget = null;
        var formatWidgetIndex = -1;
        for (let i = 0; i < this.widgets.length; i++) {
            if (this.widgets[i].name === "format") {
                formatWidget = this.widgets[i];
                formatWidgetIndex = i + 1;
                break
            }
        }
        let formatWidgetsCount = 0;
        chainCallback(formatWidget, "callback", (value) => {
            const formats = (LiteGraph.registered_node_types[this.type]
                ?.nodeData?.input?.required?.format?.[1]?.formats)
            let newWidgets = [];
            if (formats?.[value]) {
                let formatWidgets = formats[value]
                for (let wDef of formatWidgets) {
                    let type = wDef[2]?.widgetType ?? wDef[1]
                    if (Array.isArray(type)) {
                        type = "COMBO"
                    }
                    if (app.widgets[type]) {
                        app.widgets[type](this, wDef[0], wDef.slice(1), app)
                        let w = this.widgets.pop()
                        w.config = wDef.slice(1)
                        newWidgets.push(w)
                    }
                }
            }
            let removed = this.widgets.splice(formatWidgetIndex,
                formatWidgetsCount, ...newWidgets);
            let newNames = new Set(newWidgets.map((w) => w.name))
            for (let w of removed) {
                w?.onRemove?.()
                if (w.name in newNames) {
                    continue
                }
                let slot = this.inputs.findIndex((i) => i.name == w.name)
                if (slot >= 0) {
                    this.removeInput(slot)
                }
            }
            for (let w of newWidgets) {
                let existingInput = this.inputs.find((i) => i.name == w.name)
                if (existingInput) {
                    setWidgetConfig(existingInput, w.config)
                } else {
                    this.addInput(w.name, w.config[0], { widget: { name: w.name } })
                }
            }
            fitHeight(this);
            formatWidgetsCount = newWidgets.length;
        });

        // FORCE EXECUTION ON LOAD to ensure widgets appear
        if (formatWidget && formatWidget.value) {
            formatWidget.callback(formatWidget.value);
        }
    });
}

function addVAEInputToggle(nodeType, nodeData) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        this.reject_ue_connection = (input) => input?.name == "vae"
    })
    chainCallback(nodeType.prototype, "onConnectionsChange", function (contype, slot, iscon, linf) {
        if (contype == LiteGraph.INPUT && slot == 3 && this.inputs[3].type == "VAE") {
            if (iscon && linf) {
                if (this.linkTimeout) {
                    clearTimeout(this.linkTimeout)
                    this.linkTimeout = false
                } else if (this.inputs[0].type == "IMAGE") {
                    this.linkTimeout = setTimeout(() => {
                        if (this.inputs[0].type != "IMAGE") {
                            return
                        }
                        this.linkTimeout = false;
                        this.disconnectInput(0);
                    }, 50)
                }
                this.inputs[0].type = 'LATENT';
            } else {
                if (this.inputs[0].type == "LATENT") {
                    this.linkTimeout = setTimeout(() => {
                        this.linkTimeout = false
                        this.disconnectInput(0);
                    }, 50)
                }
                this.inputs[0].type = "IMAGE";
            }
        }
    });
}

function addDateFormatting(nodeType, field) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const widget = this.widgets.find((w) => w.name === field);
        if (widget) {
            widget.serializeValue = () => {
                return applyTextReplacements(app, widget.value);
            };
        }
    });
}

// --- Main Extension Definition ---

app.registerExtension({
    name: "TimNodes.VideoCombineVHStim",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Strict check for OUR custom node
        if (nodeData.name === "VideoCombineVHStim" || nodeData.name === "Video Combine VHS tim") {

            console.log("TimNodes: Applying frontend logic to", nodeData.name);

            // 1. Dynamic Format Widgets
            addFormatWidgets(nodeType, nodeData);

            // 2. Date Formatting
            addDateFormatting(nodeType, "filename_prefix");

            // 3. VAE Toggle
            addVAEInputToggle(nodeType, nodeData);

            // 4. PREVIEW SUPPORT
            addVideoPreview(nodeType);

            // 5. Handle Execution Results
            chainCallback(nodeType.prototype, "onExecuted", function (message) {
                if (message?.gifs) {
                    this.updateParameters(message.gifs[0]);
                }
            });
        }
    }
});
