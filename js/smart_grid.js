// FreeFlow Smart Grid Monitor - Video Preview Extension
// Exact VHS pattern implementation
// This version uses DOM widgets instead of Canvas drawing to prevent freezing and layout issues.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("🌊 [FreeFlow] EXTENSION LOADED - VHS DOM MODE - " + Date.now());

// Helper function from VHS - chains callbacks safely
function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("[FreeFlow] chainCallback: object undefined");
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

// Helper to adjust node height
function fitHeight(node) {
    node.setSize([node.size[0], node.computeSize([node.size[0], node.size[1]])[1]]);
    node?.graph?.setDirtyCanvas(true);
}

// Add video preview widget - EXACT VHS PATTERN
function addVideoPreview(nodeType) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        console.log("🌊 [FreeFlow] Creating video preview widget (DOM mode)...");

        var element = document.createElement("div");
        const previewNode = this;

        var previewWidget = this.addDOMWidget("videopreview", "preview", element, {
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
            return [width, -4]; // no loaded src, widget should not display
        };

        // Event listeners for context menu passthrough
        element.addEventListener('contextmenu', (e) => {
            e.preventDefault();
            return app.canvas._mousedown_callback(e);
        }, true);
        element.addEventListener('pointerdown', (e) => {
            e.preventDefault();
            return app.canvas._mousedown_callback(e);
        }, true);

        // Read Global Settings (with fallback for older ComfyUI versions)
        let globalMute = true;
        let globalAutoplay = true;
        let globalLoop = true;
        try {
            if (app.ui?.settings?.getSettingValue) {
                globalMute = app.ui.settings.getSettingValue("FreeFlow.SmartGrid.MuteDefault", true);
                globalAutoplay = app.ui.settings.getSettingValue("FreeFlow.SmartGrid.Autoplay", true);
                globalLoop = app.ui.settings.getSettingValue("FreeFlow.SmartGrid.Loop", true);
            }
        } catch (e) {
            console.warn("🌊 [FreeFlow] Could not read settings, using defaults:", e);
        }

        previewWidget.value = { hidden: false, paused: !globalAutoplay, params: {}, muted: globalMute };
        previewWidget.parentEl = document.createElement("div");
        previewWidget.parentEl.className = "freeflow_preview";
        previewWidget.parentEl.style['width'] = "100%";
        element.appendChild(previewWidget.parentEl);

        previewWidget.videoEl = document.createElement("video");
        previewWidget.videoEl.controls = false;
        previewWidget.videoEl.loop = globalLoop;
        previewWidget.videoEl.muted = globalMute;
        previewWidget.videoEl.style['width'] = "100%";
        previewWidget.videoEl.addEventListener("loadedmetadata", () => {
            previewWidget.aspectRatio = previewWidget.videoEl.videoWidth / previewWidget.videoEl.videoHeight;
            fitHeight(this);
        });
        previewWidget.videoEl.addEventListener("error", (e) => {
            previewWidget.parentEl.hidden = true;
            fitHeight(this);
        });

        previewWidget.imgEl = document.createElement("img");
        previewWidget.imgEl.style['width'] = "100%";
        previewWidget.imgEl.hidden = true;
        previewWidget.imgEl.onload = () => {
            previewWidget.aspectRatio = previewWidget.imgEl.naturalWidth / previewWidget.imgEl.naturalHeight;
            fitHeight(this);
        };

        previewWidget.parentEl.appendChild(previewWidget.videoEl);
        previewWidget.parentEl.appendChild(previewWidget.imgEl);

        // updateParameters is called when execution completes
        this.updateParameters = (params, force_update) => {
            if (!previewWidget.value.params) {
                if (typeof (previewWidget.value) != 'object') {
                    previewWidget.value = { hidden: false, paused: false };
                }
                previewWidget.value.params = {};
            }
            Object.assign(previewWidget.value.params, params);
            if (force_update) {
                previewWidget.updateSource();
            }
        };

        previewWidget.updateSource = function () {
            if (this.value.params == undefined) {
                return;
            }
            let params = {};
            Object.assign(params, this.value.params); // shallow copy

            this.parentEl.hidden = this.value.hidden;

            // Check format
            if (params.format?.split('/')[0] == 'video' || params.format == 'folder') {
                this.videoEl.autoplay = !this.value.paused && !this.value.hidden;
                let finalUrl = api.apiURL('/view?' + new URLSearchParams(params));
                console.log("🌊 [FreeFlow] Setting video src:", finalUrl);
                this.videoEl.src = finalUrl;
                this.videoEl.hidden = false;
                this.imgEl.hidden = true;
            } else if (params.format?.split('/')[0] == 'image') {
                // Is animated image
                this.imgEl.src = api.apiURL('/view?' + new URLSearchParams(params));
                this.videoEl.hidden = true;
                this.imgEl.hidden = false;
            }
        };

        previewWidget.callback = previewWidget.updateSource;
    });
}

// Add preview menu options - EXACT VHS PATTERN
function addPreviewOptions(nodeType) {
    chainCallback(nodeType.prototype, "getExtraMenuOptions", function (_, options) {
        let optNew = [];
        const previewWidget = this.widgets.find((w) => w.name === "videopreview");

        if (!previewWidget) {
            return;
        }

        // Separator before our section if options exist
        if (options.length > 0) {
            // options.push(null); // No, we're unshifting. Separator at end of OUR list.
        }

        // --- GROUP: PREVIEW ---
        optNew.push({
            content: "🌊 FreeFlow Preview",
            disabled: true
        });

        // 1. Show/Hide
        const visDesc = (previewWidget.value?.hidden ? "👁 Show Preview" : "🚫 Hide Preview");
        optNew.push({
            content: visDesc,
            callback: () => {
                if (!previewWidget.videoEl.hidden && !previewWidget.value.hidden) {
                    previewWidget.videoEl.pause();
                } else if (previewWidget.value.hidden && !previewWidget.videoEl.hidden && !previewWidget.value.paused) {
                    previewWidget.videoEl.play();
                }
                previewWidget.value.hidden = !previewWidget.value.hidden;
                previewWidget.parentEl.hidden = previewWidget.value.hidden;
                fitHeight(this);
            }
        });

        // 2. Play/Pause
        const PauseDesc = (previewWidget.value?.paused ? "▶ Resume" : "⏸ Pause");
        if (previewWidget.videoEl?.hidden == false) {
            optNew.push({
                content: PauseDesc,
                callback: () => {
                    if (previewWidget.value.paused) {
                        previewWidget.videoEl?.play();
                    } else {
                        previewWidget.videoEl?.pause();
                    }
                    previewWidget.value.paused = !previewWidget.value.paused;
                }
            });
        }

        // 3. Mute/Unmute
        const muteDesc = (previewWidget.value?.muted ? "🔊 Unmute" : "🔇 Mute");
        optNew.push({
            content: muteDesc,
            callback: () => {
                previewWidget.value.muted = !previewWidget.value.muted;
                previewWidget.videoEl.muted = previewWidget.value.muted;
            }
        });

        // 4. Sync
        optNew.push({
            content: "🔄 Sync/Reload",
            callback: () => {
                if (previewWidget.videoEl) {
                    previewWidget.videoEl.currentTime = 0;
                }
            }
        });

        // Separator
        optNew.push(null);

        // --- GROUP: FILE ACTIONS ---
        let url = null;
        if (previewWidget.videoEl?.hidden == false && previewWidget.videoEl.src) {
            if (['input', 'output', 'temp'].includes(previewWidget.value?.params?.type)) {
                url = api.apiURL('/view?' + new URLSearchParams(previewWidget.value.params));
            }
        } else if (previewWidget.imgEl?.hidden == false && previewWidget.imgEl.src) {
            url = previewWidget.imgEl.src;
        }

        if (url) {
            optNew.push({
                content: "↗ Open in New Tab",
                callback: () => {
                    window.open(url, "_blank");
                },
            });
            optNew.push({
                content: "💾 Save Video",
                callback: () => {
                    const a = document.createElement("a");
                    a.href = url;
                    a.setAttribute("download", previewWidget.value?.params?.filename || "freeflow_preview");
                    document.body.append(a);
                    a.click();
                    requestAnimationFrame(() => a.remove());
                },
            });
            // Separator after file actions
            optNew.push(null);
        }

        // Add all our options to the TOP of the menu
        options.unshift(...optNew);
    });
}

// Main extension registration
// Main extension registration
app.registerExtension({
    name: "FreeFlow.SmartGridMonitor",

    settings: [
        {
            id: "FreeFlow.SmartGrid.MuteDefault",
            name: "🌊 FreeFlow: Mute previews by default",
            type: "boolean",
            defaultValue: true,
            tooltip: "If enabled, video previews will start muted."
        },
        {
            id: "FreeFlow.SmartGrid.Autoplay",
            name: "🌊 FreeFlow: Autoplay previews",
            type: "boolean",
            defaultValue: true,
            tooltip: "If enabled, video previews will start playing automatically."
        },
        {
            id: "FreeFlow.SmartGrid.Loop",
            name: "🌊 FreeFlow: Loop previews",
            type: "boolean",
            defaultValue: true,
            tooltip: "If enabled, video previews will loop."
        }
    ],

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData?.name == "FreeFlow_SmartGridMonitor") {
            console.log("🌊 [FreeFlow] Setting up SmartGridMonitor hooks...");

            // Hook onExecuted to receive data from Python
            chainCallback(nodeType.prototype, "onExecuted", function (message) {
                console.log("🌊 [FreeFlow] onExecuted triggered!", message);
                if (message?.gifs) {
                    console.log("🌊 [FreeFlow] Found gifs in output:", message.gifs);
                    this.updateParameters(message.gifs[0], true);
                } else if (message?.video) {
                    console.log("🌊 [FreeFlow] Found video in output (fallback):", message.video);
                    this.updateParameters(message.video[0], true);
                } else {
                    console.log("🌊 [FreeFlow] No gifs/video found in message:", Object.keys(message));
                }
            });

            // --- UI Visibility Logic for Labels ---
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

                const showLabelsWidget = this.widgets.find((w) => w.name === "show_labels");
                const labelSizeWidget = this.widgets.find((w) => w.name === "label_size");
                const labelPosWidget = this.widgets.find((w) => w.name === "label_position");

                const updateVisibility = () => {
                    const visible = showLabelsWidget.value;

                    const toggle = (w, shouldShow) => {
                        if (!w) return;
                        if (!shouldShow) {
                            w.lastType = w.type;
                            w.type = "converted-widget";
                            w.computeSize = () => [0, -4];
                        } else {
                            if (w.lastType) {
                                w.type = w.lastType;
                                w.computeSize = undefined;
                            }
                        }
                    };

                    toggle(labelSizeWidget, visible);
                    toggle(labelPosWidget, visible);

                    this.setSize([this.size[0], this.computeSize([this.size[0], this.size[1]])[1]]);
                    app.canvas.setDirty(true, true);
                };

                if (showLabelsWidget) {
                    showLabelsWidget.callback = updateVisibility;
                    setTimeout(updateVisibility, 100); // Initialize state
                }

                return r;
            };

            addVideoPreview(nodeType);
            addPreviewOptions(nodeType);
        }
    }
});
