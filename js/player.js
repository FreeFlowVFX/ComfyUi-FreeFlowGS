
import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "FreeFlow.InteractivePlayer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FreeFlow_InteractivePlayer") {

            // Add custom widget using onNodeCreated
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (onNodeCreated) onNodeCreated.apply(this, arguments);

                // Create Player Widget
                const w = {
                    name: "4D Player",
                    type: "DIV",
                    value: null,
                    draw: function (ctx, node, widgetWidth, y, widgetHeight) {
                        // Custom drawing managed by Player Class instance attached to node
                        if (node.playerInstance) {
                            node.playerInstance.draw(ctx, node, widgetWidth, y, widgetHeight);
                        }
                    },
                    computeSize: function (width) {
                        return [width, 400]; // Height
                    }
                };

                this.addCustomWidget(w);
                this.playerInstance = new FreeFlow4DPlayer(this);
            };

            // Handle Server Response
            nodeType.prototype.onExecuted = function (message) {
                if (message && message.player_data) {
                    const data = message.player_data[0]; // {output_dir, files, fps}
                    if (this.playerInstance) {
                        this.playerInstance.loadSequence(data);
                    }
                }
            }
        }
    }
});

class FreeFlow4DPlayer {
    constructor(node) {
        this.node = node;
        this.element = null;
        this.scene = null;
        this.camera = null;
        this.renderer = null;
        this.controls = null;

        this.sequenceData = null;
        this.frames = []; // Array of SplatMeshes (or Buffers)
        this.currentFrame = 0;
        this.renderFrameIndex = 0; // Async display frame (wait for sort)
        this.isPlaying = false;
        this.fps = 24;
        this.lastFrameTime = 0;

        this.initElement();
        this.initWorker();
    }

    initWorker() {
        // Initialize sorting worker
        this.worker = new Worker(new URL('./splat_worker.js', import.meta.url), { type: "module" });
        this.worker.onmessage = (e) => {
            this.handleWorkerMessage(e);
        };
        this.sortReady = true;
        this.lastSortTime = 0;
    }

    handleWorkerMessage(e) {
        try {
            // New Protocol: Worker sends FULL buffers, already sorted.
            const { sortedPositions, sortedColors, sortedScales, sortedRots, meshId } = e.data;

            if (!sortedPositions) return;

            // Find the correct mesh by ID
            let mesh = null;
            if (meshId) {
                mesh = this.frames.find(m => m && m.uuid === meshId);
            } else {
                mesh = this.frames[this.currentFrame];
            }

            if (!mesh || !mesh.geometry) return;

            const geometry = mesh.geometry;

            // Update Attributes (Directly from Worker)
            // Zero-Copy transfer complete.
            geometry.attributes.splatCenter.array = new Float32Array(sortedPositions);
            geometry.attributes.splatColor.array = new Float32Array(sortedColors);
            geometry.attributes.splatScale.array = new Float32Array(sortedScales);

            // Handle naming variance (splatRot vs splatRotation)
            if (geometry.attributes.splatRotation) {
                geometry.attributes.splatRotation.array = new Float32Array(sortedRots);
                geometry.attributes.splatRotation.needsUpdate = true;
            } else if (geometry.attributes.splatRot) {
                geometry.attributes.splatRot.array = new Float32Array(sortedRots);
                geometry.attributes.splatRot.needsUpdate = true;
            }

            geometry.attributes.splatCenter.needsUpdate = true;
            geometry.attributes.splatColor.needsUpdate = true;
            geometry.attributes.splatScale.needsUpdate = true;

            // Async Swap: If we just sorted the frame we WANT to see, show it now.
            // This prevents showing the frame before it is sorted (removing flicker).
            if (this.currentFrame >= 0 && this.frames[this.currentFrame] && this.frames[this.currentFrame].uuid === meshId) {
                this.renderFrameIndex = this.currentFrame;
            }

        } catch (err) {
            console.error("[Player] Error handling worker message:", err);
        } finally {
            // CRITICAL FAILSAFE: Always unlock the sorter, or player freezes forever.
            this.sortReady = true;
        }
    }

    async initElement() {
        // Container
        this.element = document.createElement("div");
        this.element.className = "freeflow-player";
        Object.assign(this.element.style, {
            position: "fixed",
            zIndex: "100", // Lowered to allow ComfyUI menus on top (was 1005)
            background: "#000",
            display: "none",
            border: "1px solid #666",
            overflow: "hidden"
        });
        document.body.appendChild(this.element);

        // Load Three.js (Reuse from Viz or CDN fallback)
        if (!window.THREE) await import('./three.module.js').then(m => window.THREE = m);
        if (!window.OrbitControls) await import('./OrbitControls.js').then(m => window.OrbitControls = m.OrbitControls);

        const THREE = window.THREE;

        // Setup 3D Scene
        this.scene = new THREE.Scene();
        this.scene.background = new THREE.Color(0x000000);

        this.camera = new THREE.PerspectiveCamera(60, 1, 0.1, 5000); // Increased Far Clip
        this.camera.position.set(0, 0, 5);

        this.renderer = new THREE.WebGLRenderer({ antialias: false }); // MSAA off for splats usually better
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.element.appendChild(this.renderer.domElement);

        this.controls = new window.OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.autoRotate = false;

        // ... (UI setup skipped for brevity in replace, assume unchanged if not matched) ...
        // Add UI Overlay (Timeline)
        this.ui = document.createElement("div");
        Object.assign(this.ui.style, {
            position: "absolute", bottom: "0", left: "0", width: "100%",
            background: "linear-gradient(to top, rgba(10,10,10,0.98), rgba(20,20,20,0.8))",
            padding: "12px 20px", display: "flex", gap: "18px",
            color: "#fff", fontFamily: "'Inter', system-ui, sans-serif", fontSize: "12px", alignItems: "center",
            backdropFilter: "blur(25px)",
            boxShadow: "0 -5px 30px rgba(0,0,0,0.8)"
        });

        // --- CONTROL STYLES ---
        const btnStyle = {
            padding: "4px 12px",
            background: "rgba(255,255,255,0.03)",
            border: "none",
            borderRadius: "4px",
            color: "#fff",
            cursor: "pointer",
            fontSize: "12px",
            fontWeight: "600",
            transition: "all 0.15s ease",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            minWidth: "34px",
            height: "32px",
            textTransform: "uppercase",
            letterSpacing: "0.5px",
            outline: "none"
        };

        const applyHover = (btn) => {
            btn.onmouseenter = () => {
                btn.style.background = "rgba(255,136,0,0.1)";
                btn.style.color = "#ff8800";
            };
            btn.onmouseleave = () => {
                btn.style.background = "rgba(255,255,255,0.03)";
                btn.style.color = (btn === this.focusBtn || btn === this.gearBtn || btn === this.infoBtn) ? "#ff8800" : "#fff";
            };
        };

        // --- TRANSPORT CONTROLS ---
        this.prevBtn = document.createElement("button");
        this.prevBtn.innerText = "❮";
        this.prevBtn.onclick = () => this.stepFrame(-1);
        Object.assign(this.prevBtn.style, btnStyle);
        applyHover(this.prevBtn);

        this.playBtn = document.createElement("button");
        this.playBtn.innerText = "▶";
        this.playBtn.onclick = () => this.togglePlay();
        Object.assign(this.playBtn.style, btnStyle);
        applyHover(this.playBtn);

        this.nextBtn = document.createElement("button");
        this.nextBtn.innerText = "❯";
        this.nextBtn.onclick = () => this.stepFrame(1);
        Object.assign(this.nextBtn.style, btnStyle);
        applyHover(this.nextBtn);

        // Add Active State
        const addActiveLike = (btn) => {
            btn.addEventListener("mousedown", () => btn.style.transform = "scale(0.95)");
            btn.addEventListener("mouseup", () => btn.style.transform = "scale(1.0)");
            btn.addEventListener("mouseleave", () => btn.style.transform = "scale(1.0)");
        };
        addActiveLike(this.prevBtn);
        addActiveLike(this.playBtn);
        addActiveLike(this.nextBtn);

        // Focus Button
        this.focusBtn = document.createElement("button");
        this.focusBtn.innerHTML = "<span style='color:#ff8800; margin-right:4px; font-size:14px;'>⌖</span> <span style='color:#fff;'>FOCUS</span>";
        this.focusBtn.onclick = () => this.centerCamera();
        Object.assign(this.focusBtn.style, btnStyle);
        applyHover(this.focusBtn);

        // Gear (Settings) Button
        this.settingsVisible = false;
        this.gearBtn = document.createElement("button");
        this.gearBtn.innerHTML = "⚙";
        this.gearBtn.onclick = () => this.toggleSettings();
        Object.assign(this.gearBtn.style, btnStyle);
        this.gearBtn.style.color = "#ff8800";
        this.gearBtn.style.fontSize = "16px";
        applyHover(this.gearBtn);

        this.scrubber = document.createElement("input");
        this.scrubber.type = "range";
        this.scrubber.min = 0;
        this.scrubber.value = 0;
        this.scrubber.style.flex = "1";
        this.scrubber.oninput = (e) => {
            const idx = parseInt(e.target.value);
            this.seek(idx);
            const real = this.getRealFrameNumber(idx);
            const count = this.sequenceData?.files?.length || 0;
            this.frameLabel.innerText = `${real} (${count})`;
        };

        this.frameLabel = document.createElement("span");
        this.frameLabel.innerText = "0 / 0";
        Object.assign(this.frameLabel.style, {
            minWidth: "70px",
            fontFamily: "'JetBrains Mono', monospace",
            textAlign: "center",
            color: "#ff8800",
            fontWeight: "700",
            fontSize: "11px",
            background: "rgba(0,0,0,0.3)",
            padding: "5px 10px",
            borderRadius: "4px",
            letterSpacing: "0.5px"
        });

        this.renderMode = "Splats"; // "Splats", "Points"
        this.renderModeSelect = document.createElement("select");
        ["Splats", "Points"].forEach(m => {
            const opt = document.createElement("option");
            opt.value = m;
            opt.innerText = m;
            this.renderModeSelect.appendChild(opt);
        });
        this.renderModeSelect.onchange = (e) => {
            this.renderMode = e.target.value;
            this.updateMaterialMode();
            if (this.showInfo && this.updateOverlayInfo) this.updateOverlayInfo();
        };
        Object.assign(this.renderModeSelect.style, {
            background: "rgba(40,40,40,0.95)",
            border: "1px solid rgba(255,136,0,0.2)",
            borderRadius: "4px",
            color: "#fff",
            fontSize: "11px",
            padding: "5px 24px 5px 10px", // Extra right padding for arrow
            outline: "none",
            cursor: "pointer",
            fontFamily: "inherit",
            appearance: "none", // Remove default OS arrow
            backgroundImage: "url(\"data:image/svg+xml;charset=UTF-8,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23ff8800' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3e%3cpolyline points='6 9 12 15 18 9'%3e%3c/polyline%3e%3c/svg%3e\")",
            backgroundRepeat: "no-repeat",
            backgroundPosition: "right 6px center",
            backgroundSize: "12px"
        });

        // Scale Control Group
        const scaleGroup = document.createElement("div");
        Object.assign(scaleGroup.style, {
            display: "flex", alignItems: "center", gap: "10px",
            background: "rgba(0,0,0,0.3)", padding: "2px 12px", borderRadius: "6px",
            height: "36px"
        });

        const scaleLbl = document.createElement("span");
        scaleLbl.innerText = "SCALE";
        scaleLbl.style.fontSize = "9px";
        scaleLbl.style.color = "rgba(255,255,255,0.5)";
        scaleLbl.style.letterSpacing = "1.5px";
        scaleLbl.style.fontWeight = "800";

        // Number Input
        this.scaleInput = document.createElement("input");
        this.scaleInput.type = "number";
        this.scaleInput.min = "0.1";
        this.scaleInput.max = "10.0";
        this.scaleInput.step = "0.1";
        this.scaleInput.value = "1.0";
        Object.assign(this.scaleInput.style, {
            width: "42px", background: "transparent", border: "none",
            color: "#ff8800", fontSize: "12px", fontFamily: "'JetBrains Mono', monospace",
            textAlign: "right", padding: "0", outline: "none", fontWeight: "bold"
        });

        // Slider
        this.scaleSlider = document.createElement("input");
        this.scaleSlider.type = "range";
        this.scaleSlider.min = 0.1;
        this.scaleSlider.max = 10.0;
        this.scaleSlider.step = 0.1;
        this.scaleSlider.value = 1.0;
        this.scaleSlider.style.width = "60px";
        this.scaleSlider.style.cursor = "pointer";

        // Sync Logic
        const updateScale = (val, fromInput) => {
            let v = parseFloat(val);
            if (isNaN(v)) return;
            v = Math.max(0.1, Math.min(10.0, v));

            if (fromInput) {
                this.scaleSlider.value = v;
            } else {
                this.scaleInput.value = v.toFixed(1);
            }

            this.frames.forEach(m => {
                if (m && m.material) m.material.uniforms.scaleModifier.value = v;
            });
        };

        this.scaleInput.onchange = (e) => updateScale(e.target.value, true);
        this.scaleSlider.oninput = (e) => updateScale(e.target.value, false);

        scaleGroup.appendChild(scaleLbl);
        scaleGroup.appendChild(this.scaleInput);
        scaleGroup.appendChild(this.scaleSlider);


        // Info Toggle
        this.showInfo = false;
        this.infoBtn = document.createElement("button");
        // Custom Orange Info Icon (SVG)
        this.infoBtn.innerHTML = `
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="10"></circle>
            <line x1="12" y1="16" x2="12" y2="12"></line>
            <line x1="12" y1="8" x2="12.01" y2="8"></line>
        </svg>
        `;
        this.infoBtn.style.color = "#ff8800"; // Default Orange
        this.infoBtn.onclick = () => {
            this.showInfo = !this.showInfo;
            if (this.infoPanel) {
                this.infoPanel.style.display = this.showInfo ? "block" : "none";
                if (this.showInfo && this.updateOverlayInfo) this.updateOverlayInfo();
            }
        };
        Object.assign(this.infoBtn.style, btnStyle);
        this.infoBtn.style.padding = "4px"; // Adjust padding for icon
        applyHover(this.infoBtn);

        // Append Bottom Bar Elements
        // Append Bottom Bar Elements
        this.ui.appendChild(this.prevBtn);
        this.ui.appendChild(this.playBtn); // Play in Middle
        this.ui.appendChild(this.nextBtn);
        this.ui.appendChild(this.focusBtn);
        this.ui.appendChild(this.scrubber);
        this.ui.appendChild(this.frameLabel);
        this.ui.appendChild(this.renderModeSelect);
        this.ui.appendChild(this.gearBtn);
        this.ui.appendChild(this.infoBtn);

        this.element.appendChild(this.ui);

        // --- SETTINGS OVERLAY ---
        this.createSettingsOverlay(scaleGroup); // Pass scaleGroup to move it there

        // Keyboard Shortcuts
        window.addEventListener("keydown", (e) => {
            if (e.key === " " || e.code === "Space") {
                this.togglePlay();
                e.preventDefault();
            } else if (e.key === "ArrowLeft") {
                this.stepFrame(-1);
                e.preventDefault();
            } else if (e.key === "ArrowRight") {
                this.stepFrame(1);
                e.preventDefault();
            }
        });

        // Add Custom Range CSS
        const style = document.createElement("style");
        style.textContent = `
            .freeflow-player input[type=range] {
                -webkit-appearance: none;
                background: transparent;
            }
            .freeflow-player input[type=range]::-webkit-slider-runnable-track {
                width: 100%;
                height: 4px;
                cursor: pointer;
                background: rgba(255,255,255,0.1);
                border-radius: 2px;
            }
            .freeflow-player input[type=range]::-webkit-slider-thumb {
                height: 16px;
                width: 16px;
                border-radius: 50%;
                background: #ff8800;
                cursor: pointer;
                -webkit-appearance: none;
                margin-top: -6px;
                box-shadow: 0 0 10px rgba(255,136,0,0.3);
                transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                border: none;
            }
            .freeflow-player input[type=range]::-webkit-slider-thumb:hover {
                transform: scale(1.3);
                background: #fff;
            }
            .freeflow-player select:focus {
                outline: none;
                border: 1px solid #ff8800 !important;
            }
        `;
        document.head.appendChild(style);

        // Info Panel Redesign
        this.infoPanel = document.createElement("div");
        Object.assign(this.infoPanel.style, {
            position: "absolute", top: "0", left: "0", padding: "20px",
            background: "rgba(10,10,10,0.92)", color: "#fff", fontFamily: "'JetBrains Mono', monospace",
            fontSize: "11px", display: "none", whiteSpace: "pre-wrap",
            maxHeight: "calc(100% - 60px)", overflowY: "auto",
            backdropFilter: "blur(25px)",
            lineHeight: "1.6", boxShadow: "5px 0 25px rgba(0,0,0,0.8)"
        });
        this.element.appendChild(this.infoPanel);

        if (this.createOverlay) this.createOverlay();
        if (this.initInteraction) this.initInteraction();

        this.animate();
    }

    createSettingsOverlay(scaleGroup) {
        this.settingsPanel = document.createElement("div");
        Object.assign(this.settingsPanel.style, {
            position: "absolute", bottom: "80px", right: "20px", width: "260px",
            background: "rgba(20, 20, 20, 0.9)", backdropFilter: "blur(15px)",
            borderRadius: "12px", padding: "16px",
            display: "none", flexDirection: "column", gap: "16px",
            border: "1px solid rgba(255,136,0,0.2)",
            boxShadow: "0 10px 40px rgba(0,0,0,0.5)",
            zIndex: "2000",
            color: "#ccc", fontSize: "12px", fontFamily: "'Inter', sans-serif"
        });

        // Header
        const header = document.createElement("div");
        header.innerText = "VIEWER SETTINGS";
        header.style.color = "#ff8800";
        header.style.fontWeight = "800";
        header.style.letterSpacing = "1px";
        header.style.fontSize = "10px";
        header.style.marginBottom = "4px";
        this.settingsPanel.appendChild(header);

        // Generic Control Helper
        const createControl = (label, min, max, step, def, callback) => {
            const container = document.createElement("div");
            container.style.display = "flex";
            container.style.justifyContent = "space-between";
            container.style.alignItems = "center";
            container.innerHTML = `<span style='width:60px; cursor:pointer' title='Double-click to reset'>${label}</span>`;

            // Text Label Reset
            container.querySelector("span").ondblclick = () => {
                resetValue();
            };

            const slider = document.createElement("input");
            slider.type = "range";
            slider.min = min; slider.max = max; slider.step = step; slider.value = def;
            slider.style.flex = "1";
            slider.style.margin = "0 8px";

            // Numeric Input
            const numInput = document.createElement("input");
            numInput.type = "number";
            numInput.min = min; numInput.max = max; numInput.step = step; numInput.value = def;
            numInput.style.width = "40px";
            numInput.style.background = "transparent";
            numInput.style.border = "none";
            numInput.style.color = "#ff8800";
            numInput.style.textAlign = "right";
            numInput.style.fontSize = "11px";
            numInput.style.fontFamily = "'JetBrains Mono', monospace";
            numInput.style.outline = "none";

            const update = (val) => {
                const v = parseFloat(val);
                slider.value = v;
                numInput.value = v;
                callback(v);
            };

            const resetValue = () => {
                update(def);
            };

            slider.oninput = (e) => update(e.target.value);
            slider.ondblclick = resetValue;

            numInput.onchange = (e) => update(e.target.value);

            container.appendChild(slider);
            container.appendChild(numInput);
            this.settingsPanel.appendChild(container);

            return { update };
        };

        // 1. Scale
        createControl("Scale", 0.1, 10.0, 0.1, 1.0, (v) => {
            this.frames.forEach(m => {
                if (m && m.material) m.material.uniforms.scaleModifier.value = v;
            });
        });

        // 2. FOV
        createControl("FOV", 10, 120, 1, 75, (v) => {
            this.camera.fov = v;
            this.camera.updateProjectionMatrix();
        });

        // 3. Grid Toggle (Custom Checkbox UI)
        const gridSection = document.createElement("div");
        gridSection.style.display = "flex";
        gridSection.style.justifyContent = "space-between";
        gridSection.style.alignItems = "center";
        gridSection.style.marginTop = "4px";
        gridSection.innerHTML = "<span style='font-weight:600;'>Show Grid</span>";

        const gridSwitch = document.createElement("input");
        gridSwitch.type = "checkbox";
        gridSwitch.style.cursor = "pointer";
        gridSwitch.onchange = (e) => {
            if (this.gridHelper) this.gridHelper.visible = e.target.checked;
            else if (e.target.checked) this.createGrid();
        };
        gridSection.appendChild(gridSwitch);
        this.settingsPanel.appendChild(gridSection);

        // 4. Playback Controls (FPS)
        const playbackTitle = document.createElement("div");
        playbackTitle.innerText = "PLAYBACK & RENDER";
        playbackTitle.style.marginTop = "12px";
        playbackTitle.style.marginBottom = "4px";
        playbackTitle.style.fontSize = "9px";
        playbackTitle.style.opacity = "0.5";
        playbackTitle.style.letterSpacing = "1px";
        playbackTitle.style.fontWeight = "800";
        this.settingsPanel.appendChild(playbackTitle);

        // Depth Sorting Toggle
        const sortSection = document.createElement("div");
        sortSection.style.display = "flex";
        sortSection.style.justifyContent = "space-between";
        sortSection.style.alignItems = "center";
        sortSection.style.marginBottom = "8px";

        // Toggle Tooltip
        sortSection.title = "Controls transparency sorting. Disable for faster playback.";
        sortSection.innerHTML = "<span style='font-weight:600;'>Depth Sorting</span>";

        const sortSwitch = document.createElement("input");
        sortSwitch.type = "checkbox";
        sortSwitch.checked = this.depthSorting;
        sortSwitch.style.cursor = "pointer";
        sortSwitch.onchange = (e) => {
            this.depthSorting = e.target.checked;
        };
        sortSection.appendChild(sortSwitch);
        this.settingsPanel.appendChild(sortSection);

        // FPS Slider/Input
        const fpsControl = createControl("FPS", 1, 120, 1, this.fps || 24, (v) => {
            this.fps = v;
        });

        // FPS Presets
        const presetContainer = document.createElement("div");
        presetContainer.style.display = "flex";
        presetContainer.style.justifyContent = "space-between";
        presetContainer.style.marginTop = "4px";

        const presets = [24, 25, 30, 50, 60];
        presets.forEach(p => {
            const btn = document.createElement("div");
            btn.innerText = p;
            Object.assign(btn.style, {
                background: "rgba(255,255,255,0.05)",
                borderRadius: "4px",
                padding: "2px 6px",
                cursor: "pointer",
                fontSize: "10px",
                color: "#ccc",
                textAlign: "center",
                flex: "1",
                margin: "0 2px"
            });
            btn.onmouseover = () => btn.style.background = "rgba(255,136,0,0.2)";
            btn.onmouseout = () => btn.style.background = "rgba(255,255,255,0.05)";
            btn.onclick = () => {
                fpsControl.update(p);
            };
            presetContainer.appendChild(btn);
        });
        this.settingsPanel.appendChild(presetContainer);

        // 5. Color Grading Controls
        const colorTitle = document.createElement("div");
        colorTitle.innerText = "COLOR GRADING";
        colorTitle.style.marginTop = "12px";
        colorTitle.style.marginBottom = "4px";
        colorTitle.style.fontSize = "9px";
        colorTitle.style.opacity = "0.5";
        colorTitle.style.letterSpacing = "1px";
        colorTitle.style.fontWeight = "800";
        this.settingsPanel.appendChild(colorTitle);

        createControl("Temp", -1.0, 1.0, 0.05, 0.0, (v) => this.updateColorUniforms('uTemperature', v));
        createControl("Sat", 0.0, 3.0, 0.1, 1.0, (v) => this.updateColorUniforms('uSaturation', v));
        createControl("Bright", -1.0, 1.0, 0.05, 0.0, (v) => this.updateColorUniforms('uBrightness', v));
        createControl("Black", 0.0, 0.5, 0.01, 0.0, (v) => this.updateColorUniforms('uBlackPoint', v));
        createControl("White", 0.5, 2.0, 0.01, 1.0, (v) => this.updateColorUniforms('uWhitePoint', v));
        createControl("Trans", 0.0, 1.0, 0.01, 0.0, (v) => this.updateColorUniforms('uTransparency', v));

        this.element.appendChild(this.settingsPanel);
    }

    createGrid() {
        this.gridHelper = new window.THREE.GridHelper(20, 20, 0x444444, 0x222222);
        this.scene.add(this.gridHelper);
    }

    toggleSettings() {
        this.settingsVisible = !this.settingsVisible;
        this.settingsPanel.style.display = this.settingsVisible ? "flex" : "none";
        this.gearBtn.style.color = this.settingsVisible ? "#fff" : "#ff8800";
        this.gearBtn.style.background = this.settingsVisible ? "#ff8800" : "rgba(255,255,255,0.03)";
    }

    stepFrame(dir) {
        if (!this.frames.length) return;
        let newIdx = this.currentFrame + dir;
        if (newIdx < 0) newIdx = this.frames.length - 1;
        if (newIdx >= this.frames.length) newIdx = 0;

        this.stop(); // Stop auto-play if stepping manual
        this.seek(newIdx);
    }


    updateColorUniforms(id, value) {
        // Store globally for new meshes
        if (!this.colorState) this.colorState = {};
        this.colorState[id] = value;

        // Update existing
        this.frames.forEach(mesh => {
            if (mesh && mesh.material && mesh.material.uniforms[id]) {
                mesh.material.uniforms[id].value = value;
            }
        });
    }

    updateMaterialMode() {
        this.frames.forEach(mesh => {
            if (mesh) {
                mesh.material.uniforms.renderMode.value = (this.renderMode === "Splats") ? 0 : 1;
                mesh.material.needsUpdate = true;
            }
        });
    }

    // ... (rest of class) ...

    cleanupSequence() {
        if (!this.frames) return;
        this.frames.forEach(mesh => {
            if (mesh) {
                this.scene.remove(mesh);
                if (mesh.geometry) mesh.geometry.dispose();
                if (mesh.material) mesh.material.dispose();
            }
        });
        this.frames = [];
        this.cachedMeshId = null; // Reset sort cache
    }

    async loadSequence(data) {
        // Cleanup existing sequence to prevent "Ghosting"
        this.cleanupSequence();

        this.sequenceData = data;
        this.frames = new Array(data.files.length).fill(null);
        this.fps = data.fps || 24;

        // update camera far clip if provided
        if (data.camera_far) {
            this.camera.far = data.camera_far;
            this.camera.updateProjectionMatrix();
        }

        // Depth Sorting Toggle
        this.depthSorting = (data.depth_sorting !== undefined) ? data.depth_sorting : true;

        this.scrubber.max = data.files.length - 1;

        // Update Label with Real Frame Number + Total Count
        const currentReal = this.getRealFrameNumber(0);
        this.frameLabel.innerText = `${currentReal} (${data.files.length})`;

        // Pre-load first frame
        if (data.files.length > 0) {
            await this.loadFrame(0);
            this.updateFrameVisibility();
        }
    }

    async loadFrame(index) {
        if (this.frames[index]) return this.frames[index]; // Already loaded

        const filename = this.sequenceData.files[index];
        const fullPath = `${this.sequenceData.output_dir}/${filename}`;

        // Use Custom Route: /freeflow/view?filename=ABS_PATH
        const url = `/freeflow/view?filename=${encodeURIComponent(fullPath)}`;

        if (this.infoPanel) this.infoPanel.innerText = `Loading: ${filename}...`;

        // Load PLY
        try {
            const mesh = await this.loadSplat(url);
            if (mesh) {
                mesh.visible = false;
                this.scene.add(mesh);
                this.frames[index] = mesh;
                if (this.infoPanel) this.updateOverlayInfo(); // Show stats on success
            } else {
                if (this.infoPanel) this.infoPanel.innerText = `Error Parsing: ${filename}`;
                console.error("FreeFlow Player: Mesh creation failed (null).");
            }
            return mesh;
        } catch (e) {
            console.error("FreeFlow Player Load Error:", e);
            if (this.infoPanel) this.infoPanel.innerText = `Error Loading: ${e.message}`;
            return null;
        }
    }

    async loadSplat(url) {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const buffer = await response.arrayBuffer();
        return this.createSplatMesh(buffer);
    }

    createSplatMesh(buffer) {
        // Minimal PLY Parser for binary little endian
        const headerEnd = "end_header\n";
        const decoder = new TextDecoder();
        const headerChunk = new Uint8Array(buffer, 0, 2000);
        const headerStr = decoder.decode(headerChunk);
        const endIdx = headerStr.indexOf(headerEnd);
        if (endIdx === -1) {
            console.error("Ply Header not found!");
            if (this.infoPanel) this.infoPanel.innerText = "Error: PLY header not found.";
            return null;
        }

        // Check Format
        const isBinary = headerStr.includes("binary_little_endian");
        if (!isBinary) {
            console.warn("FreeFlow Player: ASCII PLY not fully supported, expecting Binary Little Endian.");
            // TODO: Add ASCII parser if needed. Most 3DGS is binary.
            // We return null to warn user.
            if (this.infoPanel) this.infoPanel.innerHTML = "<div style='color:red'>Error: ASCII PLY detected.<br>Please export Binary PLY.</div>";
            return null;
        }

        const bodyStart = endIdx + headerEnd.length;
        const vertexResult = /element vertex (\d+)/.exec(headerStr);
        if (!vertexResult) return null;
        const vertexCount = parseInt(vertexResult[1]);

        // Parse properties with CORRECT byte offsets
        const propInfo = []; // {name, type, offset, size}
        let currentOffset = 0;
        const lines = headerStr.split('\n');

        for (let l of lines) {
            if (l.startsWith('property float')) {
                const name = l.split(' ')[2];
                propInfo.push({ name, type: 'float', offset: currentOffset, size: 4 });
                currentOffset += 4;
            } else if (l.startsWith('property uchar')) {
                const name = l.split(' ')[2];
                propInfo.push({ name, type: 'uchar', offset: currentOffset, size: 1 });
                currentOffset += 1;
            } else if (l.startsWith('property int')) {
                const name = l.split(' ')[2];
                propInfo.push({ name, type: 'int', offset: currentOffset, size: 4 });
                currentOffset += 4;
            }
        }

        const stride = currentOffset;

        // Helper to get property offset by name
        const getProp = (name) => propInfo.find(p => p.name === name);
        const getPropAlt = (name1, name2) => propInfo.find(p => p.name === name1 || p.name === name2);

        const dataView = new DataView(buffer, bodyStart);

        const positions = new Float32Array(vertexCount * 3);
        const colors = new Float32Array(vertexCount * 4);
        const scales = new Float32Array(vertexCount * 3);
        const rots = new Float32Array(vertexCount * 4);

        // Get property info
        const xProp = getProp('x');
        const yProp = getProp('y');
        const zProp = getProp('z');
        const rProp = getPropAlt('f_dc_0', 'red');
        const gProp = getPropAlt('f_dc_1', 'green');
        const bProp = getPropAlt('f_dc_2', 'blue');
        const opProp = getProp('opacity');
        const sxProp = getProp('scale_0');
        const syProp = getProp('scale_1');
        const szProp = getProp('scale_2');
        const r0Prop = getProp('rot_0');
        const r1Prop = getProp('rot_1');
        const r2Prop = getProp('rot_2');
        const r3Prop = getProp('rot_3');

        // Debug: Log first vertex to check values
        console.log("PLY Props:", propInfo.map(p => `${p.name}:${p.offset}`).join(', '));

        const SH_C0 = 0.28209479177387814;

        for (let i = 0; i < vertexCount; i++) {
            const base = i * stride;

            // Position
            positions[3 * i] = xProp ? dataView.getFloat32(base + xProp.offset, true) : 0;
            positions[3 * i + 1] = yProp ? dataView.getFloat32(base + yProp.offset, true) : 0;
            positions[3 * i + 2] = zProp ? dataView.getFloat32(base + zProp.offset, true) : 0;

            // Color
            let r = 0, g = 0, b = 0;
            if (rProp) {
                if (rProp.type === 'float') r = dataView.getFloat32(base + rProp.offset, true);
                else if (rProp.type === 'uchar') r = dataView.getUint8(base + rProp.offset) / 255.0;
            }
            if (gProp) {
                if (gProp.type === 'float') g = dataView.getFloat32(base + gProp.offset, true);
                else if (gProp.type === 'uchar') g = dataView.getUint8(base + gProp.offset) / 255.0;
            }
            if (bProp) {
                if (bProp.type === 'float') b = dataView.getFloat32(base + bProp.offset, true);
                else if (bProp.type === 'uchar') b = dataView.getUint8(base + bProp.offset) / 255.0;
            }

            // Convert SH to RGB if needed (values outside 0-1 range suggest SH)
            // Fix: Check explicit property name. If 'red', it's NOT SH.
            if (rProp && rProp.type === 'float') {
                const isSH = (rProp.name === 'f_dc_0' && gProp.name === 'f_dc_1' && bProp.name === 'f_dc_2');
                if (isSH) {
                    // SH coeffs are typically small floats around -1 to 1
                    r = r * SH_C0 + 0.5;
                    g = g * SH_C0 + 0.5;
                    b = b * SH_C0 + 0.5;
                }
            }

            // Opacity (sigmoid)
            let a = 1.0;
            if (opProp) {
                const opRaw = dataView.getFloat32(base + opProp.offset, true);
                a = 1 / (1 + Math.exp(-opRaw));
            }

            colors[4 * i] = Math.max(0, Math.min(1, r));
            colors[4 * i + 1] = Math.max(0, Math.min(1, g));
            colors[4 * i + 2] = Math.max(0, Math.min(1, b));
            colors[4 * i + 3] = a;

            // Scales - GS3D Reference: Always apply Math.exp() (Log Scale)
            // (INRIAV1PlyParser.js line 148: Math.exp(rawSplat[SCALE_0]))
            let sx = sxProp ? dataView.getFloat32(base + sxProp.offset, true) : -4.0;
            let sy = syProp ? dataView.getFloat32(base + syProp.offset, true) : -4.0;
            let sz = szProp ? dataView.getFloat32(base + szProp.offset, true) : -4.0;

            scales[3 * i] = Math.exp(sx);
            scales[3 * i + 1] = Math.exp(sy);
            scales[3 * i + 2] = Math.exp(sz);

            // Quaternion (w, x, y, z) - stored as rot_0=w, rot_1=x, rot_2=y, rot_3=z
            // GS3D Reference: Normalize immediately (INRIAV1PlyParser.js line 195)
            // Critical for correct rotation matrix construction in shader
            let r0 = r0Prop ? dataView.getFloat32(base + r0Prop.offset, true) : 1.0;
            let r1 = r1Prop ? dataView.getFloat32(base + r1Prop.offset, true) : 0.0;
            let r2 = r2Prop ? dataView.getFloat32(base + r2Prop.offset, true) : 0.0;
            let r3 = r3Prop ? dataView.getFloat32(base + r3Prop.offset, true) : 0.0;

            const qLen = Math.sqrt(r0 * r0 + r1 * r1 + r2 * r2 + r3 * r3);
            if (qLen > 1e-6) {
                const invLen = 1.0 / qLen;
                r0 *= invLen;
                r1 *= invLen;
                r2 *= invLen;
                r3 *= invLen;
            }

            rots[4 * i] = r0;
            rots[4 * i + 1] = r1;
            rots[4 * i + 2] = r2;
            rots[4 * i + 3] = r3;

            // Debug first vertex with computed values
            // Debug first vertex with computed values
            if (i === 0) {
                console.log(`[PLY Debug] First vertex: pos=(${positions[0].toFixed(3)},${positions[1].toFixed(3)},${positions[2].toFixed(3)})`);
                console.log(`[PLY Debug] First vertex: scale=(${scales[0].toFixed(6)},${scales[1].toFixed(6)},${scales[2].toFixed(6)}) [Format: LOG (Forced)]`);
                console.log(`[PLY Debug] First vertex: rot=(${rots[0].toFixed(4)},${rots[1].toFixed(4)},${rots[2].toFixed(4)},${rots[3].toFixed(4)})`);
            }
        }

        // --- INSTANCED QUAD GEOMETRY (Like GaussianSplats3D/SuperSplat) ---
        // GL_POINTS cannot render oriented ellipses. Must use instanced quads.

        const geo = new window.THREE.InstancedBufferGeometry();

        // Base quad vertices (will be transformed per-instance)
        // Quad from -1 to 1 (matching GaussianSplats3D standard)
        const quadPositions = new Float32Array([
            -1, -1, 0,
            1, -1, 0,
            1, 1, 0,
            -1, 1, 0
        ]);

        const quadUVs = new Float32Array([
            0, 0,
            1, 0,
            1, 1,
            0, 1
        ]);

        const quadIndices = new Uint16Array([0, 1, 2, 0, 2, 3]);

        geo.setAttribute('position', new window.THREE.BufferAttribute(quadPositions, 3));
        geo.setAttribute('uv', new window.THREE.BufferAttribute(quadUVs, 2));
        geo.setIndex(new window.THREE.BufferAttribute(quadIndices, 1));

        // Instance attributes (per-splat data)
        geo.setAttribute('splatCenter', new window.THREE.InstancedBufferAttribute(positions, 3));
        geo.setAttribute('splatColor', new window.THREE.InstancedBufferAttribute(colors, 4));
        geo.setAttribute('splatScale', new window.THREE.InstancedBufferAttribute(scales, 3));
        geo.setAttribute('splatRot', new window.THREE.InstancedBufferAttribute(rots, 4));

        // Set instance count
        geo.instanceCount = vertexCount;

        // --- GAUSSIAN SPLATS 3D SHADER PORT (Native Three.js) ---
        // Ported from https://github.com/mkkellogg/GaussianSplats3D/blob/main/src/splatmesh/SplatMaterial3D.js
        // Correctly handles Jacobian, Viewport, and Three.js coordinate systems.

        const vertexShader = `
            attribute vec3 splatCenter;
            attribute vec4 splatColor;
            attribute vec3 splatScale;
            attribute vec4 splatRot;

            varying vec4 vColor;
            varying vec2 vPosition;

            uniform vec2 viewport;
            uniform vec2 basisViewport;
            uniform vec2 focal;
            uniform float inverseFocalAdjustment;
            uniform float scaleModifier;
            uniform float renderMode;

            const float sqrt8 = sqrt(8.0);

            // Helper: Quaternion to Rotation Matrix
            mat3 quaternionToRotationMatrix(float x, float y, float z, float w) {
                float s = 1.0 / sqrt(w * w + x * x + y * y + z * z);
                return mat3(
                    1.0 - 2.0 * (y * y + z * z),
                    2.0 * (x * y + w * z),
                    2.0 * (x * z - w * y),
                    2.0 * (x * y - w * z),
                    1.0 - 2.0 * (x * x + z * z),
                    2.0 * (y * z + w * x),
                    2.0 * (x * z + w * y),
                    2.0 * (y * z - w * x),
                    1.0 - 2.0 * (x * x + y * y)
                );
            }

            void main () {
                vColor = splatColor;
                vPosition = position.xy;

                if (renderMode > 0.5) {
                    // Point Cloud Mode
                    vec4 centerWorld = modelMatrix * vec4(splatCenter, 1.0);
                    vec4 centerView = viewMatrix * centerWorld;
                    vec4 clipPos = projectionMatrix * centerView;
                    // Dynamic Point Size based on Global Scale slider
                    float pointSize = 3.0 * scaleModifier;  
                    vec2 offset = position.xy * pointSize;
                    clipPos.xy += offset * 2.0 / viewport * clipPos.w;
                    gl_Position = clipPos;
                    return;
                }

                // Compute 3D Covariance Matrix from Scale/Rot
                // M = R * S
                // Reverting to the empirically correct swizzle for this dataset
                mat3 R = quaternionToRotationMatrix(splatRot.y, splatRot.z, splatRot.w, splatRot.x);
                vec3 s = splatScale * scaleModifier;
                s = max(s, vec3(0.001)); // Prevent degenerate needle splats (aliasing fix)
                // Reverted scale swap
                mat3 S = mat3(
                    s.x, 0.0, 0.0,
                    0.0, s.y, 0.0,
                    0.0, 0.0, s.z
                );
                mat3 M = R * S;
                mat3 Vrk = M * transpose(M); // 3D Covariance

                // Compute View Center
                vec4 ViewCenterWorld = modelMatrix * vec4(splatCenter, 1.0);
                vec4 viewCenter = viewMatrix * ViewCenterWorld;

                // Clip if behind camera (Three.js view space is negative Z)
                // Relaxed clip to avoid popping
                if (viewCenter.z > -0.1) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }

                // Construct Jacobian of the affine approximation of the projection matrix
                // 3DGS Paper / GaussianSplats3D implementation
                float sz = 1.0 / (viewCenter.z * viewCenter.z);
                mat3 J = mat3(
                    focal.x / viewCenter.z, 0., -(focal.x * viewCenter.x) * sz,
                    0., focal.y / viewCenter.z, -(focal.y * viewCenter.y) * sz,
                    0., 0., 0.
                );

                // Concatenate projection approx with model-view
                // Note: GaussianSplats3D assumes modelViewMatrix which is viewMatrix * modelMatrix.
                // Three.js provides modelViewMatrix attribute/uniform automatically!
                
                mat3 W_mat = transpose(mat3(modelViewMatrix));
                mat3 T = W_mat * J;

                // Transform 3D covariance to 2D
                mat3 cov2Dm = transpose(T) * Vrk * T;

                // Low-pass filter (antialiasing - standard minimum variance)
                cov2Dm[0][0] += 0.5;
                cov2Dm[1][1] += 0.5;

                // Eigenvalue decomposition
                float a = cov2Dm[0][0];
                float d = cov2Dm[1][1];
                float b = cov2Dm[0][1];
                
                float D = a * d - b * b;
                float trace = a + d;
                float traceOver2 = 0.5 * trace;
                float term2 = sqrt(max(0.1, traceOver2 * traceOver2 - D));
                float eigenValue1 = traceOver2 + term2;
                float eigenValue2 = traceOver2 - term2;

                if (eigenValue2 <= 0.0) {
                    gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
                    return;
                }

                vec2 eigenVector1 = normalize(vec2(b, eigenValue1 - a));
                vec2 eigenVector2 = vec2(eigenVector1.y, -eigenVector1.x);

                // Scale by sqrt(8) standard deviations
                vec2 basisVector1 = eigenVector1 * min(sqrt8 * sqrt(eigenValue1), 2048.0);
                vec2 basisVector2 = eigenVector2 * min(sqrt8 * sqrt(eigenValue2), 2048.0);

                // NDC Offset - MUST include inverseFocalAdjustment per GS3D reference
                vec2 ndcOffset = vec2(vPosition.x * basisVector1 + vPosition.y * basisVector2) * basisViewport * 2.0 * inverseFocalAdjustment;

                // Combine with Projected Center
                vec4 clipCenter = projectionMatrix * viewCenter;
                vec3 ndcCenter = clipCenter.xyz / clipCenter.w;
                
                vec4 quadPos = vec4(ndcCenter.xy + ndcOffset, ndcCenter.z, 1.0);
                gl_Position = quadPos;

                // CRITICAL: Scale vPosition for Fragment Shader - GS3D reference line 213
                vPosition *= sqrt8;
            }
        `;

        // Fragment shader exactly matching GaussianSplats3D (temp_gs3d/src/splatmesh/SplatMaterial3D.js lines 235-251)
        const fragmentShader = `
            precision highp float;
            varying vec4 vColor;
            varying vec2 vPosition;
            uniform float renderMode;
            
            // Color Grading Uniforms
            uniform float uTemperature;
            uniform float uSaturation;
            uniform float uBrightness;
            uniform float uBlackPoint;
            uniform float uWhitePoint;
            uniform float uTransparency;

            void main () {
                if (renderMode > 0.5) {
                     // Circle Crop (Unit Circle). vPosition is -1..1 generally.
                     // dot > 1.0 discards corners -> Circular point.
                     if (dot(vPosition, vPosition) > 1.0) discard;
                     gl_FragColor = vec4(vColor.rgb * vColor.a, vColor.a);
                     return;
                }

                // Compute squared distance from center
                float A = dot(vPosition, vPosition);
                if (A > 8.0) discard;

                // Base Opacity
                float opacity = exp(-0.5 * A) * (vColor.a * (1.0 - uTransparency));

                // --- COLOR GRADING ---
                vec3 c = vColor.rgb;

                // 1. Temperature (Warm = +R -B, Cool = -R +B)
                // Linear approximation
                c.r += uTemperature * 0.15;
                c.b -= uTemperature * 0.15;

                // 2. Saturation
                float lum = dot(c, vec3(0.2126, 0.7152, 0.0722));
                c = mix(vec3(lum), c, uSaturation);

                // 3. Brightness
                c += uBrightness;

                // 4. Levels (Black/White Point)
                c = (c - vec3(uBlackPoint)) / max(vec3(0.001), vec3(uWhitePoint - uBlackPoint));

                // Clamp
                c = clamp(c, 0.0, 1.0);

                // Output
                gl_FragColor = vec4(c, opacity);
            }
        `;

        const material = new window.THREE.ShaderMaterial({
            uniforms: {
                renderMode: { value: 0 },
                scaleModifier: { value: 1.0 },
                resolution: { value: new window.THREE.Vector2(this.element.clientWidth || 800, this.element.clientHeight || 600) },
                viewport: { value: new window.THREE.Vector2(this.element.clientWidth || 800, this.element.clientHeight || 600) },
                basisViewport: { value: new window.THREE.Vector2(1.0 / 800, 1.0 / 600) },
                focal: { value: new window.THREE.Vector2(800, 800) },
                inverseFocalAdjustment: { value: 1.0 },
                // Color Grading
                // Color Grading (Use stored state or defaults)
                uTemperature: { value: this.colorState?.uTemperature ?? 0.0 },
                uSaturation: { value: this.colorState?.uSaturation ?? 1.0 },
                uBrightness: { value: this.colorState?.uBrightness ?? 0.0 },
                uBlackPoint: { value: this.colorState?.uBlackPoint ?? 0.0 },
                uWhitePoint: { value: this.colorState?.uWhitePoint ?? 1.0 },
                uTransparency: { value: this.colorState?.uTransparency ?? 0.0 }
            },
            vertexShader: vertexShader,
            fragmentShader: fragmentShader,
            transparent: true,
            depthWrite: false,
            depthTest: true,
            blending: window.THREE.NormalBlending,
            side: window.THREE.DoubleSide
        });

        const mesh = new window.THREE.Mesh(geo, material);
        mesh.rotation.x = Math.PI;
        mesh.frustumCulled = false;

        // Backup original data for sorting (Immutable Source)
        // We must reorder attributes on CPU, so we need a stable source.
        mesh.userData.originalData = {
            positions: new Float32Array(positions),
            colors: new Float32Array(colors),
            scales: new Float32Array(scales),
            rots: new Float32Array(rots),
            vertexCount: vertexCount
        };

        return mesh;
    }

    // --- PLAYBACK ---
    togglePlay() {
        this.isPlaying = !this.isPlaying;
        this.playBtn.innerText = this.isPlaying ? "❚❚" : "▶";
    }

    stop() {
        this.isPlaying = false;
        this.playBtn.innerText = "▶";
    }

    play() {
        this.isPlaying = true;
        this.playBtn.innerText = "❚❚";
    }

    seek(frame) {
        this.currentFrame = frame;
        this.updateFrameVisibility();
    }

    updateFrameVisibility() {
        if (!this.frames || this.frames.length === 0) return;

        const idx = this.currentFrame % this.frames.length;

        // Auto-Load if missing
        if (!this.frames[idx]) {
            this.loadFrame(idx).then(() => {
                if (this.currentFrame % this.frames.length === idx) {
                    this.updateOverlayInfo();
                }
            });
        }

        if (this.frames[idx]) {
            this.updateOverlayInfo();
        }
        // Update UI
        this.scrubber.value = idx;
        const total = this.frames.length;
        // Correctly use Real Frame Number during Playback
        const real = this.getRealFrameNumber(idx);
        this.frameLabel.innerText = `${real} (${idx + 1}/${total})`;

        if (this.infoPanel) this.updateOverlayInfo();
    }

    centerCamera() {
        // Robust Focus (Debug Enabled)
        let targetMesh = this.frames[this.currentFrame];
        if (!targetMesh) {
            // Find ANY loaded mesh
            targetMesh = this.frames.find(m => m !== null);
        }

        if (!targetMesh) {
            if (this.infoPanel) this.infoPanel.innerText += "\n[Focus] No Mesh Loaded.";
            return;
        }

        if (!targetMesh.geometry.boundingBox) targetMesh.geometry.computeBoundingBox();
        const box = targetMesh.geometry.boundingBox;

        if (!box) {
            if (this.infoPanel) this.infoPanel.innerText += "\n[Focus] No BoundingBox.";
            return;
        }

        const center = new window.THREE.Vector3();
        box.getCenter(center);

        const size = new window.THREE.Vector3();
        box.getSize(size);
        const maxDim = Math.max(size.x, size.y, size.z);

        // Debug Info
        if (this.infoPanel) {
            this.infoPanel.innerHTML += `<div style='color:orange'>[Focus] C: ${center.x.toFixed(2)}, ${center.y.toFixed(2)}, ${center.z.toFixed(2)} | Hz: ${maxDim.toFixed(2)}</div>`;
        }

        // Filter outliers (optional? NaNs?)
        if (isNaN(center.x) || isNaN(maxDim)) {
            if (this.infoPanel) this.infoPanel.innerText += "\n[Focus] NaN detected in BBox.";
            return;
        }

        // Apply mesh transform to center if needed
        const worldCenter = center.clone().applyMatrix4(targetMesh.matrixWorld);

        this.controls.target.copy(worldCenter);

        // Move camera back
        const dir = this.camera.position.clone().sub(worldCenter).normalize();
        if (dir.lengthSq() < 0.001) dir.set(0, 0, 1); // fallback

        // If maxDim is 0 (one point?), assumes 1.0
        const dist = (maxDim || 1.0) * 2.0;

        this.camera.position.copy(worldCenter).add(dir.multiplyScalar(dist));
        this.camera.near = Math.max(0.1, dist / 1000.0);
        this.camera.far = Math.max(5000, dist * 10.0);
        this.camera.updateProjectionMatrix();

        this.camera.lookAt(worldCenter);
        this.controls.update();
    }

    // --- INTERACTION ---

    initInteraction() {
        if (!this.raycaster) this.raycaster = new window.THREE.Raycaster();
        this.raycaster.params.Points.threshold = 0.1; // Threshold

        this.renderer.domElement.addEventListener('dblclick', (e) => this.onDoubleClick(e));

        // Interactive Sorting (Fix for "Broken View" during rotation)
        if (this.controls) {
            this.controls.addEventListener('change', () => {
                if (this.renderMode === "Splats") {
                    this.sortSplats();
                }
            });
        }
    }

    intersectSplatMesh(mesh, raycaster) {
        if (!mesh || !mesh.geometry) return null;

        const splatCenter = mesh.geometry.attributes.splatCenter;
        if (!splatCenter) return null;

        const positions = splatCenter.array;
        const count = mesh.geometry.instanceCount || splatCenter.count;

        let minDist = Infinity;
        let closestIndex = -1;
        const closestPoint = new window.THREE.Vector3();
        const tempPt = new window.THREE.Vector3();

        // Transform Ray to Local Space to avoid transforming every point
        const inverseMatrix = new window.THREE.Matrix4().copy(mesh.matrixWorld).invert();
        const localRay = raycaster.ray.clone().applyMatrix4(inverseMatrix);

        // Threshold in local units (approx 0.1-0.5)
        const thresholdSq = 0.5 * 0.5;

        for (let i = 0; i < count; i++) {
            const x = positions[3 * i];
            const y = positions[3 * i + 1];
            const z = positions[3 * i + 2];

            tempPt.set(x, y, z);

            const distSq = localRay.distanceSqToPoint(tempPt);

            if (distSq < thresholdSq && distSq < minDist) {
                minDist = distSq;
                closestIndex = i;
                closestPoint.copy(tempPt);
            }
        }

        if (closestIndex !== -1) {
            // Transform back to world for camera usage
            closestPoint.applyMatrix4(mesh.matrixWorld);
            return {
                point: closestPoint,
                index: closestIndex,
                object: mesh,
                distance: raycaster.ray.origin.distanceTo(closestPoint)
            };
        }
        return null;
    }

    onDoubleClick(event) {
        if (!this.camera || !this.frames[this.currentFrame]) return;

        const rect = this.renderer.domElement.getBoundingClientRect();
        const x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        const y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

        this.raycaster.setFromCamera({ x, y }, this.camera);

        const mesh = this.frames[this.currentFrame];
        if (!mesh) return;

        // Custom Intersect for Splats
        const hit = this.intersectSplatMesh(mesh, this.raycaster);

        if (hit) {
            const p = hit.point;

            // Update Target (Pivot)
            this.controls.target.copy(p);

            // Frame it (Move Camera Closer)
            const dist = this.camera.position.distanceTo(p);
            // Don't get closer than 0.5, but allow zooming in up to 2.0 dist.
            // If already close, stay close.
            const targetDist = Math.max(0.5, Math.min(dist, 2.0));

            const dir = this.camera.position.clone().sub(p).normalize();
            this.camera.position.copy(p).add(dir.multiplyScalar(targetDist));

            this.controls.update();

            if (this.infoPanel) {
                this.infoPanel.innerHTML += `<div style='color:cyan'>[Focused] ${p.x.toFixed(2)},${p.y.toFixed(2)},${p.z.toFixed(2)}</div>`;
            }
        }
    }

    animate() {
        requestAnimationFrame(() => this.animate());

        // Visibility Check (Minimize Logic)
        if (this.node && this.node.flags && this.node.flags.collapsed) {
            if (this.element && this.element.style.display !== "none") {
                this.element.style.display = "none";
            }
            // Skip rendering if minimized
            return;
        }

        // Time logic
        if (this.isPlaying && this.sequenceData) {
            const now = performance.now();
            const interval = 1000 / this.fps;

            if (now - this.lastFrameTime > interval) {
                this.currentFrame = (this.currentFrame + 1) % this.frames.length;
                this.updateFrameVisibility();
                this.lastFrameTime = now;
            }
        }

        if (this.controls) this.controls.update();

        // Sort splats by depth before rendering (critical for correct alpha blending)
        // Respects user toggle (default: true)
        if (this.renderMode === "Splats" && this.depthSorting) {
            this.sortSplats();
        }

        // Toggle Visibility based on Render Frame (Async Swap)
        // This ensures what we see is sorted.
        if (this.frames.length > 0) {
            this.frames.forEach((m, i) => {
                if (m) m.visible = (i === this.renderFrameIndex);
            });
        }

        if (this.renderer) this.renderer.render(this.scene, this.camera);

        // WATCHDOG: Fix for "Freeze" / "Stuck" Playback
        // If sorting takes too long (>500ms), assume worker is stuck/deadlocked and reset.
        if (!this.sortReady && (performance.now() - this.lastSortTime > 500)) {
            console.warn("[Player] Sort watchdog triggered. Resetting sortReady.");
            this.sortReady = true;
        }
    }

    sortSplats() {
        if (!this.sortReady) return; // Worker busy

        const mesh = this.frames[this.currentFrame];
        if (!mesh || !mesh.geometry) return;

        const now = performance.now();
        // if (now - this.lastSortTime < 16) return; // REMOVED: Allow Maximum Throughput

        const geometry = mesh.geometry;
        const positions = geometry.attributes.splatCenter; // Correct attribute!
        if (!positions) return;

        // Explicitly update camera matrix ensure we sort for the CURRENT position
        // otherwise we might use the previous frame's matrix (causing "shimmering" or wrong occlusion)
        this.camera.updateMatrixWorld();
        this.camera.matrixWorldInverse.copy(this.camera.matrixWorld).invert();

        // CRITICAL FIX: Calculate ModelViewMatrix (view * model)
        // This accounts for the Mesh's rotation (e.g. the 180 flip)
        // If we only use CameraInverse, we are sorting in World Space but assuming points are World Space (which they are not, they are Local)
        mesh.updateMatrixWorld();
        const modelViewMatrix = new window.THREE.Matrix4();
        modelViewMatrix.multiplyMatrices(this.camera.matrixWorldInverse, mesh.matrixWorld);

        const viewMatrix = modelViewMatrix.elements;

        this.sortReady = false;
        this.lastSortTime = now;

        // Check cache
        const meshId = mesh.uuid;
        const msg = {
            viewMatrix: viewMatrix,
            meshId: meshId
        };

        if (this.cachedMeshId !== meshId) {
            // Send ALL attributes for caching (Original immutable source)
            if (mesh.userData.originalData) {
                msg.positions = mesh.userData.originalData.positions.buffer; // We send copies implicitly via postMessage unless transferred
                msg.colors = mesh.userData.originalData.colors.buffer;
                msg.scales = mesh.userData.originalData.scales.buffer;
                msg.rots = mesh.userData.originalData.rots.buffer;
                msg.vertexCount = mesh.userData.originalData.vertexCount;
            } else {
                // Fallback (Should typically imply originalData exists if loaded correctly)
                msg.positions = positions.array;
                msg.colors = geometry.attributes.splatColor.array;
                msg.scales = geometry.attributes.splatScale.array;
                msg.rots = geometry.attributes.splatRotation.array;
                msg.vertexCount = positions.count;
            }
            this.cachedMeshId = meshId;
        }

        this.worker.postMessage(msg);
    }

    // --- LITEGRAPH DRAW HOOK ---
    draw(ctx, node, widgetWidth, y, widgetHeight) {
        if (!this.element) return;

        // Sync Position Logic (Same as new visualizer)
        const ds = app.canvas.ds;
        const scale = ds.scale;
        const offset = ds.offset;

        const graphX = node.pos[0] + 10;
        const graphY = node.pos[1] + y;

        const cssX = (graphX + offset[0]) * scale;
        const cssY = (graphY + offset[1]) * scale;

        const canvasRect = ctx.canvas.getBoundingClientRect();
        const screenX = canvasRect.left + cssX;
        const screenY = canvasRect.top + cssY;

        const cssWidth = (widgetWidth - 20) * scale;

        const nodeHeight = node.size[1];
        const availableHeight = nodeHeight - y - 10;
        const cssHeight = (availableHeight > 0 ? availableHeight : widgetHeight) * scale;

        this.element.style.left = `${screenX}px`;
        this.element.style.top = `${screenY}px`;
        this.element.style.width = `${cssWidth}px`;
        this.element.style.height = `${cssHeight}px`;

        const isVisible = !node.flags.collapsed;
        this.element.style.display = isVisible ? 'block' : 'none';

        if (this.renderer) {
            // Get physical dimensions
            const pW = Math.max(1.0, Math.floor(cssWidth * window.devicePixelRatio));
            const pH = Math.max(1.0, Math.floor(cssHeight * window.devicePixelRatio));

            // Resize if needed (managing size manually)
            if (this.renderer.domElement.width !== pW || this.renderer.domElement.height !== pH) {
                this.renderer.setSize(cssWidth, cssHeight, false);
                this.camera.aspect = cssWidth / cssHeight;
                this.camera.updateProjectionMatrix();
            }

            // ALWAYS update uniforms based on current state to prevent tiny points
            const proj = this.camera.projectionMatrix.elements;
            const fx = proj[0] * pW * 0.5;
            const fy = proj[5] * pH * 0.5;
            const resVec = new window.THREE.Vector2(pW, pH);
            const focalVec = new window.THREE.Vector2(fx, fy);
            const basisViewportVec = new window.THREE.Vector2(1.0 / pW, 1.0 / pH);

            this.frames.forEach(m => {
                if (m && m.material && m.material.uniforms) {
                    if (m.material.uniforms.resolution) m.material.uniforms.resolution.value.copy(resVec);
                    if (m.material.uniforms.focal) m.material.uniforms.focal.value.copy(focalVec);
                    if (m.material.uniforms.viewport) m.material.uniforms.viewport.value.copy(resVec);
                    if (m.material.uniforms.basisViewport) m.material.uniforms.basisViewport.value.copy(basisViewportVec);
                }
            });

            // Still always update uniforms? If camera zooms (fov change), focal changes.
            // OrbitControls does NOT change fov, only position. So focal is constant unless resize.
            // But let's check just in case.
            // If new frame is loaded, it needs init setup.
        }

        // Update Info Overlay if visible
        if (this.showInfo && this.infoPanel) {
            this.updateOverlayInfo();
        }
    }

    createOverlay() {
        this.infoPanel = document.createElement("div");
        Object.assign(this.infoPanel.style, {
            position: "absolute", top: "10px", left: "10px",
            background: "rgba(0,0,0,0.6)", padding: "8px",
            color: "#0f0", fontFamily: "monospace", fontSize: "11px",
            pointerEvents: "none", border: "1px solid #333", borderRadius: "4px",
            display: "none", lineHeight: "1.4"
        });
        this.infoPanel.innerHTML = "<div>Initializing...</div>";
        this.element.appendChild(this.infoPanel);
    }

    updateOverlayInfo() {
        if (!this.infoPanel || !this.sequenceData) return;

        const idx = this.currentFrame % (this.frames.length || 1);
        const filename = this.sequenceData.files[idx] || "Loading...";
        const mesh = this.frames[idx];

        let countStr = "0";
        let memStr = "0 MB";

        if (mesh && mesh.geometry) {
            // With instanced geometry, position.count is the quad (4 vertices)
            // The actual splat count is in the instanced attributes like splatCenter
            const splatCenterAttr = mesh.geometry.attributes.splatCenter;
            const count = splatCenterAttr ? splatCenterAttr.count : 0;
            countStr = count.toLocaleString();
            // Approx memory: 3(pos) + 4(col) + 3(scale) + 4(rot) = 14 floats * 4 bytes = 56 bytes/splat
            const bytes = count * 56;
            memStr = (bytes / (1024 * 1024)).toFixed(2) + " MB";
        }

        this.infoPanel.innerHTML = `
            <div style="color: #ff8800; font-weight: 800; border-bottom: 1px solid rgba(255,255,255,0.05); padding-bottom: 5px; margin-bottom: 8px; letter-spacing: 1px;">SPLAT STATISTICS</div>
            <div style="margin-bottom: 4px; color: #228B22;"><span style="color: rgba(255,255,255,0.4)">FILE:</span> ${filename}</div>
             <div style="margin-bottom: 4px; color: #228B22;"><span style="color: rgba(255,255,255,0.4)">FRAME:</span> ${this.getRealFrameNumber(idx)} <span style="font-size:9px; opacity:0.5">(${idx + 1}/${this.frames.length})</span></div>
            <div style="margin-bottom: 4px; color: #228B22;"><span style="color: rgba(255,255,255,0.4)">COUNT:</span> ${countStr}</div>
            <div style="margin-bottom: 4px; color: #228B22;"><span style="color: rgba(255,255,255,0.4)">MEMORY:</span> ${memStr}</div>
            <div style="margin-bottom: 4px; color: #228B22;"><span style="color: rgba(255,255,255,0.4)">RENDER:</span> ${this.renderMode}</div>
            <div style="margin-bottom: 4px; color: #228B22;"><span style="color: rgba(255,255,255,0.4)">SORTING:</span> ${(this.depthSorting ? "ON" : "OFF")}</div>
            <div style="color: rgba(255,255,255,0.2); font-size: 9px; margin-top: 10px;">Viewport: ${this.element.clientWidth}x${this.element.clientHeight}</div>
        `;
    }

    getRealFrameNumber(index) {
        if (!this.sequenceData || !this.sequenceData.files || !this.sequenceData.files[index]) return index + 1;
        const filename = this.sequenceData.files[index];
        const match = filename.match(/(\d+)(?=\.\w+$)/); // Match last number before extension
        if (match) return parseInt(match[1]);

        // Fallback: Try any number in string?
        const matchAny = filename.match(/(\d+)/g);
        if (matchAny && matchAny.length > 0) return parseInt(matchAny[matchAny.length - 1]);

        return index + 1;
    }
}
