/**
 * FreeFlow 3D Visualizer - Three.js 3D View
 * Supports COLMAP Sparse Clouds + Guidance Meshes
 * STRICT PARITY WITH LEGACY VISUALIZER + NEW TIMELINE & GIZMOS
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("🌊 FreeFlow 3D Visualizer Loaded");

// Load Three.js + Addons from CDN
const THREE_CDN = "https://unpkg.com/three@0.160.0/build/three.module.js";
const ORBIT_CDN = "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";
const TRANSFORM_CDN = "https://unpkg.com/three@0.160.0/examples/jsm/controls/TransformControls.js";

let THREE = null;
let OrbitControls = null;
let TransformControls = null;

async function loadThreeJS() {
    if (THREE) return;
    try {
        THREE = await import(THREE_CDN);
        const orbitModule = await import(ORBIT_CDN);
        OrbitControls = orbitModule.OrbitControls;
        const transformModule = await import(TRANSFORM_CDN);
        TransformControls = transformModule.TransformControls;
        console.log("🌊 Three.js + Gizmos loaded successfully");
    } catch (e) {
        console.error("Failed to load Three.js:", e);
    }
}

function createVisualizerWidget(node) {
    const widget = {
        type: "colmap_viz",
        name: "3d_view",

        // State
        vizData: null,
        element: null,
        renderer: null,
        scene: null,
        camera: null,
        controls: null,
        transformControl: null, // GIZMO
        animationFrameId: null,
        isInitializing: false,
        hasError: false,

        // Mesh Sequence State
        meshFrames: [],
        currentFrame: 0,
        isPlaying: false,
        fps: 24,
        lastFrameTime: 0,

        // Optimize DOM updates
        lastBounds: { x: 0, y: 0, w: 0, h: 0, visible: false },
        options: { serialize: false },

        getValue() { return this.vizData; },
        setValue(v) {
            this.vizData = v;
            if (this.scene) { this.updateScene(v); }
        },

        computeSize(width) { return [width, 500]; }, // Taller for UI

        draw(ctx, node, widgetWidth, y, widgetHeight) {
            if (this.hasError) {
                ctx.fillStyle = "#311"; ctx.fillRect(10, y, widgetWidth - 20, widgetHeight);
                ctx.fillStyle = "#c55"; ctx.textAlign = "center"; ctx.fillText("3D Engine Failed", widgetWidth / 2, y + widgetHeight / 2);
                return;
            }

            if (!this.element) {
                if (!this.isInitializing) { this.initElement(); }
                ctx.fillStyle = "#222"; ctx.fillRect(10, y, widgetWidth - 20, widgetHeight);
                ctx.fillStyle = "#666"; ctx.textAlign = "center"; ctx.fillText("Initializing 3D...", widgetWidth / 2, y + widgetHeight / 2);
                return;
            }

            const ds = app.canvas.ds;
            const scale = ds.scale;
            // Force hide if scale is too small to be usable, or if collapsed
            const isVisible = !node.flags.collapsed && scale > 0.1;

            if (!isVisible) {
                this.element.style.display = 'none';
                this.stopLoop();
                return;
            }

            const offset = ds.offset;
            const graphX = node.pos[0] + 10;
            const graphY = node.pos[1] + y;
            const cssX = (graphX + offset[0]) * scale;
            const cssY = (graphY + offset[1]) * scale;
            const canvasRect = ctx.canvas.getBoundingClientRect();
            const screenX = canvasRect.left + cssX;
            const screenY = canvasRect.top + cssY;
            const cssWidth = (widgetWidth - 20) * scale;
            const availableHeight = node.size[1] - y - 10;
            const cssHeight = (availableHeight > 0 ? availableHeight : widgetHeight) * scale;

            // Apply Size/Pos
            const el = this.element;
            el.style.display = 'block';
            el.style.left = `${screenX}px`; el.style.top = `${screenY}px`;
            el.style.width = `${cssWidth}px`; el.style.height = `${cssHeight}px`;

            // Resize internal renderer
            const w = Math.floor(cssWidth); const h = Math.floor(cssHeight);
            if (this.renderer && (this.renderer.domElement.width !== w * window.devicePixelRatio || this.renderer.domElement.height !== h * window.devicePixelRatio)) {
                this.renderer.setSize(w, h, false);
                this.camera.aspect = w / h;
                this.camera.updateProjectionMatrix();
            }

            // UI Scaling: Inverse scale the UI layer so it stays relative to the Node's visual size, 
            // OR scale it so it shrinks with the node? 
            // User complained "Menus look enormous" when zooming out (node shrinks, UI text stays fixed px).
            // So we WANT the UI to shrink with the node.
            // Actually, HTML text stays fixed size. We need to applying transform scale!
            if (this.uiLayer) {
                // Determine a comfortable scale. 
                // If standard scale is 1.0 (100% zoom).
                // If zoom is 0.5, node is half size. UI should be half size?
                // Yes, to maintain "relative" look.
                this.uiLayer.style.transform = `scale(${scale})`;
                this.uiLayer.style.transformOrigin = "top left";
                this.uiLayer.style.width = `${100 / scale}%`; // Compensate width to fill 100% of scaled container
                this.uiLayer.style.height = `${100 / scale}%`;
            }

            this.startLoop();

            if (!this.vizData) {
                ctx.fillStyle = "#666"; ctx.textAlign = "center";
                ctx.fillText("Waiting for COLMAP Data...", widgetWidth / 2, y + widgetHeight / 2);
            }
        },

        async initElement() {
            this.isInitializing = true;
            try {
                this.element = document.createElement("div");
                this.element.className = "comfy-colmap-viz";
                Object.assign(this.element.style, {
                    position: "fixed", zIndex: "100", background: "#111", display: "none",
                    overflow: "hidden", border: "1px solid #444", pointerEvents: "auto", fontFamily: "sans-serif"
                });
                document.body.appendChild(this.element);

                // Load Libraries
                console.log("🌊 FreeFlow 3D Visualizer - V3 Loaded");
                try {
                    if (!window.THREE) window.THREE = await import('https://unpkg.com/three@0.160.0/build/three.module.js');
                    if (!window.OrbitControls) window.OrbitControls = (await import('https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js')).OrbitControls;
                    if (!window.TransformControls) window.TransformControls = (await import('https://unpkg.com/three@0.160.0/examples/jsm/controls/TransformControls.js')).TransformControls;
                    console.log("🌊 Three.js loaded successfully");
                } catch (e) {
                    console.error("🌊 Failed to load Three.js:", e);
                    throw e;
                }

                if (!window.THREE) throw new Error("Could not load Three.js");
                const THREE = window.THREE;
                const OrbitControls = window.OrbitControls;
                const TransformControls = window.TransformControls;

                // Scene
                this.scene = new THREE.Scene();
                this.scene.background = new THREE.Color(0x111111);
                this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
                this.camera.position.set(2, 2, 2);

                this.renderer = new THREE.WebGLRenderer({ alpha: false, antialias: true });
                this.renderer.setPixelRatio(window.devicePixelRatio);
                this.element.appendChild(this.renderer.domElement);
                this.renderer.domElement.style.width = "100%";
                this.renderer.domElement.style.height = "100%";

                // Controls
                this.controls = new OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.1;

                // Grid & Lights
                const grid = new THREE.GridHelper(20, 20, 0x333333, 0x222222);
                this.scene.add(grid);
                this.scene.add(new THREE.AmbientLight(0xffffff, 0.5));
                const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
                dirLight.position.set(5, 10, 7);
                this.scene.add(dirLight);

                // Data Group
                this.dataGroup = new THREE.Group();
                this.dataGroup.rotation.x = Math.PI; // COLMAP Y-Down
                this.scene.add(this.dataGroup);

                // Gizmo
                if (TransformControls) {
                    this.transformControl = new TransformControls(this.camera, this.renderer.domElement);
                    this.transformControl.addEventListener('change', () => {
                        if (this.transformControl.dragging && this.meshGroup) this.syncUIFromMesh();
                    });
                    this.transformControl.addEventListener('dragging-changed', (e) => this.controls.enabled = !e.value);
                    this.scene.add(this.transformControl);
                }

                // --- UI ---
                this.createBlenderUI();

                this.startLoop();
                if (this.vizData) this.updateScene(this.vizData);

                // Double click to focus (Attached to Container)
                this.element.addEventListener('dblclick', (e) => {
                    const rect = this.element.getBoundingClientRect();
                    const x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
                    const y = -((e.clientY - rect.top) / rect.height) * 2 + 1;

                    const raycaster = new THREE.Raycaster();
                    raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera);

                    // Increase threshold for points (COLMAP clouds are often sparse)
                    raycaster.params.Points.threshold = 0.2;

                    // Raycast against DataGroup (Points + Mesh)
                    const intersects = raycaster.intersectObjects(this.dataGroup.children, true);
                    if (intersects.length > 0) {
                        const p = intersects[0].point;

                        // 1. Center orbit on point
                        this.controls.target.copy(p);

                        // 2. Zoom in (Move camera 50% closer to point)
                        // Vector from Point to Camera
                        const offset = this.camera.position.clone().sub(p);
                        const dist = offset.length();
                        const newDist = dist * 0.4; // Zoom to 40% of current distance

                        // Limit min distance to avoid clipping
                        if (newDist > 0.1) {
                            const newPos = p.clone().add(offset.normalize().multiplyScalar(newDist));
                            this.camera.position.copy(newPos);
                        }

                        this.controls.update();
                        console.log("FreeFlow: Focused & Zoomed on point", p);
                    }
                });

                // Shortcuts
                window.addEventListener('keydown', (e) => {
                    if (this.element.style.display !== 'none' && !this.inputFocused) {
                        if (e.key === " ") { this.togglePlay(); e.preventDefault(); }
                        else if (e.key === "ArrowLeft") { this.stepFrame(-1); e.preventDefault(); }
                        else if (e.key === "ArrowRight") { this.stepFrame(1); e.preventDefault(); }
                        else if (e.key.toLowerCase() === "t") this.setGizmoMode('translate');
                        else if (e.key.toLowerCase() === "r") this.setGizmoMode('rotate');
                        else if (e.key.toLowerCase() === "s") this.setGizmoMode('scale');
                        else if (e.key.toLowerCase() === "f") this.frameAll(); // 'F' for Frame All
                    }
                });

            } catch (e) {
                console.error("FreeFlow Viz Error:", e);
                this.hasError = true;
            } finally {
                this.isInitializing = false;
            }
        },

        // --- BLENDER STYLE UI IMPLEMENTATION ---

        createBlenderUI() {
            // Container for all overlays
            this.uiLayer = document.createElement("div");
            Object.assign(this.uiLayer.style, {
                position: "absolute", top: "0", left: "0", width: "100%", height: "100%",
                pointerEvents: "none", display: "flex", flexDirection: "column", justifyContent: "space-between"
            });
            this.element.appendChild(this.uiLayer);

            this.createTopBar();
            this.createLeftToolbar();
            this.createRightSidebar();
            this.createBottomTimeline();
        },

        createTopBar() {
            const bar = document.createElement("div");
            Object.assign(bar.style, {
                position: "absolute", top: "10px", right: "10px", width: "110px", // Exact width match
                display: "flex", justifyContent: "space-between", pointerEvents: "auto",
                background: "rgba(30,30,30,0.9)", padding: "4px",
                borderRadius: "6px 6px 0 0", // Rounded top only
                border: "1px solid #444", borderBottom: "none",
                backdropFilter: "blur(4px)"
            });

            // Shading Modes
            const createIconBtn = (iconSvg, title, onClick, isActive) => {
                const btn = document.createElement("div");
                btn.title = title;
                btn.innerHTML = iconSvg;
                Object.assign(btn.style, {
                    width: "24px", height: "24px", cursor: "pointer", display: "flex",
                    alignItems: "center", justifyContent: "center", borderRadius: "4px",
                    fill: isActive ? "#4a90e2" : "#ccc", background: isActive ? "#222" : "transparent"
                });
                btn.onmouseenter = () => { if (!btn.classList.contains('active')) btn.style.fill = "#fff"; };
                btn.onmouseleave = () => { if (!btn.classList.contains('active')) btn.style.fill = "#ccc"; };
                btn.onclick = (e) => {
                    // Update Active State
                    Array.from(bar.children).forEach(c => {
                        c.classList.remove('active'); c.style.fill = "#ccc"; c.style.background = "transparent";
                    });
                    btn.classList.add('active'); btn.style.fill = "#4a90e2"; btn.style.background = "#222";
                    onClick(e);
                };
                return btn;
            };

            // Icons (Simple SVGs)
            const iconWire = `<svg viewBox="0 0 24 24"><path d="M12 2L2 12l10 10 10-10L12 2zm0 2.8l7.2 7.2-7.2 7.2-7.2-7.2 7.2-7.2z"/></svg>`;
            const iconSolid = `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/></svg>`;
            const iconPoints = `<svg viewBox="0 0 24 24"><circle cx="12" cy="12" r="2"/><circle cx="6" cy="6" r="2"/><circle cx="18" cy="6" r="2"/><circle cx="6" cy="18" r="2"/><circle cx="18" cy="18" r="2"/></svg>`;

            bar.appendChild(createIconBtn(iconWire, "Wireframe", () => this.setRenderMode("Wireframe"), true));
            bar.appendChild(createIconBtn(iconSolid, "Solid", () => this.setRenderMode("Solid"), false));
            bar.appendChild(createIconBtn(iconPoints, "Points", () => this.setRenderMode("Points"), false));

            this.uiLayer.appendChild(bar);
        },

        createLeftToolbar() {
            const bar = document.createElement("div");
            Object.assign(bar.style, {
                position: "absolute", top: "50px", left: "10px",
                display: "flex", flexDirection: "column", gap: "5px", pointerEvents: "auto",
                background: "rgba(30,30,30,0.8)", padding: "4px", borderRadius: "8px",
                border: "1px solid #444", backdropFilter: "blur(4px)", zIndex: "1000"
            });

            const createToolBtn = (iconSvg, title, onClick, active = false) => {
                const btn = document.createElement("div");
                btn.title = title;
                btn.innerHTML = iconSvg;
                Object.assign(btn.style, {
                    width: "32px", height: "32px", cursor: "pointer", display: "flex",
                    alignItems: "center", justifyContent: "center", borderRadius: "4px",
                    fill: active ? "#4a90e2" : "#aaa", background: active ? "#333" : "transparent"
                });
                btn.onmousedown = (e) => e.stopPropagation(); // Prevent drag start on canvas
                btn.onclick = (e) => {
                    e.stopPropagation();
                    e.stopImmediatePropagation();

                    // Reset all
                    Array.from(bar.children).forEach(c => { c.style.fill = "#aaa"; c.style.background = "transparent"; });
                    btn.style.fill = "#4a90e2"; btn.style.background = "#333";
                    // Reset all other tool buttons
                    Array.from(bar.children).forEach(c => {
                        if (c.classList.contains('tool-btn')) {
                            c.classList.remove('active');
                            c.style.fill = "#aaa";
                            c.style.background = "transparent";
                        }
                    });
                    btn.classList.add('active');
                    btn.style.fill = "#4a90e2";
                    btn.style.background = "#333";
                    onClick();
                };
                btn.classList.add('tool-btn'); // Add a class to identify tool buttons for reset
                if (active) {
                    btn.classList.add('active');
                }
                return btn;
            };

            // Select (Pointer)
            const btnSelect = createToolBtn(`<svg viewBox="0 0 24 24"><path d="M7 2l12 11.2-5.8.5 3.3 7.3-2.2.9-3.2-7.4-4.4 4V2z"/></svg>`, "Select", () => this.setGizmoMode(null));

            // Transform Tools
            const btnMove = createToolBtn(`<svg viewBox="0 0 24 24"><path d="M12 2L2 12h8v8h4v-8h8L12 2z"/></svg>`, "Move", () => this.setGizmoMode("translate"), true); // Default active
            const btnRotate = createToolBtn(`<svg viewBox="0 0 24 24"><path d="M12 2C6.5 2 2 6.5 2 12s4.5 10 10 10 10-4.5 10-10S17.5 2 12 2zm5 11h-4v4h-2v-4H7v-2h4V7h2v4h5v2z"/></svg>`, "Rotate", () => this.setGizmoMode("rotate"));
            const btnScale = createToolBtn(`<svg viewBox="0 0 24 24"><path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-7 9h-2V7h-2v5H6v2h2v5h2v-5h2v-2z"/></svg>`, "Scale", () => this.setGizmoMode("scale"));

            // Frame / Focus
            const btnFrame = createToolBtn(`<svg viewBox="0 0 24 24"><path d="M5 5h5V3H5c-1.1 0-2 .9-2 2v5h2V5zm10 0h5v5h2V5c0-1.1-.9-2-2-2h-5v2zM5 19v-5H3v5c0 1.1.9 2 2 2h5v-2H5zm14-5v5h-5v2h5c1.1 0 2-.9 2-2v-5h-2zM12 9c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/></svg>`, "Frame All / Center", () => this.frameAll());

            bar.appendChild(btnSelect);
            bar.appendChild(document.createElement("hr")); // Visual separator
            bar.appendChild(btnMove);
            bar.appendChild(btnRotate);
            bar.appendChild(btnScale);
            bar.appendChild(document.createElement("hr"));
            bar.appendChild(btnFrame);

            this.uiLayer.appendChild(bar);
        },

        createRightSidebar() {
            const panel = document.createElement("div");
            Object.assign(panel.style, {
                position: "absolute", top: "50px", right: "10px", width: "110px",
                background: "rgba(20,20,20,0.9)", backdropFilter: "blur(6px)",
                border: "1px solid #444", borderRadius: "6px",
                display: "flex", flexDirection: "column", gap: "4px", padding: "8px",
                color: "#ddd", fontSize: "10px", pointerEvents: "auto",
                boxShadow: "0 4px 12px rgba(0,0,0,0.4)", zIndex: "10000"
            });

            // Header with Collapse
            const header = document.createElement("div");
            Object.assign(header.style, {
                fontWeight: "bold", paddingBottom: "4px", borderBottom: "1px solid #444", marginBottom: "4px",
                cursor: "pointer", display: "flex", justifyContent: "space-between", alignItems: "center"
            });
            header.innerHTML = `<span>Transform</span><span style="font-size:8px">▼</span>`;
            const content = document.createElement("div");
            header.onclick = () => {
                if (content.style.display === "none") { content.style.display = "block"; header.children[1].innerText = "▼"; }
                else { content.style.display = "none"; header.children[1].innerText = "▶"; }
            };
            panel.appendChild(header);
            panel.appendChild(content);

            // Inputs (Vertical Stack)
            const addInputGroup = (label, onChange) => {
                const group = document.createElement("div");
                group.style.marginBottom = "8px";
                const lbl = document.createElement("div");
                lbl.innerText = label;
                lbl.style.marginBottom = "2px"; lbl.style.color = "#aaa";
                group.appendChild(lbl);

                // Container for vertical inputs
                const vStack = document.createElement("div");
                vStack.style.display = "flex"; vStack.style.flexDirection = "column"; vStack.style.gap = "2px";

                ["X", "Y", "Z"].forEach(axis => {
                    const row = document.createElement("div");
                    Object.assign(row.style, { display: "flex", alignItems: "center", background: "#333", borderRadius: "3px", overflow: "hidden" });

                    const l = document.createElement("div");
                    l.innerText = axis;
                    l.style.width = "14px"; l.style.textAlign = "center"; l.style.fontSize = "9px"; l.style.fontWeight = "bold";
                    l.style.color = axis === 'X' ? "#ff5555" : axis === 'Y' ? "#55ff55" : "#5555ff";
                    l.style.background = "rgba(255,255,255,0.05)";

                    const i = document.createElement("input");
                    i.type = "number"; i.step = 0.1; i.value = 0;
                    Object.assign(i.style, {
                        width: "100%", background: "transparent", border: "none", color: "#fff",
                        fontSize: "10px", padding: "2px 4px", outline: "none"
                    });

                    i.onfocus = () => this.inputFocused = true; i.onblur = () => this.inputFocused = false;
                    i.onchange = (e) => onChange(axis, parseFloat(e.target.value));

                    row.appendChild(l); row.appendChild(i);
                    vStack.appendChild(row);
                });
                group.appendChild(vStack);
                content.appendChild(group);
                return vStack; // Return stack to access children (XYZ rows)
            };

            this.uiPos = addInputGroup("Location", (a, v) => { if (this.meshGroup) this.meshGroup.position[a.toLowerCase()] = v; });
            this.uiRot = addInputGroup("Rotation", (a, v) => { if (this.meshGroup) this.meshGroup.rotation[a.toLowerCase()] = v * (Math.PI / 180); });
            this.uiScale = addInputGroup("Scale", (a, v) => { if (this.meshGroup) this.meshGroup.scale[a.toLowerCase()] = v; });

            // Opacity Slider (Extra)
            const opRow = document.createElement("div");
            opRow.style.marginTop = "4px";
            opRow.innerHTML = "<div style='color:#aaa;margin-bottom:2px'>Opacity</div>";
            const op = document.createElement("input");
            op.type = "range"; op.min = 0; op.max = 1; op.step = 0.1; op.value = 0.6; op.style.width = "100%";
            op.oninput = (e) => {
                if (this.activeMesh && this.activeMesh.material) this.activeMesh.material.opacity = parseFloat(e.target.value);
            };
            opRow.appendChild(op);
            content.appendChild(opRow);

            this.uiLayer.appendChild(panel);
        },

        createBottomTimeline() {
            const bar = document.createElement("div");
            Object.assign(bar.style, {
                position: "absolute", bottom: "10px", left: "10px", right: "10px",
                height: "32px", background: "rgba(15,15,15,0.9)",
                borderRadius: "4px", border: "1px solid #333", display: "flex", alignItems: "center",
                padding: "0 10px", gap: "10px", pointerEvents: "auto", zIndex: "10000"
            });

            // Play
            this.playBtn = document.createElement("button");
            this.playBtn.innerText = "▶";
            Object.assign(this.playBtn.style, {
                background: "none", border: "none", color: "#ccc", fontSize: "14px", cursor: "pointer", width: "20px"
            });
            this.playBtn.onclick = () => this.togglePlay();

            // Scrubber
            this.scrubber = document.createElement("input");
            this.scrubber.type = "range"; this.scrubber.min = 0; this.scrubber.max = 100; this.scrubber.value = 0;
            Object.assign(this.scrubber.style, { flex: "1", height: "4px", cursor: "pointer" });
            this.scrubber.oninput = (e) => {
                this.currentFrame = parseInt(e.target.value);
                this.updateMeshFrame(this.currentFrame);
                this.updateFrameLabel();
            };

            // Frame Label
            this.lblFrame = document.createElement("div");
            Object.assign(this.lblFrame.style, { color: "#888", fontSize: "11px", fontFamily: "monospace", minWidth: "100px", textAlign: "right" });
            this.lblFrame.innerText = "Frame: 0";

            bar.appendChild(this.playBtn);
            bar.appendChild(this.scrubber);
            bar.appendChild(this.lblFrame);

            this.uiLayer.appendChild(bar);
            this.timelineBar = bar;
        },

        updateFrameLabel() {
            if (!this.meshFrames) return;
            // "Frame: 123 (Idx/Total)" style
            const idx = this.currentFrame;
            let realFrameId = idx;

            // Use Backend Frame Numbers if available
            if (this.vizData && this.vizData.frame_numbers && this.vizData.frame_numbers[idx] !== undefined) {
                realFrameId = this.vizData.frame_numbers[idx];
            }

            this.lblFrame.innerText = `Frame: ${realFrameId} (${idx + 1}/${this.meshFrames.length})`;
        },

        // Overrides or Helpers
        setGizmoMode(mode) {
            if (this.transformControl) {
                if (mode) {
                    this.transformControl.enabled = true;
                    this.transformControl.visible = true;
                    this.transformControl.setMode(mode);
                } else {
                    this.transformControl.enabled = false;
                    this.transformControl.visible = false;
                }
            }
        },

        syncUIFromMesh() {
            if (!this.meshGroup || !this.uiPos) return;
            // Helper to find inputs in our new "vStack" structure
            const setV = (vStack, vec, isRot) => {
                // vStack has 3 children (rows for X, Y, Z). Each row has [Label, Input]
                const rows = Array.from(vStack.children);
                const xInp = rows[0].children[1];
                const yInp = rows[1].children[1];
                const zInp = rows[2].children[1];

                xInp.value = (vec.x * (isRot ? 180 / Math.PI : 1)).toFixed(2);
                yInp.value = (vec.y * (isRot ? 180 / Math.PI : 1)).toFixed(2);
                zInp.value = (vec.z * (isRot ? 180 / Math.PI : 1)).toFixed(2);
            };
            setV(this.uiPos, this.meshGroup.position);
            setV(this.uiRot, this.meshGroup.rotation, true);
            setV(this.uiScale, this.meshGroup.scale);
        },

        setRenderMode(mode) {
            if (!this.activeMesh) return;
            const THREE = window.THREE;
            const geo = this.activeMesh.geometry;
            let mat;
            if (mode === "Wireframe") mat = new THREE.MeshBasicMaterial({ color: 0x00ffff, wireframe: true, transparent: true, opacity: 0.5 });
            else if (mode === "Solid") mat = new THREE.MeshBasicMaterial({
                color: 0x00ffff,
                side: THREE.DoubleSide, transparent: true, opacity: 0.8
            });
            else mat = new THREE.PointsMaterial({ color: 0x00ffff, size: 0.04 });

            const isPoints = this.activeMesh.isPoints;
            const wantPoints = (mode === "Points");

            if (isPoints !== wantPoints) {
                const newObj = wantPoints ? new THREE.Points(geo, mat) : new THREE.Mesh(geo, mat);
                newObj.userData.isData = true;
                this.meshGroup.remove(this.activeMesh);
                this.meshGroup.add(newObj);
                this.activeMesh = newObj;
            } else {
                this.activeMesh.material = mat;
            }
        },


        togglePlay() {
            this.isPlaying = !this.isPlaying;
            this.playBtn.innerHTML = this.isPlaying ? "❚❚" : "▶";
        },

        stopLoop() { if (this.animationFrameId) { cancelAnimationFrame(this.animationFrameId); this.animationFrameId = null; } },
        startLoop() {
            if (this.animationFrameId) return;
            const animate = () => {
                if (this.element && this.element.style.display !== 'none') {
                    // Update Playback
                    if (this.isPlaying && this.meshFrames.length > 1) {
                        const now = Date.now();
                        const interval = 1000 / this.fps;
                        if (now - this.lastFrameTime > interval) {
                            this.currentFrame = (this.currentFrame + 1) % this.meshFrames.length;
                            this.updateMeshFrame(this.currentFrame);
                            // Update UI
                            this.scrubber.value = this.currentFrame;
                            this.updateFrameLabel();
                            this.lastFrameTime = now;
                        }
                    }

                    if (this.renderer && this.scene && this.camera) {
                        this.controls.update();
                        this.renderer.render(this.scene, this.camera);
                    }
                }
                this.animationFrameId = requestAnimationFrame(animate);
            };
            animate();
        },

        updateScene(data) {
            if (!this.dataGroup || !window.THREE) return;
            const THREE = window.THREE;

            // Clear
            const toRemove = [];
            this.dataGroup.traverse(o => { if (o.userData.isData) toRemove.push(o); });
            toRemove.forEach(o => this.dataGroup.remove(o));
            if (this.transformControl) this.transformControl.detach();

            // 1. Points Cloud
            if (data.points && data.points.positions) {
                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.Float32BufferAttribute(data.points.positions, 3));
                const cols = new Float32Array(data.points.colors.length);
                for (let i = 0; i < cols.length; i++) cols[i] = data.points.colors[i] / 255;
                geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
                const mat = new THREE.PointsMaterial({ size: 0.02, vertexColors: true });
                const mesh = new THREE.Points(geo, mat);
                mesh.userData.isData = true;
                this.dataGroup.add(mesh);
            }

            // 2. Cameras
            if (data.images) {
                // ... (Camera Viz Logic) ...
                const scale = 0.1;
                const geo = new THREE.ConeGeometry(scale, scale * 2, 4, 1, true);
                geo.rotateX(-Math.PI / 2); geo.rotateZ(Math.PI / 4);
                Object.values(data.images).forEach(img => {
                    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true }));
                    const q = img.qvec; const t = img.tvec;
                    const quat = new THREE.Quaternion(q[1], q[2], q[3], q[0]).invert();
                    const pos = new THREE.Vector3(t[0], t[1], t[2]).applyQuaternion(quat).negate();
                    mesh.position.copy(pos); mesh.quaternion.copy(quat);
                    mesh.userData.isData = true;
                    this.dataGroup.add(mesh);
                });
            }

            // 3. Guidance Mesh Sequence (Fixed Pivot & Rendering)
            if (data.mesh_sequence && data.mesh_sequence.length > 0) {
                console.log("FreeFlow: Mesh Sequence Received. Frames:", data.mesh_sequence.length);
                this.meshFrames = data.mesh_sequence;
                const faces = data.mesh_faces;
                console.log("FreeFlow: Mesh Faces:", faces ? faces.length : "None");

                // Calculate Centroid of Frame 0 to Center the Pivot
                const frame0 = this.meshFrames[0];
                console.log("FreeFlow: frame0 type:", typeof frame0, "isArray:", Array.isArray(frame0));
                console.log("FreeFlow: frame0 length:", frame0?.length, "first element:", frame0?.[0]);

                // Handle different data formats
                let verts;
                if (Array.isArray(frame0) && Array.isArray(frame0[0])) {
                    // Nested array: [[x,y,z], [x,y,z], ...]
                    verts = frame0.flat();
                } else if (Array.isArray(frame0)) {
                    // Already flat: [x,y,z,x,y,z,...]
                    verts = frame0;
                } else {
                    console.error("FreeFlow: Unknown frame0 format!");
                    return;
                }
                console.log("FreeFlow: After processing, verts length:", verts.length);

                // Compute Centroid
                const center = new THREE.Vector3();
                for (let i = 0; i < verts.length; i += 3) {
                    center.x += verts[i]; center.y += verts[i + 1]; center.z += verts[i + 2];
                }
                center.divideScalar(verts.length / 3);

                // Store Offset
                this.meshCentroid = center.clone();

                // Localize Vertices (Subtract Centroid)
                const localVerts = new Float32Array(verts.length);
                for (let i = 0; i < verts.length; i += 3) {
                    localVerts[i] = verts[i] - center.x;
                    localVerts[i + 1] = verts[i + 1] - center.y;
                    localVerts[i + 2] = verts[i + 2] - center.z;
                }

                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.BufferAttribute(localVerts, 3));

                const vertexCount = localVerts.length / 3;
                console.log("FreeFlow: Vertex count:", vertexCount, "Local verts length:", localVerts.length);

                // DECIDE RENDERING MODE based on Faces presence
                let object;
                if (faces && faces.length > 0) {
                    const indices = faces.flat();
                    const maxIndex = Math.max(...indices);
                    console.log("FreeFlow: Face indices count:", indices.length, "Max index:", maxIndex, "Vertex count:", vertexCount);

                    // Validate: max index must be < vertex count
                    if (maxIndex >= vertexCount) {
                        console.error("FreeFlow: INVALID MESH! Max face index", maxIndex, "exceeds vertex count", vertexCount);
                        console.warn("FreeFlow: Falling back to Points mode");
                        // Fallback to Points
                        const mat = new THREE.PointsMaterial({ color: 0x00ffff, size: 0.02 });
                        object = new THREE.Points(geo, mat);
                    } else {
                        geo.setIndex(indices);
                        geo.computeVertexNormals(); // Compute AFTER setting index
                        const mat = new THREE.MeshStandardMaterial({
                            color: 0x00ffff, roughness: 0.3, metalness: 0.2,
                            side: THREE.DoubleSide, transparent: true, opacity: 0.8
                        });
                        object = new THREE.Mesh(geo, mat);
                    }
                } else {
                    // Fallback to Points
                    const mat = new THREE.PointsMaterial({ color: 0x00ffff, size: 0.02 });
                    object = new THREE.Points(geo, mat);
                }

                object.userData.isData = true;
                this.activeMesh = object;

                // Create Group at Centroid
                this.meshGroup = new THREE.Group();
                this.meshGroup.position.copy(center);
                this.meshGroup.userData.isData = true;
                this.meshGroup.add(object);

                this.dataGroup.add(this.meshGroup);

                if (this.transformControl) {
                    this.transformControl.attach(this.meshGroup);
                    this.transformControl.setMode('translate');
                }

                // Sync UI
                this.syncUIFromMesh();

                // Initial UI State
                if (this.timelineBar) {
                    this.timelineBar.style.display = "flex";
                    this.scrubber.max = this.meshFrames.length - 1;
                    this.scrubber.value = 0;
                    this.currentFrame = 0;
                    this.updateMeshFrame(0); // Force initial update/normals
                    this.updateFrameLabel();
                    this.isPlaying = false;
                    this.playBtn.innerHTML = "▶";
                }
            }

            // Auto-Frame only if requested or default
            const shouldCenter = (data.options && data.options.auto_center !== undefined) ? data.options.auto_center : true;
            if (shouldCenter) {
                this.updateSceneCamera(data);
            }
        },

        updateMeshFrame(frameIdx) {
            if (!this.activeMesh || !this.meshFrames || !this.meshFrames[frameIdx]) return;
            const rawVerts = this.meshFrames[frameIdx].flat();

            // Apply Centroid Offset (Convert World -> Local)
            const posAttr = this.activeMesh.geometry.attributes.position;
            const arr = posAttr.array;
            const cx = this.meshCentroid.x;
            const cy = this.meshCentroid.y;
            const cz = this.meshCentroid.z;

            for (let i = 0; i < rawVerts.length; i += 3) {
                arr[i] = rawVerts[i] - cx;
                arr[i + 1] = rawVerts[i + 1] - cy;
                arr[i + 2] = rawVerts[i + 2] - cz;
            }

            posAttr.needsUpdate = true;
            this.activeMesh.geometry.computeVertexNormals();
        },

        updateSceneCamera(data) {
            // Strict Legacy Logic: 
            // Center on Points (or Cameras). 
            // (2,2,2) default if empty.
            if (!this.controls || !this.camera || !window.THREE) return;
            const THREE = window.THREE;
            const groupMatrix = this.dataGroup.matrixWorld;

            let pointsCenterLocal = new THREE.Vector3();
            let hasPoints = false;

            if (data.points && data.points.positions && data.points.positions.length > 0) {
                const geo = new THREE.BufferGeometry();
                geo.setAttribute('position', new THREE.Float32BufferAttribute(data.points.positions, 3));
                geo.computeBoundingSphere();
                if (geo.boundingSphere) {
                    pointsCenterLocal.copy(geo.boundingSphere.center);
                    hasPoints = true;
                }
                geo.dispose();
            }

            let camCenterLocal = new THREE.Vector3();
            let camCount = 0;
            if (data.images) {
                Object.values(data.images).forEach(img => {
                    const q = img.qvec; const t = img.tvec;
                    const quat = new THREE.Quaternion(q[1], q[2], q[3], q[0]).invert();
                    const pos = new THREE.Vector3(t[0], t[1], t[2]).applyQuaternion(quat).negate();
                    camCenterLocal.add(pos);
                    camCount++;
                });
            }
            if (camCount > 0) camCenterLocal.divideScalar(camCount);

            this.dataGroup.updateMatrixWorld(true);
            const pointsCenterWorld = pointsCenterLocal.clone().applyMatrix4(groupMatrix);
            const camCenterWorld = camCenterLocal.clone().applyMatrix4(groupMatrix);

            const targetWorld = hasPoints ? pointsCenterWorld : (camCount > 0 ? camCenterWorld : new THREE.Vector3(0, 0, 0));
            this.controls.target.copy(targetWorld);

            if (camCount > 0 && hasPoints) {
                const dir = new THREE.Vector3().subVectors(camCenterWorld, pointsCenterWorld);
                if (dir.lengthSq() < 0.001) dir.set(0, 0, 1);
                dir.normalize();
                const dist = camCenterWorld.distanceTo(pointsCenterWorld) * 2.0 + 2.0;
                this.camera.position.copy(pointsCenterWorld).add(dir.multiplyScalar(dist));
            } else {
                this.camera.position.copy(targetWorld).add(new THREE.Vector3(0, 2, 4));
            }
            this.controls.update();
        },

        frameAll() {
            if (!this.scene) return;
            if (this.vizData) this.updateSceneCamera(this.vizData);
        },

        onRemove() {
            if (this.element) { this.element.remove(); }
            if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
            if (this.renderer) { this.renderer.dispose(); this.renderer.domElement = null; }
            this.vizData = null;
        }
    };
    return widget;
}

app.registerExtension({
    name: "FreeFlow.3DVisualizer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("🌊 beforeRegisterNodeDef:", nodeData.name);
        if (nodeData.name === "FreeFlow_3DVisualizer") {
            console.log("🌊 MATCHED! Hooking FreeFlow_3DVisualizer...");
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origOnNodeCreated) origOnNodeCreated.apply(this, arguments);
                const vizWidget = createVisualizerWidget(this);
                this.addCustomWidget(vizWidget);
                this._vizWidget = vizWidget;
                this.setSize([400, 500]);
            };
            const origOnExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                if (origOnExecuted) origOnExecuted.apply(this, arguments);
                console.log("🌊 FreeFlow_3DVisualizer.onExecuted called:", message);
                if (message && message.freeflow_3d_viz && message.freeflow_3d_viz[0]) {
                    console.log("🌊 FreeFlow_3DVisualizer: Data received!", message.freeflow_3d_viz[0]);
                    if (this._vizWidget) this._vizWidget.setValue(message.freeflow_3d_viz[0]);
                } else {
                    console.warn("🌊 FreeFlow_3DVisualizer: NO freeflow_3d_viz in message!", Object.keys(message || {}));
                }
            };
            const origOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (this._vizWidget && this._vizWidget.onRemove) this._vizWidget.onRemove();
                if (origOnRemoved) origOnRemoved.apply(this, arguments);
            };
        }
    }
});
