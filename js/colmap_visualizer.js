/**
 * FreeFlow COLMAP Visualizer - Three.js 3D View
 * Replicates COLMAP Desktop GUI inside ComfyUI
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

console.log("🌊 [FreeFlow] COLMAP Visualizer Extension Loaded - " + Date.now());

// Load Three.js from CDN
const THREE_CDN = "https://unpkg.com/three@0.160.0/build/three.module.js";
const ORBIT_CDN = "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";

let THREE = null;
let OrbitControls = null;

async function loadThreeJS() {
    if (THREE) return;

    try {
        THREE = await import(THREE_CDN);
        const orbitModule = await import(ORBIT_CDN);
        OrbitControls = orbitModule.OrbitControls;
        console.log("🌊 Three.js loaded successfully");
    } catch (e) {
        console.error("Failed to load Three.js:", e);
    }
}
// Preload Three.js (No longer preloaded globally, loaded on demand by widget)

/**
 * Create a 3D visualizer widget for the node
 * Architecture: DOM Overlay synced with LiteGraph Canvas
 */
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
        animationFrameId: null,
        isInitializing: false,
        hasError: false,

        // Optimize DOM updates
        lastBounds: { x: 0, y: 0, w: 0, h: 0, visible: false },

        // Configuration
        options: { serialize: false },

        getValue() {
            return this.vizData;
        },

        setValue(v) {
            this.vizData = v;
            if (this.scene) {
                this.updateScene(v);
            }
        },

        computeSize(width) {
            return [width, 400];
        },

        draw(ctx, node, widgetWidth, y, widgetHeight) {
            // Guard: Error state
            if (this.hasError) {
                ctx.fillStyle = "#311";
                ctx.fillRect(10, y, widgetWidth - 20, widgetHeight);
                ctx.fillStyle = "#c55";
                ctx.textAlign = "center";
                ctx.fillText("3D Engine Failed", widgetWidth / 2, y + widgetHeight / 2);
                return;
            }

            // Guard: Initializing
            if (!this.element) {
                if (!this.isInitializing) {
                    this.initElement();
                }
                // Show loading placeholder
                ctx.fillStyle = "#222";
                ctx.fillRect(10, y, widgetWidth - 20, widgetHeight);
                ctx.fillStyle = "#666";
                ctx.textAlign = "center";
                ctx.fillText("Initializing 3D...", widgetWidth / 2, y + widgetHeight / 2);
                return; // Skip the rest
            }

            // Calculate coordinates
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

            // Visibility Check
            const isVisible = !node.flags.collapsed;

            // OPTIMIZATION: Only touch DOM if values changed significantly (prevent Layout Thrashing)
            // Precision: 1px
            const b = this.lastBounds;
            if (Math.abs(b.x - screenX) > 1 || Math.abs(b.y - screenY) > 1 ||
                Math.abs(b.w - cssWidth) > 1 || Math.abs(b.h - cssHeight) > 1 ||
                b.visible !== isVisible) {

                const el = this.element;
                el.style.display = isVisible ? 'block' : 'none';

                if (isVisible) {
                    el.style.left = `${screenX}px`;
                    el.style.top = `${screenY}px`;
                    el.style.width = `${cssWidth}px`;
                    el.style.height = `${cssHeight}px`;

                    // Resize Renderer if needed
                    // Use standard logical pixels for setSize, internal resolution handled by setPixelRatio
                    const w = Math.floor(cssWidth);
                    const h = Math.floor(cssHeight);
                    if (this.renderer && (this.renderer.domElement.width !== w * window.devicePixelRatio || this.renderer.domElement.height !== h * window.devicePixelRatio)) {
                        this.renderer.setSize(w, h, false); // Update canvas style
                        this.camera.aspect = w / h;
                        this.camera.updateProjectionMatrix();
                    }
                } else {
                    // hidden
                }

                // Update Cache
                this.lastBounds = { x: screenX, y: screenY, w: cssWidth, h: cssHeight, visible: isVisible };

                // Animation Control
                if (isVisible) this.startLoop(); else this.stopLoop();
            }

            // Fallback Text
            if (!this.vizData && isVisible) {
                ctx.fillStyle = "#666";
                ctx.textAlign = "center";
                ctx.fillText("Waiting for COLMAP Data...", widgetWidth / 2, y + widgetHeight / 2);
            }
        },

        async initElement() {
            this.isInitializing = true;
            try {
                // 1. Container
                this.element = document.createElement("div");
                this.element.className = "comfy-colmap-viz";
                Object.assign(this.element.style, {
                    position: "fixed",
                    zIndex: "100",
                    background: "#000",
                    display: "none",
                    overflow: "hidden",
                    border: "1px solid #444",
                    pointerEvents: "auto" // Crucial for interaction
                });
                document.body.appendChild(this.element);

                // 2. Load Libs
                try {
                    if (!window.THREE) window.THREE = await import('./three.module.js');
                    if (!window.OrbitControls) window.OrbitControls = (await import('./OrbitControls.js')).OrbitControls;
                } catch (e) {
                    if (!window.THREE) window.THREE = await import('https://unpkg.com/three@0.160.0/build/three.module.js');
                }

                if (!window.THREE) throw new Error("Could not load Three.js");

                const THREE = window.THREE;
                const OrbitControls = window.OrbitControls;

                // 3. Scene Setup
                this.scene = new THREE.Scene();
                this.scene.background = new THREE.Color(0x222222);

                this.camera = new THREE.PerspectiveCamera(50, 1, 0.1, 1000);
                this.camera.position.set(2, 2, 2);

                this.renderer = new THREE.WebGLRenderer({ alpha: false, antialias: true });
                // CRITICAL FIX: High DPI support
                this.renderer.setPixelRatio(window.devicePixelRatio);
                this.element.appendChild(this.renderer.domElement);
                this.renderer.domElement.style.width = "100%";
                this.renderer.domElement.style.height = "100%";

                this.controls = new OrbitControls(this.camera, this.renderer.domElement);
                this.controls.enableDamping = true;
                this.controls.dampingFactor = 0.1; // Smooth

                this.scene.add(new THREE.GridHelper(10, 10));
                this.scene.add(new THREE.AxesHelper(1));

                // Group for COLMAP data (to handle coordinate conversion)
                this.dataGroup = new THREE.Group();
                // Fix Orientation: COLMAP is Y-Down, Three.js is Y-Up. Flip X axis 180 deg to align.
                this.dataGroup.rotation.x = Math.PI;
                this.scene.add(this.dataGroup);

                // 4. Start Loop
                this.startLoop();

                // 5. Initial Data
                // Help Overlay with Clickable Actions
                const help = document.createElement("div");
                Object.assign(help.style, {
                    position: "absolute",
                    bottom: "10px",
                    left: "10px",
                    color: "rgba(255, 255, 255, 0.6)",
                    fontFamily: "sans-serif",
                    fontSize: "11px",
                    userSelect: "none",
                    pointerEvents: "auto", // Allow clicking buttons
                    background: "rgba(0,0,0,0.3)",
                    padding: "4px",
                    borderRadius: "4px"
                });

                // Build HTML with clickable spans
                const btnStyle = "cursor:pointer;text-decoration:underline;color:#fff";
                help.innerHTML = `
                    <b>LMB</b> Rotate &nbsp; 
                    <b>RMB</b> Pan &nbsp; 
                    <b>DblClick</b> Focus &nbsp; 
                    <span style="${btnStyle}" id="btn_frame_${this.element.id}">[F] Frame All</span>
                `;
                this.element.appendChild(help);

                // Bind Click
                setTimeout(() => { // simple delay to ensure inserted
                    const btn = help.querySelector("span");
                    if (btn) btn.onclick = (e) => { e.stopPropagation(); this.frameAll(); };
                }, 100);

                if (this.vizData) this.updateScene(this.vizData);

                // Interaction: Double Click to Focus
                this.raycaster = new THREE.Raycaster();
                this.mouse = new THREE.Vector2();

                this.element.addEventListener('dblclick', (event) => {
                    const rect = this.renderer.domElement.getBoundingClientRect();
                    this.mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
                    this.mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

                    this.raycaster.setFromCamera(this.mouse, this.camera);
                    // Raycast against the data group children
                    const intersects = this.raycaster.intersectObjects(this.dataGroup.children, false);

                    if (intersects.length > 0) {
                        const p = intersects[0].point;
                        this.controls.target.copy(p);
                        this.controls.update();
                    }
                });

                // Interaction: Keyboard Shortcuts (Capture Phase)
                window.addEventListener('keydown', (e) => {
                    if (this.element && this.element.style.display !== 'none') {
                        if (e.key.toLowerCase() === 'f') {
                            // Check if mouse is over component to avoid stealing 'F' from other UI
                            if (this.element.matches(':hover')) {
                                e.preventDefault();
                                e.stopPropagation();
                                this.frameAll();
                            }
                        }
                    }
                }, { capture: true }); // Capture phase!

            } catch (e) {
                console.error("FreeFlow Viz Error:", e);
                this.hasError = true;
                if (this.element) {
                    this.element.remove();
                    this.element = null;
                }
            } finally {
                this.isInitializing = false;
            }
        },

        stopLoop() {
            if (this.animationFrameId) {
                cancelAnimationFrame(this.animationFrameId);
                this.animationFrameId = null;
            }
        },

        startLoop() {
            if (this.animationFrameId) return;
            const animate = () => {
                if (this.element && this.element.style.display !== 'none') {
                    if (this.renderer && this.scene && this.camera) {
                        this.controls.update();
                        this.renderer.render(this.scene, this.camera);
                    }
                }
                this.animationFrameId = requestAnimationFrame(animate);
            };
            animate();
        },

        // Data Update
        updateScene(data) {
            if (!this.dataGroup || !window.THREE) return;
            const THREE = window.THREE;

            // Clear Old
            const toRemove = [];
            this.dataGroup.traverse(o => { if (o.userData.isData) toRemove.push(o); });
            toRemove.forEach(o => this.dataGroup.remove(o));

            // Points
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

            // Cameras
            if (data.images) {
                const scale = 0.1;
                const geo = new THREE.ConeGeometry(scale, scale * 2, 4, 1, true);
                geo.rotateX(-Math.PI / 2);
                geo.rotateZ(Math.PI / 4);

                Object.values(data.images).forEach(img => {
                    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: 0xff0000, wireframe: true }));
                    const q = img.qvec; const t = img.tvec;
                    const quat = new THREE.Quaternion(q[1], q[2], q[3], q[0]).invert();
                    const pos = new THREE.Vector3(t[0], t[1], t[2]).applyQuaternion(quat).negate(); // C = -R't

                    mesh.position.copy(pos);
                    mesh.quaternion.copy(quat);
                    mesh.userData.isData = true;
                    this.dataGroup.add(mesh);
                });
            }

            this.updateSceneCamera(data); // smart framing
        },

        updateSceneCamera(data) {
            if (!this.controls || !this.camera || !window.THREE) return;
            const THREE = window.THREE;

            // WE MUST ACCOUNT FOR DATA GROUP ROTATION (X = 180 deg)!
            // Transform points to world space before calculating camera position?
            // Or calculate in Local Space and let the Group handle it?
            // OrbitControls operates in World Space.
            // If we set target to (0,0,0) world, that is (0,0,0) local.
            // But if we calculate 'behind points', we need to respect the flip.

            // Easiest way: Calculate centroids in LOCAL space (data coordinates), 
            // then apply the group matrix to get WORLD space targets for the camera.

            const groupMatrix = this.dataGroup.matrixWorld;

            // 1. Calculate Centroids (LOCAL)
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

            // Convert to WORLD
            // Since we just modified rotation, ensure matrix is updated
            this.dataGroup.updateMatrixWorld(true);

            const pointsCenterWorld = pointsCenterLocal.clone().applyMatrix4(groupMatrix);
            const camCenterWorld = camCenterLocal.clone().applyMatrix4(groupMatrix);

            const targetWorld = hasPoints ? pointsCenterWorld : (camCount > 0 ? camCenterWorld : new THREE.Vector3(0, 0, 0));
            this.controls.target.copy(targetWorld);

            // Place Camera (World Space)
            if (camCount > 0 && hasPoints) {
                const dir = new THREE.Vector3().subVectors(camCenterWorld, pointsCenterWorld);
                if (dir.lengthSq() < 0.001) dir.set(0, 0, 1);
                dir.normalize();

                const dist = camCenterWorld.distanceTo(pointsCenterWorld) * 2.0 + 2.0;
                this.camera.position.copy(pointsCenterWorld).add(dir.multiplyScalar(dist));
            } else {
                this.camera.position.copy(targetWorld).add(new THREE.Vector3(0, 2, 4)); // Generic offset
            }

            this.controls.update();
        },

        frameAll() {
            if (!this.scene) return;
            // Recalculate based on current data
            if (this.vizData) this.updateSceneCamera(this.vizData);
        },

        onRemove() {
            if (this.element) {
                this.element.remove();
            }
            if (this.animationFrameId) cancelAnimationFrame(this.animationFrameId);
            if (this.renderer) {
                this.renderer.dispose();
                this.renderer.domElement = null; // Clear reference
            }
            this.scene = null;
            this.camera = null;
            this.controls = null;
            this.vizData = null;
        }
    };

    return widget;
}

// Extension registration
app.registerExtension({
    name: "FreeFlow.ColmapVisualizer",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "FreeFlow_ColmapVisualizer") {

            // Add visualizer widget on node creation
            const origOnNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                if (origOnNodeCreated) {
                    origOnNodeCreated.apply(this, arguments);
                }

                const vizWidget = createVisualizerWidget(this);
                this.addCustomWidget(vizWidget);
                this._vizWidget = vizWidget;

                // Set minimum size
                this.setSize([400, 500]);
            };

            // Handle execution result
            nodeType.prototype.onExecuted = function (message) {
                console.log("🌊 Visualizer received data:", message);

                if (message && message.colmap_viz && message.colmap_viz[0]) {
                    if (this._vizWidget) {
                        this._vizWidget.setValue(message.colmap_viz[0]);
                    }
                }
            };

            // Cleanup on remove
            const origOnRemoved = nodeType.prototype.onRemoved;
            nodeType.prototype.onRemoved = function () {
                if (this._vizWidget && this._vizWidget.onRemove) {
                    this._vizWidget.onRemove();
                }
                if (origOnRemoved) {
                    origOnRemoved.apply(this, arguments);
                }
            };
        }
    }
});
