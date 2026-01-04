import { app } from "../../scripts/app.js";

const TypeSlot = {
    Input: 1,
    Output: 2,
};

function registerDynamicInputNode(nodeName, prefix, type) {
    app.registerExtension({
        name: `TimNodes.${nodeName}`,
        async beforeRegisterNodeDef(nodeType, nodeData, appRef) {
            if (nodeData.name !== nodeName) return;

            // 1. On Node Created: Add initial input if missing
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const me = onNodeCreated?.apply(this);
                const hasInput = (this.inputs || []).some((i) => i && i.type === type);
                if (!hasInput) {
                    this.addInput(`${prefix}1`, type);
                }
                return me;
            };

            // 2. On Configure (Load): Restore inputs + ensure one empty slot
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function (info) {
                const me = onConfigure?.apply(this, arguments);
                try {
                    // Ensure we have enough slots for saved links
                    const savedInputs = Array.isArray(info?.inputs) ? info.inputs : [];
                    const desiredCount = savedInputs.filter((inp) => inp && inp.type === type).length;
                    const currentCount = (this.inputs || []).filter((i) => i && i.type === type).length;

                    // Add missing inputs
                    for (let i = currentCount; i < desiredCount; i++) {
                        this.addInput(`${prefix}${i + 1}`, type);
                    }

                    // Always ensure one empty slot at the end
                    let last = this.inputs[this.inputs.length - 1];
                    if (!last || last.link !== null) {
                        this.addInput(`${prefix}${this.inputs.length + 1}`, type);
                    }
                } catch (e) { }
                return me;
            };

            // 3. On Connections Change: Manage slots dynamically
            const onConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
                const me = onConnectionsChange?.apply(this, arguments);

                if (slotType === TypeSlot.Input) {
                    // Remove all empty inputs (except the last one)
                    // We iterate backwards to avoid index shifting issues
                    for (let i = this.inputs.length - 2; i >= 0; i--) {
                        const inp = this.inputs[i];
                        if (inp && inp.type === type && inp.link === null) {
                            this.removeInput(i);
                        }
                    }

                    // Rename all inputs to be sequential (image_1, image_2... or mask_1, mask_2...)
                    let count = 0;
                    for (let i = 0; i < this.inputs.length; i++) {
                        const inp = this.inputs[i];
                        if (inp && inp.type === type) {
                            count++;
                            inp.name = `${prefix}${count}`;
                        }
                    }

                    // Ensure the last input is empty; if not, add one
                    const last = this.inputs[this.inputs.length - 1];
                    if (last && last.link !== null) {
                        this.addInput(`${prefix}${count + 1}`, type);
                    }

                    // Gray out the last (empty) input
                    const emptySlot = this.inputs[this.inputs.length - 1];
                    if (emptySlot && emptySlot.link === null) {
                        emptySlot.color_off = "#666";
                    }

                    // If we have multiple empty slots at the end (rare race condition), trim them
                    while (this.inputs.length > 1) {
                        const last = this.inputs[this.inputs.length - 1];
                        const secondLast = this.inputs[this.inputs.length - 2];
                        if (last.link === null && secondLast.link === null) {
                            this.removeInput(this.inputs.length - 1);
                        } else {
                            break;
                        }
                    }

                    this?.graph?.setDirtyCanvas(true);
                }
                return me;
            };
        },
    });
}

// Register for NanoBatch (Images)
registerDynamicInputNode("ComfyG_NanoBatch", "image_", "IMAGE");

// Register for NanoBatchMask (Masks)
registerDynamicInputNode("ComfyG_NanoBatchMask", "mask_", "MASK");
