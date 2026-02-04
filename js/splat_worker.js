// FreeFlow Splat Sorting Worker
// Handles Depth Sorting AND Data Reordering to unblock Main Thread

let cachedPositions = null;
let cachedColors = null;
let cachedScales = null;
let cachedRots = null;

let cachedVertexCount = 0;
let cachedMeshId = null;

// Dual-Buffer for Output (Double Buffering to avoid allocation)
// We flip between buffer A and B
let bufferSet = 0;
let outBuffers = [
    { positions: null, colors: null, scales: null, rots: null },
    { positions: null, colors: null, scales: null, rots: null }
];

// Pre-allocated Sort Arrays
let depths = new Float32Array(0);
let indices = new Uint32Array(0);

self.onmessage = function (e) {
    const { viewMatrix, meshId, positions, colors, scales, rots, vertexCount } = e.data;

    // 1. Update Cache if New Mesh Data
    if (meshId !== cachedMeshId && positions) {
        cachedMeshId = meshId;
        cachedVertexCount = vertexCount;

        // Copy Input Data (Immutable Source)
        // We assume Transferable was used if possible, or copy
        cachedPositions = new Float32Array(positions);
        cachedColors = new Float32Array(colors);
        cachedScales = new Float32Array(scales);
        cachedRots = new Float32Array(rots);

        // Resize Internal Buffers
        if (depths.length !== cachedVertexCount) {
            depths = new Float32Array(cachedVertexCount);
            indices = new Uint32Array(cachedVertexCount);

            // Resize Output Buffers
            for (let i = 0; i < 2; i++) {
                outBuffers[i].positions = new Float32Array(cachedVertexCount * 3);
                outBuffers[i].colors = new Float32Array(cachedVertexCount * 4);
                outBuffers[i].scales = new Float32Array(cachedVertexCount * 3);
                outBuffers[i].rots = new Float32Array(cachedVertexCount * 4);
            }
        }
    }

    if (!cachedPositions || cachedVertexCount === 0) return;

    // 2. Calculate Depths
    const m2 = viewMatrix[2];
    const m6 = viewMatrix[6];
    const m10 = viewMatrix[10];
    const m14 = viewMatrix[14];

    let minDepth = Infinity;
    let maxDepth = -Infinity;

    for (let i = 0; i < cachedVertexCount; i++) {
        const x = cachedPositions[3 * i];
        const y = cachedPositions[3 * i + 1];
        const z = cachedPositions[3 * i + 2];

        // Dot Product with ViewDir (Z-row of ViewMatrix)
        const val = (x * m2) + (y * m6) + (z * m10) + m14;

        depths[i] = val;
        indices[i] = i;

        if (val < minDepth) minDepth = val;
        if (val > maxDepth) maxDepth = val;
    }

    // 3. High-Performance Counting Sort (O(N))
    const range = maxDepth - minDepth;
    if (range > 0.0001) {
        const invRange = 1.0 / range;
        const totalBuckets = Math.min(65535, cachedVertexCount);
        const counts = new Uint32Array(totalBuckets);
        const mappedDepths = new Uint16Array(cachedVertexCount);

        for (let i = 0; i < cachedVertexCount; i++) {
            let norm = (depths[i] - minDepth) * invRange;
            // Standard Norm (0..1)
            // Far (-Large) -> 0. Near (-Small) -> 1.
            // Bucket 0 = Far.

            const bucket = Math.floor(norm * (totalBuckets - 1));
            counts[bucket]++;
            mappedDepths[i] = bucket;
        }

        // Cumulative
        let starts = new Uint32Array(totalBuckets);
        let run = 0;
        for (let i = 0; i < totalBuckets; i++) {
            starts[i] = run;
            run += counts[i];
        }

        // Place
        const sortedIndices = new Uint32Array(cachedVertexCount);
        for (let i = 0; i < cachedVertexCount; i++) {
            const bucket = mappedDepths[i];
            const pos = starts[bucket];
            starts[bucket]++;
            sortedIndices[pos] = i;
        }

        indices = sortedIndices;
    }


    // 4. Reorder Data (The Heavy Lift)
    bufferSet = 1 - bufferSet;
    const out = outBuffers[bufferSet];
    const sInd = indices;

    const sP = cachedPositions;
    const sC = cachedColors;
    const sS = cachedScales;
    const sR = cachedRots;

    const tP = out.positions;
    const tC = out.colors;
    const tS = out.scales;
    const tR = out.rots;

    for (let i = 0; i < cachedVertexCount; i++) {
        const src = sInd[i];

        // Position (3)
        const i3 = i * 3;
        const s3 = src * 3;
        tP[i3] = sP[s3]; tP[i3 + 1] = sP[s3 + 1]; tP[i3 + 2] = sP[s3 + 2];

        // Scale (3)
        tS[i3] = sS[s3]; tS[i3 + 1] = sS[s3 + 1]; tS[i3 + 2] = sS[s3 + 2];

        // Color (4)
        const i4 = i * 4;
        const s4 = src * 4;
        tC[i4] = sC[s4]; tC[i4 + 1] = sC[s4 + 1]; tC[i4 + 2] = sC[s4 + 2]; tC[i4 + 3] = sC[s4 + 3];

        // Rot (4)
        tR[i4] = sR[s4]; tR[i4 + 1] = sR[s4 + 1]; tR[i4 + 2] = sR[s4 + 2]; tR[i4 + 3] = sR[s4 + 3];
    }

    // 5. Send Back (Transferables)
    const pBuf = tP.buffer.slice(0); // Copy to transfer
    const cBuf = tC.buffer.slice(0);
    const sBuf = tS.buffer.slice(0);
    const rBuf = tR.buffer.slice(0);

    self.postMessage({
        meshId: meshId,
        sortedPositions: pBuf,
        sortedColors: cBuf,
        sortedScales: sBuf,
        sortedRots: rBuf,
    }, [pBuf, cBuf, sBuf, rBuf]);
};
