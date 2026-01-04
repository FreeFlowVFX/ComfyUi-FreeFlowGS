// Web Worker for 3DGS Splat Sorting
// Optimized for performance

const cache = {};

self.onmessage = function (e) {
    let { positions, viewMatrix, vertexCount, meshId } = e.data;

    // Update Cache if new data provided
    if (meshId && positions) {
        cache[meshId] = {
            positions: (positions instanceof ArrayBuffer) ? new Float32Array(positions) : positions,
            vertexCount: vertexCount
        };
    }

    // Retrieve from Cache if simple update
    if (!positions && meshId && cache[meshId]) {
        positions = cache[meshId].positions;
        vertexCount = cache[meshId].vertexCount;
    }

    if (!positions || !viewMatrix) {
        // If we still don't have positions (and no cache), we can't sort.
        // But we MUST reply to unlock the main thread (sortReady = true).
        if (meshId) {
            self.postMessage({ sortedIndices: null, meshId: meshId });
        }
        return;
    }

    // View matrix Layout:
    // [0, 1, 2, 3]
    // [4, 5, 6, 7]
    // [8, 9, 10, 11]
    // [12, 13, 14, 15]
    // We need the Z-axis vector (row 2 for column-major, or correct indices for dot product)
    // 3D Point p -> View Space v = V * p
    // v.z = m[2]*x + m[6]*y + m[10]*z + m[14]

    const m = viewMatrix;
    // const vertices = (positions instanceof ArrayBuffer) ? new Float32Array(positions) : positions; // Handled in cache
    const vertices = positions;
    const count = vertexCount;

    // 1. Calculate Depths
    // We can use a shared buffer or create one.
    // Let's assume we create a Float32Array for depths.
    const depths = new Float32Array(count);
    const indices = new Uint32Array(count);

    for (let i = 0; i < count; i++) {
        const x = vertices[3 * i];
        const y = vertices[3 * i + 1];
        const z = vertices[3 * i + 2];

        // Calculate depth (view Z)
        // Note: For standard OpenGL/WebGL, -Z is forward. 
        // Larger Z = closer? Or usually standard view matrix transforms world to negative Z.
        // We want to sort from Farthest to Nearest (Back to Front) for Painter's Algorithm.
        // Higher depth value usually means "farther" in positive Z, but in view space, often -Z is forward.
        // Let's standardly sort by projected View Z.
        // If sorting max->min, we render far first.

        depths[i] = m[2] * x + m[6] * y + m[10] * z + m[14];
        indices[i] = i;
    }

    // 2. Sort Indices based on Depth
    // A standard `sort` is O(N log N).
    // Count Sort is O(N) but requires integer depths.

    // Efficient Counting Sort:
    // Determine bounds
    let minDepth = Infinity;
    let maxDepth = -Infinity;

    for (let i = 0; i < count; i++) {
        const d = depths[i];
        if (d < minDepth) minDepth = d;
        if (d > maxDepth) maxDepth = d;
    }

    const range = maxDepth - minDepth;

    // If range is tiny, just return (avoid div/0)
    if (range < 0.0001) {
        self.postMessage({ sortedIndices: indices }, [indices.buffer]);
        return;
    }

    // Bucket Count (16-bit sufficient usually)
    const totalBuckets = 65536;
    const counts = new Uint32Array(totalBuckets);
    const mappedIndices = new Uint32Array(count); // Store bucket ID per vertex

    // Pass 1: Count
    for (let i = 0; i < count; i++) {
        // Map depth to 0..bucketCount
        // Sort Back-to-Front (High Z to Low Z? or Low Z to High Z?)
        // Standard View Space: Camera at 0, looking down -Z. 
        // Z = -10 (Far), Z = -1 (Near).
        // Max Z is Near (-1), Min Z is Far (-10).
        // We want to render Far (-10) first, then Near (-1).
        // So we want Ascending Sort of signed Z (-10, -9, ..., -1).
        // OR Descending Sort of Distance (-distance).

        // Let's use simple normalized value 0..1
        // (depth - min) / range => 0 (MinDepth) to 1 (MaxDepth).

        // If we want MinDepth first:
        // bucket = normalized * totalBuckets

        // Invert the normalization to sort Descending (MaxDepth -> MinDepth) ?
        // Or Ascending?
        // Current symptom: Seeing back of head -> Near drawn first, Far drawn last (on top).
        // So we are sorting Near -> Far.
        // We want Far -> Near.
        // So we need to reverse the current sorting direction.

        let norm = (depths[i] - minDepth) / range;
        norm = 1.0 - norm; // Restore Inversion (Correct for this coordinate system)

        let bucket = Math.floor(norm * (totalBuckets - 1));

        counts[bucket]++;
        mappedIndices[i] = bucket;
    }

    // Prefix Sum
    let runCount = 0;
    for (let i = 0; i < totalBuckets; i++) {
        const c = counts[i];
        counts[i] = runCount;
        runCount += c;
    }

    // Pass 2: Reorder
    const sortedIndices = new Uint32Array(count);

    for (let i = 0; i < count; i++) {
        const bucket = mappedIndices[i];
        const destIdx = counts[bucket]++;
        sortedIndices[destIdx] = indices[i];
    }

    // Return
    self.postMessage({ sortedIndices, meshId: e.data.meshId }, [sortedIndices.buffer]);
};
