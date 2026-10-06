// Linear-cost spring layout and spatial-cell repulsion; no all-pairs force loop.
let positions = new Float32Array(0), velocity = new Float32Array(0);
let edges = new Uint32Array(0), degree = new Uint32Array(0);
let pinned = -1, paused = false, iteration = 0, timer: ReturnType<typeof setTimeout> | undefined;
let lastSent = 0;

function schedule() {
  if (timer !== undefined || paused || iteration >= 240) return;
  timer = setTimeout(step, 25);
}

function step() {
  timer = undefined;
  if (paused) return;
  const n = degree.length, cellSize = 18;
  const cells = new Map<number, [number, number, number, number]>();
  const key = (x: number, y: number, z: number) =>
    (Math.floor(x / cellSize) + 32) * 4096 + (Math.floor(y / cellSize) + 32) * 64 + Math.floor(z / cellSize) + 32;
  for (let i = 0; i < n; i++) {
    const j = i * 3, k = key(positions[j], positions[j + 1], positions[j + 2]);
    const c = cells.get(k);
    if (c) { c[0] += positions[j]; c[1] += positions[j + 1]; c[2] += positions[j + 2]; c[3]++; }
    else cells.set(k, [positions[j], positions[j + 1], positions[j + 2], 1]);
  }
  const cooling = Math.max(0.12, 1 - iteration / 240);
  for (let e = 0; e < edges.length; e += 2) {
    const a = edges[e], b = edges[e + 1], ai = a * 3, bi = b * 3;
    const dx = positions[bi] - positions[ai], dy = positions[bi + 1] - positions[ai + 1], dz = positions[bi + 2] - positions[ai + 2];
    const distance = Math.hypot(dx, dy, dz) || 1;
    const force = (distance - 32) / distance * 0.025 * cooling;
    for (let axis = 0; axis < 3; axis++) {
      const delta = (positions[bi + axis] - positions[ai + axis]) * force;
      velocity[ai + axis] += delta / Math.max(1, degree[a]);
      velocity[bi + axis] -= delta / Math.max(1, degree[b]);
    }
  }
  for (let i = 0; i < n; i++) {
    const j = i * 3, c = cells.get(key(positions[j], positions[j + 1], positions[j + 2]))!;
    const length = Math.hypot(positions[j], positions[j + 1], positions[j + 2]) || 1;
    for (let axis = 0; axis < 3; axis++) {
      const away = positions[j + axis] - c[axis] / c[3];
      const radial = positions[j + axis] / length * (72 - length) * 0.006;
      velocity[j + axis] = (velocity[j + axis] + (away * 0.014 + radial) * cooling) * 0.82;
      if (i !== pinned) positions[j + axis] += Math.max(-1, Math.min(1, velocity[j + axis]));
      else velocity[j + axis] = 0;
    }
  }
  iteration++;
  if (performance.now() - lastSent > 100 || iteration === 240) {
    const copy = positions.slice();
    self.postMessage({ positions: copy, settled: iteration >= 240 }, { transfer: [copy.buffer] });
    lastSent = performance.now();
  }
  schedule();
}

self.onmessage = (event: MessageEvent) => {
  const data = event.data;
  if (data.kind === 'init') {
    edges = data.edges; degree = new Uint32Array(data.nodes);
    positions = new Float32Array(data.nodes * 3); velocity = new Float32Array(positions.length);
    let seed = data.seed >>> 0;
    const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 4294967296; };
    for (let i = 0; i < data.nodes; i++) {
      const z = random() * 2 - 1, angle = random() * Math.PI * 2, r = 35 + 55 * Math.cbrt(random());
      positions[i * 3] = Math.sqrt(1 - z * z) * Math.cos(angle) * r;
      positions[i * 3 + 1] = Math.sqrt(1 - z * z) * Math.sin(angle) * r;
      positions[i * 3 + 2] = z * r;
    }
    for (const node of edges) degree[node]++;
    const copy = positions.slice();
    self.postMessage({ positions: copy, settled: false }, { transfer: [copy.buffer] });
    iteration = 0; schedule();
  } else if (data.kind === 'drag') {
    pinned = data.node;
    positions.set(data.position, pinned * 3);
    iteration = Math.min(iteration, 160); schedule();
  } else if (data.kind === 'release') {
    pinned = -1;
  } else if (data.kind === 'pause') {
    paused = data.paused;
    if (!paused && iteration >= 240) iteration = 160;
    schedule();
  }
};
