import { useEffect, useRef, useState } from 'react';
import { clusterCenter, partitionBounds, partitionFor, type NetworkGraph } from './network';

type Props = {
  graph: NetworkGraph; seed: number; ranks: number; partitioned: boolean;
  selected: number; states?: string; disabled: boolean; onSelect: (node: number) => void;
};

export default function NetworkView(props: Props) {
  const host = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const live = useRef(props); live.current = props;
  const [motion, setMotion] = useState(true), [links, setLinks] = useState(true);
  const [failure, setFailure] = useState(''), [settled, setSettled] = useState(false);
  const [nodeInput, setNodeInput] = useState(''), [fullscreen, setFullscreen] = useState(false);
  const [viewRevision, setViewRevision] = useState(0);
  const settings = useRef({ motion, links }); settings.current = { motion, links };
  const workerRef = useRef<Worker | null>(null);
  const reset = useRef<() => void>(() => {});
  const focus = useRef<() => void>(() => {});

  useEffect(() => { workerRef.current?.postMessage({ kind: 'pause', paused: !motion }); }, [motion]);
  useEffect(() => {
    const update = () => setFullscreen(document.fullscreenElement === host.current);
    document.addEventListener('fullscreenchange', update);
    return () => document.removeEventListener('fullscreenchange', update);
  }, []);

  useEffect(() => {
    const element = canvas.current!, container = host.current!;
    const gl = element.getContext('webgl2', { antialias: true, alpha: false });
    if (!gl) { setFailure('This browser needs WebGL 2 enabled to display the 3D network.'); return; }
    setFailure(''); setSettled(false);
    const n = props.graph.totalNodes;
    let positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3), offsets = new Float32Array(n * 3), sizes = new Float32Array(n);
    const edgeIndices = Uint32Array.from(props.graph.edges);
    const shader = (type: number, source: string) => {
      const s = gl.createShader(type)!; gl.shaderSource(s, source); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || 'Shader compilation failed');
      return s;
    };
    let program: WebGLProgram;
    const shaders: WebGLShader[] = [];
    try {
      shaders.push(shader(gl.VERTEX_SHADER, `#version 300 es
        in vec3 position; in vec3 color; in vec3 cluster; in float size;
        uniform float yaw; uniform float pitch; uniform float distance; uniform float aspect;
        uniform float blend; uniform float pixelRatio; uniform bool lines; uniform vec3 target;
        out vec3 tint;
        void main(){
          vec3 p=position+cluster*blend-target;
          float x=cos(yaw)*p.x+sin(yaw)*p.z;
          float z=-sin(yaw)*p.x+cos(yaw)*p.z;
          float y=cos(pitch)*p.y-sin(pitch)*z;
          float depth=distance-(sin(pitch)*p.y+cos(pitch)*z);
          gl_Position=vec4(x*1.8/aspect,y*1.8,depth*0.999-0.2,depth);
          gl_PointSize=clamp(size*pixelRatio*300.0/max(1.0,depth),2.0*pixelRatio,18.0*pixelRatio);
          tint=lines?vec3(0.18,0.36,0.46):color;
        }`));
      shaders.push(shader(gl.FRAGMENT_SHADER, `#version 300 es
        precision mediump float; in vec3 tint; uniform bool lines; uniform float edgeOpacity; out vec4 outputColor;
        void main(){
          if(!lines){float r=length(gl_PointCoord-vec2(0.5));if(r>0.5)discard;
            outputColor=vec4(tint*(1.1-r*0.5),1.0);}
          else outputColor=vec4(tint,edgeOpacity);
        }`));
      program = gl.createProgram()!; shaders.forEach(s => gl.attachShader(program, s)); gl.linkProgram(program);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program) || 'Shader link failed');
    } catch (error) { setFailure(String(error)); shaders.forEach(s => gl.deleteShader(s)); return; }
    gl.useProgram(program);
    const buffers: WebGLBuffer[] = [];
    const attribute = (name: string, data: Float32Array, width: number) => {
      const b = gl.createBuffer()!; buffers.push(b); gl.bindBuffer(gl.ARRAY_BUFFER, b);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.DYNAMIC_DRAW);
      const loc = gl.getAttribLocation(program, name); gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, width, gl.FLOAT, false, 0, 0); return b;
    };
    const positionBuffer = attribute('position', positions, 3), colorBuffer = attribute('color', colors, 3);
    const offsetBuffer = attribute('cluster', offsets, 3), sizeBuffer = attribute('size', sizes, 1);
    const indexBuffer = gl.createBuffer()!; buffers.push(indexBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, edgeIndices, gl.STATIC_DRAW);
    const uniforms = Object.fromEntries(['yaw', 'pitch', 'distance', 'aspect', 'blend', 'pixelRatio', 'lines', 'target', 'edgeOpacity'].map(k => [k, gl.getUniformLocation(program, k)]));
    const upload = (buffer: WebGLBuffer, data: Float32Array) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    };
    let yaw = 0, pitch = 0.15, distance = 300, blend = 0;
    const target = [0, 0, 0];
    let previousStates: string | undefined, previousSelected = -2, previousRanks = 0;
    let previousPartitioned = false, frame = 0, disposed = false;
    let lastTime = performance.now(), width = 1, height = 1, pixelRatio = 1;
    const resize = new ResizeObserver(() => {
      width = container.clientWidth; height = element.clientHeight;
      pixelRatio = Math.min(2, window.devicePixelRatio || 1);
      element.width = Math.round(width * pixelRatio); element.height = Math.round(height * pixelRatio);
      gl.viewport(0, 0, element.width, element.height);
    }); resize.observe(container);
    const fit = () => { yaw = 0; pitch = 0.15; target.fill(0); distance = live.current.partitioned ? 300 + Math.ceil(Math.sqrt(live.current.ranks)) * 190 : 300; };
    reset.current = fit;
    focus.current = () => {
      const selected = live.current.selected;
      if (selected < 0) return;
      for (let axis = 0; axis < 3; axis++) target[axis] = positions[selected * 3 + axis] + offsets[selected * 3 + axis] * blend;
      distance = 140;
    };
    const worker = new Worker(new URL('./layout.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;
    worker.onmessage = event => {
      if (disposed) return;
      positions = event.data.positions; upload(positionBuffer, positions); setSettled(event.data.settled);
    };
    worker.onerror = () => setFailure('Network layout failed. Regenerate the network to retry.');
    const workerEdges = edgeIndices.slice();
    worker.postMessage({ kind: 'init', nodes: n, seed: props.seed, edges: workerEdges }, [workerEdges.buffer]);
    worker.postMessage({ kind: 'pause', paused: !settings.current.motion });

    function project(x: number, y: number, z: number) {
      x -= target[0]; y -= target[1]; z -= target[2];
      const rx = Math.cos(yaw) * x + Math.sin(yaw) * z, rz = -Math.sin(yaw) * x + Math.cos(yaw) * z;
      const ry = Math.cos(pitch) * y - Math.sin(pitch) * rz, depth = distance - (Math.sin(pitch) * y + Math.cos(pitch) * rz);
      return { x: width / 2 + rx * 0.9 * height / depth, y: height / 2 - ry * 0.9 * height / depth, depth };
    }
    const labels = Array.from({ length: 64 }, () => {
      const label = document.createElement('div'); label.className = 'cluster-label'; label.hidden = true;
      container.appendChild(label); return label;
    });
    const render = (time: number) => {
      if (disposed) return;
      const dt = Math.min(0.05, (time - lastTime) / 1000); lastTime = time;
      const p = live.current;
      if (p.partitioned !== previousPartitioned) { previousPartitioned = p.partitioned; fit(); }
      blend += ((p.partitioned ? 1 : 0) - blend) * Math.min(1, dt * 3);
      if (p.ranks !== previousRanks) {
        previousRanks = p.ranks;
        for (let i = 0; i < n; i++) offsets.set(clusterCenter(partitionFor(i, n, p.ranks), p.ranks), i * 3);
        upload(offsetBuffer, offsets);
      }
      if (p.states !== previousStates || p.selected !== previousSelected) {
        previousStates = p.states; previousSelected = p.selected;
        const palette = [[0.59, 0.69, 0.78], [0.25, 0.60, 1], [1, 0.32, 0.46]];
        for (let i = 0; i < n; i++) {
          const state = p.states ? p.states.charCodeAt(i) - 48 : i === p.selected ? 1 : 0;
          colors.set(i === p.selected ? [1, 0.86, 0.35] : palette[state] || palette[0], i * 3);
          sizes[i] = i === p.selected ? 12 : n > 20000 ? 2.4 : 4;
        }
        upload(colorBuffer, colors); upload(sizeBuffer, sizes);
      }
      gl.clearColor(0.025, 0.065, 0.10, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.uniform1f(uniforms.yaw, yaw); gl.uniform1f(uniforms.pitch, pitch);
      gl.uniform1f(uniforms.distance, distance); gl.uniform1f(uniforms.aspect, width / height);
      gl.uniform1f(uniforms.blend, blend); gl.uniform1f(uniforms.pixelRatio, pixelRatio);
      gl.uniform3f(uniforms.target, target[0], target[1], target[2]);
      gl.uniform1f(uniforms.edgeOpacity, n > 20000 ? 0.025 : 0.15);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.disable(gl.DEPTH_TEST);
      if (settings.current.links) {
        gl.uniform1i(uniforms.lines, 1); gl.drawElements(gl.LINES, edgeIndices.length, gl.UNSIGNED_INT, 0);
      }
      gl.enable(gl.DEPTH_TEST); gl.uniform1i(uniforms.lines, 0); gl.drawArrays(gl.POINTS, 0, n);
      if (p.selected >= 0) {
        gl.disable(gl.DEPTH_TEST); gl.drawArrays(gl.POINTS, p.selected, 1);
      }
      labels.forEach((label, rank) => {
        label.hidden = !p.partitioned || rank >= p.ranks;
        if (label.hidden) return;
        const center = clusterCenter(rank, p.ranks), pt = project(center[0] * blend, center[1] * blend + 106, center[2] * blend);
        const [start, end] = partitionBounds(n, p.ranks, rank);
        label.textContent = `C${rank + 1} · ${(end - start).toLocaleString()} nodes`;
        label.style.left = `${pt.x}px`; label.style.top = `${pt.y + element.offsetTop}px`;
        label.style.opacity = pt.depth > 0 ? '1' : '0';
      });
      frame = requestAnimationFrame(render);
    };
    frame = requestAnimationFrame(render);
    let pointer: { id: number; x: number; y: number; startX: number; startY: number; node: number; moved: boolean; pan: boolean } | null = null;
    const down = (event: PointerEvent) => {
      if (event.button !== 0 && event.button !== 2) return;
      const rect = element.getBoundingClientRect(), x = event.clientX - rect.left, y = event.clientY - rect.top;
      let node = -1, closest = 100, nearestDepth = Infinity;
      if (event.button === 0 && !event.shiftKey) for (let i = 0; i < n; i++) {
        const j = i * 3, pt = project(positions[j] + offsets[j] * blend, positions[j + 1] + offsets[j + 1] * blend, positions[j + 2] + offsets[j + 2] * blend);
        const delta = (x - pt.x) ** 2 + (y - pt.y) ** 2;
        if (pt.depth > 0 && delta < 100 && (delta < closest - 4 || Math.abs(delta - closest) < 4 && pt.depth < nearestDepth)) {
          closest = delta; nearestDepth = pt.depth; node = i;
        }
      }
      pointer = { id: event.pointerId, x: event.clientX, y: event.clientY, startX: event.clientX, startY: event.clientY, node, moved: false, pan: event.button === 2 || event.shiftKey };
      element.setPointerCapture(event.pointerId);
    };
    const move = (event: PointerEvent) => {
      if (!pointer || event.pointerId !== pointer.id) return;
      const dx = event.clientX - pointer.x, dy = event.clientY - pointer.y;
      pointer.moved ||= Math.hypot(event.clientX - pointer.startX, event.clientY - pointer.startY) > 4;
      if (pointer.node >= 0) {
        const j = pointer.node * 3;
        const pt = project(positions[j] + offsets[j] * blend, positions[j + 1] + offsets[j + 1] * blend, positions[j + 2] + offsets[j + 2] * blend);
        const sx = dx * pt.depth / (0.9 * height), sy = -dy * pt.depth / (0.9 * height);
        positions[j] += Math.cos(yaw) * sx + Math.sin(yaw) * Math.sin(pitch) * sy;
        positions[j + 1] += Math.cos(pitch) * sy;
        positions[j + 2] += Math.sin(yaw) * sx - Math.cos(yaw) * Math.sin(pitch) * sy;
        upload(positionBuffer, positions);
        worker.postMessage({ kind: 'drag', node: pointer.node, position: Array.from(positions.subarray(j, j + 3)) });
      } else if (pointer.pan) {
        const sx = -dx * distance / (0.9 * height), sy = dy * distance / (0.9 * height);
        target[0] += Math.cos(yaw) * sx + Math.sin(yaw) * Math.sin(pitch) * sy;
        target[1] += Math.cos(pitch) * sy;
        target[2] += Math.sin(yaw) * sx - Math.cos(yaw) * Math.sin(pitch) * sy;
      } else {
        yaw += dx * 0.006; pitch = Math.max(-1.45, Math.min(1.45, pitch + dy * 0.006));
      }
      pointer.x = event.clientX; pointer.y = event.clientY;
    };
    const up = (event: PointerEvent) => {
      if (!pointer || event.pointerId !== pointer.id) return;
      if (!pointer.moved && pointer.node >= 0 && !live.current.disabled) live.current.onSelect(pointer.node);
      worker.postMessage({ kind: 'release' }); pointer = null;
      if (element.hasPointerCapture(event.pointerId)) element.releasePointerCapture(event.pointerId);
    };
    const wheel = (event: WheelEvent) => { event.preventDefault(); distance = Math.max(115, Math.min(12000, distance * Math.exp(event.deltaY * 0.001))); };
    const context = (event: Event) => event.preventDefault();
    const cancel = () => { worker.postMessage({ kind: 'release' }); pointer = null; };
    const lost = () => setFailure('WebGL context was lost. Click Reload view to restore it.');
    element.addEventListener('pointerdown', down); element.addEventListener('pointermove', move);
    element.addEventListener('pointerup', up); element.addEventListener('pointercancel', cancel);
    element.addEventListener('wheel', wheel, { passive: false }); element.addEventListener('contextmenu', context);
    element.addEventListener('webglcontextlost', lost);
    return () => {
      disposed = true; cancelAnimationFrame(frame); resize.disconnect(); worker.terminate(); workerRef.current = null;
      labels.forEach(label => label.remove()); buffers.forEach(b => gl.deleteBuffer(b)); shaders.forEach(s => gl.deleteShader(s)); gl.deleteProgram(program);
      element.removeEventListener('pointerdown', down); element.removeEventListener('pointermove', move);
      element.removeEventListener('pointerup', up); element.removeEventListener('pointercancel', cancel);
      element.removeEventListener('wheel', wheel); element.removeEventListener('contextmenu', context); element.removeEventListener('webglcontextlost', lost);
    };
  }, [props.graph, props.seed, viewRevision]);

  async function toggleFullscreen() {
    try {
      if (document.fullscreenElement) await document.exitFullscreen();
      else await host.current?.requestFullscreen();
    } catch { setFailure('This embedded window blocks fullscreen. Hide parameters to expand the simulation workspace.'); }
  }
  const selectById = () => {
    const node = Number(nodeInput);
    if (nodeInput.trim() && Number.isInteger(node) && node >= 0 && node < props.graph.totalNodes) props.onSelect(node);
  };
  return <div className="network-stage" ref={host}>
    <div className="network-toolbar">
      <div><strong>{props.graph.totalNodes.toLocaleString()} nodes</strong><span> · {props.graph.edgeCount.toLocaleString()} connections · {settled ? 'Layout settled' : motion ? 'Layout flowing' : 'Layout paused'}</span></div>
      <div className="view-actions">
        <button onClick={() => setMotion(!motion)} aria-pressed={!motion}>{motion ? 'Pause motion' : 'Resume motion'}</button>
        <button onClick={() => setLinks(!links)} aria-pressed={links}>{links ? 'Hide connections' : 'Show connections'}</button>
        <button onClick={() => reset.current()}>Fit network</button>
        <button disabled={props.selected < 0} onClick={() => focus.current()}>Focus source</button>
        <button onClick={toggleFullscreen}>{fullscreen ? 'Exit fullscreen' : 'Fullscreen'}</button>
      </div>
    </div>
    <canvas ref={canvas} aria-label={`Interactive 3D network with all ${props.graph.totalNodes} nodes. Drag background to rotate, scroll to zoom, drag a node to move it.`} />
    <div className="network-caption">
      <span>Drag background to rotate · Shift-drag to pan · Scroll to zoom · Drag nodes to move · Click to select source</span>
      <form onSubmit={e => { e.preventDefault(); selectById(); }}>
        <label htmlFor="source-id">Source ID</label><input id="source-id" type="number" min={0} max={props.graph.totalNodes - 1} value={nodeInput} onChange={e => setNodeInput(e.target.value)} disabled={props.disabled} placeholder="Node ID" />
        <button disabled={props.disabled}>Select</button><span>{props.selected >= 0 ? `Selected: ${props.selected}` : 'None selected'}</span>
      </form>
    </div>
    {failure && <div className="view-error" role="alert">{failure}<button onClick={() => setViewRevision(v => v + 1)}>Reload view</button></div>}
  </div>;
}
