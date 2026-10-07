import { GpuLayout } from './gpuLayout';
import { useEffect, useRef, useState } from 'react';
import { clusterCenter, partitionBounds, partitionFor, type NetworkGraph } from './network';

type Props = {
  graph: NetworkGraph; seed: number; ranks: number; partitioned: boolean;
  selected: number; states?: string | Uint8Array; disabled: boolean; onSelect: (node: number) => void;
};

export default function NetworkView(props: Props) {
  const host = useRef<HTMLDivElement>(null), canvas = useRef<HTMLCanvasElement>(null);
  const live = useRef(props); live.current = props;
  const [layoutDevice, setLayoutDevice] = useState('CPU worker');
  const [allLinks, setAllLinks] = useState(false);
  const [motion, setMotion] = useState(true), [links, setLinks] = useState(true);
  const [failure, setFailure] = useState(''), [settled, setSettled] = useState(false);
  const [nodeInput, setNodeInput] = useState(''), [fullscreen, setFullscreen] = useState(false);
  const [viewRevision, setViewRevision] = useState(0);
  const settings = useRef({ motion, links, allLinks }); settings.current = { motion, links, allLinks };
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
    const gl = element.getContext('webgl2', { antialias: props.graph.totalNodes < 100000, alpha: false });
    if (!gl) { setFailure('This browser needs WebGL 2 enabled to display the 3D network.'); return; }
    setFailure(''); setSettled(false);
    const n = props.graph.totalNodes;
    let positions = new Float32Array(n * 3);
    const nodeStates = new Uint8Array(n);
    let gpu: GpuLayout | undefined;
    let gpuWasPaused = false;
    const overviewLimit = 100000;
    const overviewCount = Math.min(props.graph.edgeCount, overviewLimit);
    const edgeIndices = props.graph.edges instanceof Uint32Array ? props.graph.edges : Uint32Array.from(props.graph.edges);
    const shader = (type: number, source: string) => {
      const s = gl.createShader(type)!; gl.shaderSource(s, source); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) || 'Shader compilation failed');
      return s;
    };
    let program: WebGLProgram;
    const shaders: WebGLShader[] = [];
    try {
      shaders.push(shader(gl.VERTEX_SHADER, `#version 300 es
        in vec3 position; in float state;
        uniform int nodes; uniform int ranks; uniform int selected; uniform bool picking;
        flat out uint nodeId;
        uniform float yaw; uniform float pitch; uniform float distance; uniform float aspect;
        uniform float blend; uniform float pixelRatio; uniform bool lines; uniform vec3 target;
        out vec3 tint;
        void main(){
          int rank=min(ranks-1,((gl_VertexID+1)*ranks-1)/nodes);
          int cols=int(ceil(sqrt(float(ranks)))), rows=(ranks+cols-1)/cols;
          vec3 cluster=ranks==1?vec3(0):vec3((float(rank%cols)-float(cols-1)*0.5)*230.,(float(rank/cols)-float(rows-1)*0.5)*230.,0);
          vec3 p=position+cluster*blend-target;
          float size=gl_VertexID==selected?12.:(nodes>20000?1.2:4.);
          vec3 color=state<0.5?vec3(.59,.69,.78):(state<1.5?vec3(.25,.60,1):vec3(1,.32,.46));
          if(gl_VertexID==selected) color=vec3(1,.86,.35);
          nodeId=uint(gl_VertexID+1);
          float x=cos(yaw)*p.x+sin(yaw)*p.z;
          float z=-sin(yaw)*p.x+cos(yaw)*p.z;
          float y=cos(pitch)*p.y-sin(pitch)*z;
          float depth=distance-(sin(pitch)*p.y+cos(pitch)*z);
          gl_Position=vec4(x*1.8/aspect,y*1.8,depth*0.999-0.2,depth);
          gl_PointSize=clamp(size*pixelRatio*300.0/max(1.0,depth),1.0*pixelRatio,18.0*pixelRatio);
          if(picking) gl_PointSize=max(gl_PointSize,8.0*pixelRatio);
          tint=lines?vec3(0.18,0.36,0.46):color;
        }`));
      shaders.push(shader(gl.FRAGMENT_SHADER, `#version 300 es
        precision highp float; precision highp int; flat in uint nodeId; uniform bool picking; in vec3 tint; uniform bool lines; uniform float edgeOpacity; out vec4 outputColor;
        void main(){
          if(picking){outputColor=vec4(float(nodeId&255u),float((nodeId>>8)&255u),float((nodeId>>16)&255u),255.)/255.;return;}
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
    const positionBuffer = attribute('position', positions, 3);
    const stateBuffer = gl.createBuffer()!; buffers.push(stateBuffer); gl.bindBuffer(gl.ARRAY_BUFFER, stateBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, nodeStates, gl.DYNAMIC_DRAW);
    const stateLoc = gl.getAttribLocation(program, 'state'); gl.enableVertexAttribArray(stateLoc); gl.vertexAttribPointer(stateLoc, 1, gl.UNSIGNED_BYTE, false, 0, 0);
    const indexBuffer = gl.createBuffer()!; buffers.push(indexBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, indexBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, edgeIndices, gl.STATIC_DRAW);
    const overviewEdges = new Uint32Array(overviewCount * 2);
    for (let e = 0; e < overviewCount; e++) {
      const source = Math.floor(e * props.graph.edgeCount / overviewCount) * 2;
      overviewEdges[e * 2] = edgeIndices[source]; overviewEdges[e * 2 + 1] = edgeIndices[source + 1];
    }
    const overviewBuffer = gl.createBuffer()!; buffers.push(overviewBuffer);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, overviewBuffer); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, overviewEdges, gl.STATIC_DRAW);
    const pickingBuffer = gl.createFramebuffer()!, pickingTexture = gl.createTexture()!, pickingDepth = gl.createRenderbuffer()!;
    const uniforms = Object.fromEntries(['yaw', 'pitch', 'distance', 'aspect', 'blend', 'pixelRatio', 'lines', 'target', 'edgeOpacity', 'nodes', 'ranks', 'selected', 'picking'].map(k => [k, gl.getUniformLocation(program, k)]));
    const upload = (buffer: WebGLBuffer, data: Float32Array) => {
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, data);
    };
    let yaw = 0, pitch = 0.15, distance = 300, blend = 0;
    const target = [0, 0, 0];
    let previousStates: string | Uint8Array | undefined, previousSelected = -2;
    let previousPartitioned = false, frame = 0, disposed = false;
    let lastDraw = 0;
    let lastTime = performance.now(), width = 1, height = 1, pixelRatio = 1;
    const resize = new ResizeObserver(() => {
      width = container.clientWidth; height = element.clientHeight;
      pixelRatio = Math.min(n >= 100000 ? 1 : 2, window.devicePixelRatio || 1);
      element.width = Math.round(width * pixelRatio); element.height = Math.round(height * pixelRatio);
      gl.viewport(0, 0, element.width, element.height);
      gl.bindTexture(gl.TEXTURE_2D, pickingTexture); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, element.width, element.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.bindRenderbuffer(gl.RENDERBUFFER, pickingDepth); gl.renderbufferStorage(gl.RENDERBUFFER, gl.DEPTH_COMPONENT16, element.width, element.height);
      gl.bindFramebuffer(gl.FRAMEBUFFER, pickingBuffer); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, pickingTexture, 0);
      gl.framebufferRenderbuffer(gl.FRAMEBUFFER, gl.DEPTH_ATTACHMENT, gl.RENDERBUFFER, pickingDepth); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }); resize.observe(container);
    const fit = () => { yaw = 0; pitch = 0.15; target.fill(0); distance = live.current.partitioned ? 300 + Math.ceil(Math.sqrt(live.current.ranks)) * 190 : 300; };
    reset.current = fit;
    focus.current = () => {
      const selected = live.current.selected;
      if (selected < 0) return;
      const value = gpu ? gpu.position(selected) : positions.subarray(selected * 3, selected * 3 + 3);
      const offset = clusterCenter(partitionFor(selected, n, live.current.ranks), live.current.ranks);
      for (let axis = 0; axis < 3; axis++) target[axis] = value[axis] + offset[axis] * blend;
      distance = 140;
    };
    const worker = new Worker(new URL('./layout.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker; setLayoutDevice(n >= 20000 ? 'Preparing GPU layout' : 'CPU worker');
    worker.onmessage = event => {
      if (disposed) return;
      positions = event.data.positions; upload(positionBuffer, positions);
      if (event.data.neighbours) {
        try { gpu = new GpuLayout(gl, n, positions, event.data.neighbours); setLayoutDevice('Browser GPU · approximate springs'); }
        catch (error) { setFailure(`GPU layout could not start: ${String(error)}`); }
        worker.terminate(); workerRef.current = null;
      }
      setSettled(event.data.settled);
    };
    worker.onerror = () => setFailure('Network layout failed. Regenerate the network to retry.');
    const workerEdges = edgeIndices.slice();
    worker.postMessage({ kind: 'init', gpu: n >= 20000, nodes: n, seed: props.seed, edges: workerEdges }, [workerEdges.buffer]);
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
      if (n >= 100000 && time - lastDraw < 32) { frame = requestAnimationFrame(render); return; }
      lastDraw = time;
      const dt = Math.min(0.05, (time - lastTime) / 1000); lastTime = time;
      const p = live.current;
      if (p.partitioned !== previousPartitioned) { previousPartitioned = p.partitioned; fit(); }
      blend += ((p.partitioned ? 1 : 0) - blend) * Math.min(1, dt * 3);
      if (gpu) {
        if (settings.current.motion && gpuWasPaused) gpu.resume();
        gpuWasPaused = !settings.current.motion;
        // One pass at most per displayed frame; camera interaction takes priority.
        if (settings.current.motion && !pointer) gpu.step();
        if (gpu.settled) setSettled(true);
      }
      gl.useProgram(program); gl.bindVertexArray(null);
      gl.bindBuffer(gl.ARRAY_BUFFER, gpu?.buffer || positionBuffer);
      const positionLoc = gl.getAttribLocation(program, 'position'); gl.vertexAttribPointer(positionLoc, 3, gl.FLOAT, false, 0, 0);
      if (p.states !== previousStates || (!p.states && p.selected !== previousSelected)) {
        previousStates = p.states;
        if (p.states instanceof Uint8Array) nodeStates.set(p.states);
        else if (p.states) { for (let i = 0; i < n; i++) nodeStates[i] = Math.max(0, p.states.charCodeAt(i) - 48); }
        else { nodeStates.fill(0); if (p.selected >= 0) nodeStates[p.selected] = 1; }
        gl.bindBuffer(gl.ARRAY_BUFFER, stateBuffer); gl.bufferSubData(gl.ARRAY_BUFFER, 0, nodeStates);
      }
      previousSelected = p.selected;
      gl.uniform1i(uniforms.nodes, n); gl.uniform1i(uniforms.ranks, p.ranks); gl.uniform1i(uniforms.selected, p.selected); gl.uniform1i(uniforms.picking, 0);
      gl.clearColor(0.025, 0.065, 0.10, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.uniform1f(uniforms.yaw, yaw); gl.uniform1f(uniforms.pitch, pitch);
      gl.uniform1f(uniforms.distance, distance); gl.uniform1f(uniforms.aspect, width / height);
      gl.uniform1f(uniforms.blend, blend); gl.uniform1f(uniforms.pixelRatio, pixelRatio);
      gl.uniform3f(uniforms.target, target[0], target[1], target[2]);
      gl.uniform1f(uniforms.edgeOpacity, n > 20000 ? 0.025 : 0.15);
      gl.enable(gl.BLEND); gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.disable(gl.DEPTH_TEST);
      if (settings.current.links) {
        gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, settings.current.allLinks ? indexBuffer : overviewBuffer);
        gl.uniform1i(uniforms.lines, 1); gl.drawElements(gl.LINES, settings.current.allLinks ? edgeIndices.length : overviewEdges.length, gl.UNSIGNED_INT, 0);
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
        const text = `C${rank + 1} · ${(end - start).toLocaleString()} nodes`;
        if (label.textContent !== text) label.textContent = text;
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
      let node = -1;
      if (event.button === 0 && !event.shiftKey) {
        gl.bindFramebuffer(gl.FRAMEBUFFER, pickingBuffer); gl.disable(gl.BLEND); gl.enable(gl.DEPTH_TEST);
        gl.clearColor(0, 0, 0, 0); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
        gl.uniform1i(uniforms.lines, 0); gl.uniform1i(uniforms.picking, 1); gl.drawArrays(gl.POINTS, 0, n);
        const pixel = new Uint8Array(4);
        gl.readPixels(Math.max(0, Math.min(element.width - 1, Math.floor(x * pixelRatio))), Math.max(0, Math.min(element.height - 1, Math.floor((height - y) * pixelRatio))), 1, 1, gl.RGBA, gl.UNSIGNED_BYTE, pixel);
        node = pixel[0] + pixel[1] * 256 + pixel[2] * 65536 - 1;
        if (node >= n) node = -1;
        gl.uniform1i(uniforms.picking, 0); gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        if (node >= 0 && gpu) positions.set(gpu.position(node), node * 3);
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
        const offset = clusterCenter(partitionFor(pointer.node, n, live.current.ranks), live.current.ranks);
        const pt = project(positions[j] + offset[0] * blend, positions[j + 1] + offset[1] * blend, positions[j + 2] + offset[2] * blend);
        const sx = dx * pt.depth / (0.9 * height), sy = -dy * pt.depth / (0.9 * height);
        positions[j] += Math.cos(yaw) * sx + Math.sin(yaw) * Math.sin(pitch) * sy;
        positions[j + 1] += Math.cos(pitch) * sy;
        positions[j + 2] += Math.sin(yaw) * sx - Math.cos(yaw) * Math.sin(pitch) * sy;
        const value = positions.subarray(j, j + 3);
        if (gpu) gpu.drag(pointer.node, value);
        else { gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer); gl.bufferSubData(gl.ARRAY_BUFFER, j * 4, value); }
        if (!gpu) worker.postMessage({ kind: 'drag', node: pointer.node, position: Array.from(positions.subarray(j, j + 3)) });
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
      gpu?.release(); if (!gpu) worker.postMessage({ kind: 'release' }); pointer = null;
      if (element.hasPointerCapture(event.pointerId)) element.releasePointerCapture(event.pointerId);
    };
    const wheel = (event: WheelEvent) => { event.preventDefault(); distance = Math.max(115, Math.min(12000, distance * Math.exp(event.deltaY * 0.001))); };
    const context = (event: Event) => event.preventDefault();
    const cancel = () => { gpu?.release(); if (!gpu) worker.postMessage({ kind: 'release' }); pointer = null; };
    const lost = () => setFailure('WebGL context was lost. Click Reload view to restore it.');
    element.addEventListener('pointerdown', down); element.addEventListener('pointermove', move);
    element.addEventListener('pointerup', up); element.addEventListener('pointercancel', cancel);
    element.addEventListener('wheel', wheel, { passive: false }); element.addEventListener('contextmenu', context);
    element.addEventListener('webglcontextlost', lost);
    return () => {
      disposed = true; cancelAnimationFrame(frame); resize.disconnect(); worker.terminate(); workerRef.current = null; gpu?.dispose();
      gl.deleteFramebuffer(pickingBuffer); gl.deleteTexture(pickingTexture); gl.deleteRenderbuffer(pickingDepth);
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
      <div><strong>{props.graph.totalNodes.toLocaleString()} nodes</strong><span> · {props.graph.edgeCount.toLocaleString()} connections · {layoutDevice} · {settled ? 'Layout settled' : motion ? 'Layout flowing' : 'Layout paused'}</span></div>
      <div className="view-actions">
        <button onClick={() => setMotion(!motion)} aria-pressed={!motion}>{motion ? 'Pause motion' : 'Resume motion'}</button>
        <button onClick={() => setLinks(!links)} aria-pressed={links}>{links ? 'Hide connections' : 'Show connections'}</button>
        {props.graph.edgeCount > 100000 && <button onClick={() => setAllLinks(!allLinks)} aria-pressed={allLinks}>{allLinks ? 'Use overview connections' : 'Draw all connections (slower)'}</button>}
        <button onClick={() => reset.current()}>Fit network</button>
        <button disabled={props.selected < 0} onClick={() => focus.current()}>Focus source</button>
        <button onClick={toggleFullscreen}>{fullscreen ? 'Exit fullscreen' : 'Fullscreen'}</button>
      </div>
    </div>
    <canvas ref={canvas} aria-label={`Interactive 3D network with all ${props.graph.totalNodes} nodes. Drag background to rotate, scroll to zoom, drag a node to move it.`} />
    <div className="network-caption">
      <span>{props.graph.edgeCount > 100000 && !allLinks ? `Overview: 100,000 of ${props.graph.edgeCount.toLocaleString()} connections · All nodes shown · ` : ''}Drag background to rotate · Shift-drag to pan · Scroll to zoom · Drag nodes to move · Click to select source</span>
      <form onSubmit={e => { e.preventDefault(); selectById(); }}>
        <label htmlFor="source-id">Source ID</label><input id="source-id" type="number" min={0} max={props.graph.totalNodes - 1} value={nodeInput} onChange={e => setNodeInput(e.target.value)} disabled={props.disabled} placeholder="Node ID" />
        <button disabled={props.disabled}>Select</button><span>{props.selected >= 0 ? `Selected: ${props.selected}` : 'None selected'}</span>
      </form>
    </div>
    {failure && <div className="view-error" role="alert">{failure}<button onClick={() => setViewRevision(v => v + 1)}>Reload view</button></div>}
  </div>;
}
