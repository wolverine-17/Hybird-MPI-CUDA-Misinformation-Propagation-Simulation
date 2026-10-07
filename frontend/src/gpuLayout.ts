// GPU-resident, bounded-neighbour spring layout for large graphs. The neighbour
// sample affects only the picture; the simulation always uses the complete graph.
export class GpuLayout {
  private program: WebGLProgram;
  private buffers: WebGLBuffer[];
  private texture: WebGLTexture;
  private neighbours: WebGLTexture;
  private vao: WebGLVertexArrayObject;
  private feedback: WebGLTransformFeedback;
  private side = 0;
  private iteration = 0;
  private pinned = -1;
  private width: number;
  private height: number;
  constructor(private gl: WebGL2RenderingContext, private count: number, positions: Float32Array, neighbours: Uint32Array) {
    this.width = Math.min(4096, gl.getParameter(gl.MAX_TEXTURE_SIZE));
    this.height = Math.ceil(count * 3 / this.width);
    const neighbourHeight = Math.ceil(count * 2 / this.width);
    if (Math.max(this.height, neighbourHeight) > gl.getParameter(gl.MAX_TEXTURE_SIZE)) throw new Error('Graph exceeds GPU texture capacity');
    const shaders = [gl.createShader(gl.VERTEX_SHADER)!, gl.createShader(gl.FRAGMENT_SHADER)!];
    gl.shaderSource(shaders[0], `#version 300 es
      precision highp float; precision highp int;
      in vec3 position; uniform sampler2D points; uniform highp usampler2D neighbours;
      uniform int width; uniform int count; uniform int pinned; uniform float cooling;
      out vec3 nextPosition;
      vec3 point(int id){int j=id*3; return vec3(texelFetch(points,ivec2(j%width,j/width),0).r,
        texelFetch(points,ivec2((j+1)%width,(j+1)/width),0).r,texelFetch(points,ivec2((j+2)%width,(j+2)/width),0).r);}
      void main(){
        int id=gl_VertexID; vec3 force=vec3(0); float used=0.;
        for(int block=0;block<2;block++){
          int j=id*2+block; uvec4 ids=texelFetch(neighbours,ivec2(j%width,j/width),0);
          for(int k=0;k<4;k++) if(ids[k]<uint(count)){
            vec3 delta=point(int(ids[k]))-position; float d=max(length(delta),0.01);
            force+=delta*((d-32.)/d); used+=1.;
          }
        }
        float radius=max(length(position),0.01);
        force=force/max(used,1.)*0.018 + position/radius*(72.-radius)*0.008;
        nextPosition=id==pinned?position:position+clamp(force*cooling,vec3(-0.6),vec3(0.6));
        gl_Position=vec4(0); gl_PointSize=1.;
      }`);
    gl.shaderSource(shaders[1], '#version 300 es\nprecision mediump float; out vec4 c; void main(){c=vec4(0);}');
    this.program = gl.createProgram()!;
    for (const s of shaders) { gl.compileShader(s); gl.attachShader(this.program, s); }
    gl.transformFeedbackVaryings(this.program, ['nextPosition'], gl.SEPARATE_ATTRIBS);
    gl.linkProgram(this.program);
    const valid = gl.getProgramParameter(this.program, gl.LINK_STATUS);
    const log = gl.getProgramInfoLog(this.program);
    shaders.forEach(s => gl.deleteShader(s));
    if (!valid) { gl.deleteProgram(this.program); throw new Error(log || 'GPU layout shader failed'); }
    const padded = new Float32Array(this.width * this.height); padded.set(positions);
    this.buffers = [0, 1].map(() => {
      const b = gl.createBuffer()!; gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, padded, gl.DYNAMIC_COPY); return b;
    });
    this.texture = gl.createTexture()!; gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.R32F, this.width, this.height);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.neighbours = gl.createTexture()!; gl.bindTexture(gl.TEXTURE_2D, this.neighbours);
    const paddedIds = new Uint32Array(this.width * neighbourHeight * 4); paddedIds.fill(count); paddedIds.set(neighbours);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA32UI, this.width, neighbourHeight);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.width, neighbourHeight, gl.RGBA_INTEGER, gl.UNSIGNED_INT, paddedIds);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    this.vao = gl.createVertexArray()!; this.feedback = gl.createTransformFeedback()!;
  }
  get buffer() { return this.buffers[this.side]; }
  get settled() { return this.iteration >= 240; }
  step() {
    if (this.settled) return;
    const gl = this.gl;
    gl.useProgram(this.program); gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    const loc = gl.getAttribLocation(this.program, 'position'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this.texture);
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, this.buffer);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, this.width, this.height, gl.RED, gl.FLOAT, 0);
    gl.bindBuffer(gl.PIXEL_UNPACK_BUFFER, null);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.neighbours);
    for (const [key, value] of Object.entries({ points: 0, neighbours: 1, width: this.width, count: this.count, pinned: this.pinned }))
      gl.uniform1i(gl.getUniformLocation(this.program, key), value);
    gl.uniform1f(gl.getUniformLocation(this.program, 'cooling'), Math.max(0.12, 1 - this.iteration / 240));
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.feedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.buffers[1 - this.side]);
    gl.enable(gl.RASTERIZER_DISCARD); gl.beginTransformFeedback(gl.POINTS); gl.drawArrays(gl.POINTS, 0, this.count); gl.endTransformFeedback(); gl.disable(gl.RASTERIZER_DISCARD);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null); gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null); gl.bindVertexArray(null);
    this.side = 1 - this.side; this.iteration++;
  }
  position(node: number) {
    const value = new Float32Array(3); this.gl.bindBuffer(this.gl.ARRAY_BUFFER, this.buffer);
    this.gl.getBufferSubData(this.gl.ARRAY_BUFFER, node * 12, value); return value;
  }
  drag(node: number, value: Float32Array) {
    const gl = this.gl; this.pinned = node; this.iteration = Math.min(160, this.iteration);
    for (const b of this.buffers) { gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferSubData(gl.ARRAY_BUFFER, node * 12, value); }
  }
  release() { this.pinned = -1; }
  resume() { this.iteration = Math.min(160, this.iteration); }
  dispose() {
    const gl = this.gl; this.buffers.forEach(b => gl.deleteBuffer(b)); gl.deleteTexture(this.texture); gl.deleteTexture(this.neighbours);
    gl.deleteVertexArray(this.vao); gl.deleteTransformFeedback(this.feedback); gl.deleteProgram(this.program);
  }
}
