/**
 * GPU-Accelerated Blur using WebGL (Gaussian blur only, masked regions)
 */

export interface GPUBlurConfig {
  kernelSize: number
}

export class GPUBlur {
  private gl: WebGLRenderingContext
  private canvas: HTMLCanvasElement
  private program: WebGLProgram | null = null
  private vertexBuffer: WebGLBuffer | null = null
  private texture: WebGLTexture | null = null
  private config: GPUBlurConfig

  private vertexShaderSource = `
    attribute vec2 a_position;
    attribute vec2 a_texCoord;
    varying vec2 v_texCoord;

    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
      v_texCoord = a_texCoord;
    }
  `

  private fragmentShaderSource = `
    precision mediump float;

    uniform sampler2D u_texture;
    uniform sampler2D u_mask;
    uniform vec2 u_textureSize;
    uniform float u_kernelSize;
    varying vec2 v_texCoord;

    #define MAX_KERNEL_SIZE 32

    vec4 gaussianBlur(sampler2D texture, vec2 coord, vec2 texelStep, float kernelSize) {
      vec4 color = vec4(0.0);
      float total = 0.0;
      for (int i = -MAX_KERNEL_SIZE; i <= MAX_KERNEL_SIZE; i++) {
        if (abs(float(i)) > kernelSize) continue;
        float weight = exp(-(float(i) * float(i)) / (2.0 * (kernelSize * 0.6) * (kernelSize * 0.6)));
        color += texture2D(texture, coord + texelStep * float(i)) * weight;
        total += weight;
      }
      return color / total;
    }

    void main() {
      float maskValue = texture2D(u_mask, v_texCoord).r;
      vec4 originalColor = texture2D(u_texture, v_texCoord);

      if (maskValue > 0.5) {
        vec2 texelStepX = vec2(1.0 / u_textureSize.x, 0.0);
        vec2 texelStepY = vec2(0.0, 1.0 / u_textureSize.y);
        vec4 horizontal = gaussianBlur(u_texture, v_texCoord, texelStepX, u_kernelSize);
        vec4 vertical = gaussianBlur(u_texture, v_texCoord, texelStepY, u_kernelSize);
        gl_FragColor = (horizontal + vertical) * 0.5;
      } else {
        gl_FragColor = originalColor;
      }
    }
  `

  constructor(config: GPUBlurConfig) {
    this.config = config
    this.canvas = document.createElement('canvas')
    const gl = this.canvas.getContext('webgl') || this.canvas.getContext('experimental-webgl')
    if (!gl) throw new Error('WebGL not supported')
    this.gl = gl
    this.initializeWebGL()
  }

  private initializeWebGL() {
    const gl = this.gl
    const vertexShader = this.createShader(gl.VERTEX_SHADER, this.vertexShaderSource)
    const fragmentShader = this.createShader(gl.FRAGMENT_SHADER, this.fragmentShaderSource)
    if (!vertexShader || !fragmentShader) throw new Error('Shader creation failed')

    this.program = gl.createProgram()!
    gl.attachShader(this.program, vertexShader)
    gl.attachShader(this.program, fragmentShader)
    gl.linkProgram(this.program)
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) {
      throw new Error('Program link failed: ' + gl.getProgramInfoLog(this.program))
    }

    this.vertexBuffer = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer)
    const vertices = new Float32Array([
      -1, -1, 0, 0,
       1, -1, 1, 0,
      -1,  1, 0, 1,
      -1,  1, 0, 1,
       1, -1, 1, 0,
       1,  1, 1, 1
    ])
    gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW)

    this.texture = gl.createTexture()
    gl.bindTexture(gl.TEXTURE_2D, this.texture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)

    console.log('[GPUBlur] WebGL initialized')
  }

  private createShader(type: number, source: string): WebGLShader | null {
    const gl = this.gl
    const shader = gl.createShader(type)!
    gl.shaderSource(shader, source)
    gl.compileShader(shader)
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader error:', gl.getShaderInfoLog(shader))
      gl.deleteShader(shader)
      return null
    }
    return shader
  }

  private createMaskTexture(width: number, height: number, regions: { rectangles: number[][], polygons: number[][][] }) {
    const maskCanvas = document.createElement('canvas')
    maskCanvas.width = width
    maskCanvas.height = height
    const ctx = maskCanvas.getContext('2d')!
    ctx.fillStyle = 'black'
    ctx.fillRect(0, 0, width, height)
    ctx.fillStyle = 'white'
    for (const r of regions.rectangles) ctx.fillRect(r[0], r[1], r[2], r[3])
    for (const poly of regions.polygons) {
      ctx.beginPath()
      ctx.moveTo(poly[0][0], poly[0][1])
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1])
      ctx.closePath()
      ctx.fill()
    }
    const gl = this.gl
    const maskTexture = gl.createTexture()!
    gl.bindTexture(gl.TEXTURE_2D, maskTexture)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST)
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, maskCanvas)
    return maskTexture
  }

  public applyBlurToCanvas(sourceCanvas: HTMLCanvasElement, regions: { rectangles: number[][], polygons: number[][][] }): HTMLCanvasElement {
    const gl = this.gl
    this.canvas.width = sourceCanvas.width
    this.canvas.height = sourceCanvas.height
    gl.viewport(0, 0, sourceCanvas.width, sourceCanvas.height)

    const maskTexture = this.createMaskTexture(sourceCanvas.width, sourceCanvas.height, regions)

    gl.useProgram(this.program!)
    gl.activeTexture(gl.TEXTURE0)
    gl.bindTexture(gl.TEXTURE_2D, this.texture)
    gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true)
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, sourceCanvas)
    gl.activeTexture(gl.TEXTURE1)
    gl.bindTexture(gl.TEXTURE_2D, maskTexture)
    gl.uniform1i(gl.getUniformLocation(this.program!, 'u_mask'), 1)

    const positionLoc = gl.getAttribLocation(this.program!, 'a_position')
    const texCoordLoc = gl.getAttribLocation(this.program!, 'a_texCoord')
    gl.bindBuffer(gl.ARRAY_BUFFER, this.vertexBuffer)
    gl.enableVertexAttribArray(positionLoc)
    gl.vertexAttribPointer(positionLoc, 2, gl.FLOAT, false, 16, 0)
    gl.enableVertexAttribArray(texCoordLoc)
    gl.vertexAttribPointer(texCoordLoc, 2, gl.FLOAT, false, 16, 8)

    gl.uniform2f(gl.getUniformLocation(this.program!, 'u_textureSize'), sourceCanvas.width, sourceCanvas.height)
    gl.uniform1f(gl.getUniformLocation(this.program!, 'u_kernelSize'), this.config.kernelSize)

    gl.drawArrays(gl.TRIANGLES, 0, 6)

    return this.canvas
  }
}
