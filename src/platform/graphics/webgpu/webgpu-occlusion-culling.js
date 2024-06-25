import { SHADERLANGUAGE_WGSL } from "../constants";
import { Shader } from "../shader";
import { ShaderProcessor } from "../shader-processor";
import { WebgpuShader } from "./webgpu-shader";

export class AdvancedWebGPUOcclusionCulling {
  constructor(device) {
    this.device = device;
    this.visibleLastFrame = new Set();
    this.allObjects = [];
    this.initialised = false;
  }

  async initialise() {
    if (this.initialised) return;

    this.pipelineLayout = this.device.createPipelineLayout({
      bindGroupLayouts: [
        this.device.createBindGroupLayout({
          entries: [
            { binding: 0, visibility: GPUShaderStage.VERTEX | GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'depth' } }
          ]
        })
      ]
    });
    this.firstPassPipeline = await this.createFirstPassPipeline();
    this.hizGenPipeline = await this.createHizGenPipeline();
    this.occlusionCullPipeline = await this.createOcclusionCullPipeline();
    this.mainRenderPipeline = await this.createMainRenderPipeline();

    this.createBuffers();

    this.initialized = true;
  }
  async createFirstPassPipeline() {
    const shaderProcessor = new ShaderProcessor();
    const vertexShader = /*wgsl*/`
        struct Uniforms {
            viewProjectionMatrix : mat4x4<f32>;
        };
        @group(0) @binding(0) var<uniform> uniforms : Uniforms;

        struct VertexInput {
            @location(0) position : vec3<f32>;
        };

        struct VertexOutput {
            @builtin(position) position : vec4<f32>;
        };

        @vertex
        fn main(input : VertexInput) -> VertexOutput {
            var output : VertexOutput;
            output.position = uniforms.viewProjectionMatrix * vec4<f32>(input.position, 1.0);
            return output;
        }
    `;
    const fragmentShader = /*wgsl*/`
        @fragment
        fn main() -> &location(0) vec4<f32> {
            return vec4<f32>(0.0, 0.0, 0.0, 1.0);
        }
    `;
    const shader = new WebgpuShader(new Shader(this.device, {
      name: 'WebGPUResolverDepthShader',
      shaderLanguage: SHADERLANGUAGE_WGSL,
      vshader: vertexShader,
      fshader: fragmentShader
    }));




    return await this.device.createRenderPipeline({
      layout: this.pipelineLayout,
      vertex: {
        module: shader.getVertexShaderModule(),
        entryPoint: shader.vertexEntryPoint,
        buffers: [
          {
            arrayStride: 12,
            attributes: [{ shaderLocation: 0, offset: 0, format: 'float32x3' }]
          }
        ]
      },
      fragment: {
        module: shader.getFragmentShaderModule(),
        entryPoint: shader.fragmentEntryPoint,
        targets: [{ format: 'rgba8unorm' }]
      },
      depthStencil: {
        depthWriteEnabled: true,
        depthCompare: 'less',
        format: 'depth32float'
      }
    });
  }

  async createHizGenPipeline() {
    const shaderCode = /*wgsl*/`
        struct HiZBuffer {
            data : array<f32>;
        };
        group(0) @binding(0) var<storage, read_write> hizBuffer : HiZBuffer;
        group(0) @binding(1) var depthTexture : texture_depth_2d;

      @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
            let dims = textureDimensions(depthTexture);
            let x = global_id.x * 2;
            let y = global_id.y * 2;
            
            if (x >= dims.x || y >= dims.y) {
                return;
            }
            
            var maxDepth = 0.0;
            for (var i = 0u; i < 2u; i = i + 1u) {
                for (var j = 0u; j < 2u; j = j + 1u) {
                    let depth = textureLoad(depthTexture, vec2<i32>(x + i, y + j), 0);
                    maxDepth = max(maxDepth, depth);
                }
            }
            
            let index = global_id.y * (dims.x / 2) + global_id.x;
            hizBuffer.data[index] = maxDepth;
        }
    `;
    const shader = new WebgpuShader(new Shader(this.device, {
      name: 'HZBShader',
      shaderLanguage: SHADERLANGUAGE_WGSL,
      cshader: shaderCode,
    }));
    return await this.device.wgpu.createComputePipeline({
        layout: this.pipelineLayout,
        compute: {
            module: shader.getComputeShaderModule(),
            entryPoint: shader.computeEntryPoint
        }
    });
  }
}
