use std::fs::File;
use std::io::Read;
use std::sync::mpsc::channel;

use anyhow::Ok;
use wgpu::PipelineLayoutDescriptor;
use wgpu::util::DeviceExt;

const IMAGE_SIZE: u32 = 28 * 28;
const HIDDEN_SIZE: u32 = 128;
const OUTPUT_SIZE: u8 = 10;

fn bytes_to_f32(bytes: &[u8]) -> f32 {
    assert!(bytes.len() == 4, "f32 has a size of 4 bytes");
    return f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
}

fn parse_mnist_parameters() -> anyhow::Result<Vec<f32>> {
    let mut file = File::open("training/mnist_mlp.bin")?;
    let mut byte_buffer = Vec::new();
    file.read_to_end(&mut byte_buffer)?;

    let mut float_buffer = Vec::with_capacity(byte_buffer.len());
    for i in (0..byte_buffer.len()).step_by(4) {
        float_buffer.push(bytes_to_f32(&byte_buffer[i..(i + 4)]));
    }

    Ok(float_buffer)
}

fn parse_inf_image() -> anyhow::Result<Vec<f32>> {
    let mut file = File::open("training/mnist_inf.bin")?;
    let mut byte_buffer = Vec::new();
    file.read_to_end(&mut byte_buffer)?;

    let mut float_buffer = Vec::with_capacity(byte_buffer.len());
    for i in (0..byte_buffer.len()).step_by(4) {
        float_buffer.push(bytes_to_f32(&byte_buffer[i..(i + 4)]));
    }

    Ok(float_buffer)
}

struct Tensor {
    width: u32,
    height: u32,
    data: Vec<f32>,
}

impl Tensor {
    fn new_cap(width: u32, height: u32, capacity: u32) -> Self {
        Self {
            width,
            height,
            data: Vec::with_capacity(capacity as usize),
        }
    }

    fn new_init(width: u32, height: u32, data: &[f32]) -> Self {
        Self {
            width,
            height,
            data: data.to_vec(),
        }
    }

    fn at(&self, x: u32, y: u32) -> Option<&f32> {
        self.data.get((y * self.width + x) as usize)
    }

    fn matmul(&self, other: &Self) -> Tensor {
        assert!(self.width == other.height, "Matmul dimension mismatch");

        let mut result_mat = Tensor::new_cap(other.width, self.height, self.width * other.height);
        for y in 0..self.height {
            for x in 0..other.width {
                let mut sum = 0.0;

                // self.height and other.width are equivalent
                for k in 0..self.width {
                    let a_val = self.at(k, y).unwrap();
                    let b_val = other.at(x, k).unwrap();
                    sum += a_val * b_val;
                }

                result_mat.data.push(sum);
            }
        }

        result_mat
    }

    fn matadd(&self, other: &Self) -> Tensor {
        assert!(
            self.height == other.height,
            "Matadd dimension mismatch: height"
        );
        assert!(
            self.width == other.width,
            "Matadd dimension mismatch: width"
        );

        let mut result_mat = Tensor::new_cap(self.width, self.height, self.width * self.height);
        for i in 0..self.data.len() {
            let sum = self.data[i] + other.data[i];
            result_mat.data.push(sum);
        }

        result_mat
    }

    fn relu(&mut self) {
        for i in 0..self.data.len() {
            if self.data[i] < 0.0 {
                self.data[i] = 0.0;
            }
        }
    }

    fn softmax(&mut self) {
        let mut exponentials = Vec::with_capacity(self.data.len());
        for val in self.data.iter() {
            exponentials.push(val.exp());
        }
        let exponentials_sum: f32 = exponentials.iter().sum();

        for i in 0..self.data.len() {
            self.data[i] = exponentials[i] / exponentials_sum;
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    bounds: [u32; 4],
}

fn fetch_mnist_data() -> anyhow::Result<[Tensor; 5]> {
    // fetch the weights and biases
    let parameters = parse_mnist_parameters()?;

    let w1_idx = (IMAGE_SIZE * HIDDEN_SIZE) as usize;
    let b1_idx = w1_idx + HIDDEN_SIZE as usize;
    let w2_idx = b1_idx + (HIDDEN_SIZE * OUTPUT_SIZE as u32) as usize;

    let weight_1 = Tensor::new_init(
        IMAGE_SIZE as u32,
        HIDDEN_SIZE as u32,
        &parameters[0..w1_idx],
    );
    let bias_1 = Tensor::new_init(1, HIDDEN_SIZE as u32, &parameters[w1_idx..b1_idx]);
    let weight_2 = Tensor::new_init(
        HIDDEN_SIZE as u32,
        OUTPUT_SIZE as u32,
        &parameters[b1_idx..w2_idx],
    );
    let bias_2 = Tensor::new_init(1, OUTPUT_SIZE as u32, &parameters[w2_idx..]);

    // fetch an image
    let image = parse_inf_image()?;
    let image_tensor = Tensor::new_init(1, IMAGE_SIZE, &image);

    Ok([weight_1, bias_1, weight_2, bias_2, image_tensor])
}

fn cpu_mnist() -> anyhow::Result<()> {
    let [weight_1, bias_1, weight_2, bias_2, image_tensor] = fetch_mnist_data()?;

    // perform the mlp operations
    let mut hidden_layer_outputs = (weight_1.matmul(&image_tensor)).matadd(&bias_1);
    hidden_layer_outputs.relu();
    let mut output_layer = (weight_2.matmul(&hidden_layer_outputs)).matadd(&bias_2);
    // output_layer.softmax();

    println!("cpu output layer: {:?}", output_layer.data);
    Ok(())
}

#[pollster::main]
async fn main() -> anyhow::Result<()> {
    cpu_mnist()?;

    // fetch wgpu devices
    let instance = wgpu::Instance::new(&Default::default());
    let adapter = instance.request_adapter(&Default::default()).await.unwrap();
    let (device, queue) = adapter.request_device(&Default::default()).await.unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());

    // initialize the shaders
    let matrix_add_shader = device.create_shader_module(wgpu::include_wgsl!("matrix_add.wgsl"));
    let matrix_mul_shader = device.create_shader_module(wgpu::include_wgsl!("matrix_mul.wgsl"));
    let relu_shader = device.create_shader_module(wgpu::include_wgsl!("relu.wgsl"));

    // create the generic bind layouts
    let matrix_binary_operation_bind_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            // .into() converts &'static str to Cow<'static, str>
            label: Some("binary op layout").map(Into::into),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3, // Fixed the duplicate '2' from earlier!
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let matrix_inplace_operation_bind_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("inplace op layout").map(Into::into),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    let matrix_unary_operation_bind_layout =
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("unary op layout").map(Into::into),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

    // create the pipelines
    let matrix_add_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Matrix addition"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("matrix add pipeline layout").map(Into::into),
            bind_group_layouts: &[&matrix_unary_operation_bind_layout],
            immediate_size: 0,
        })),
        module: &matrix_add_shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });
    let matrix_mul_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Matrix multiplication"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("matrix mul pipeline layout").map(Into::into),
            bind_group_layouts: &[&matrix_binary_operation_bind_layout],
            immediate_size: 0,
        })),
        module: &matrix_mul_shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });
    let relu_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Rectified Linear activation"),
        layout: Some(&device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("relu layot").map(Into::into),
            bind_group_layouts: &[&matrix_inplace_operation_bind_layout],
            immediate_size: 0,
        })),
        module: &relu_shader,
        entry_point: None,
        compilation_options: Default::default(),
        cache: Default::default(),
    });

    // initialize the buffers
    let [weight_1, bias_1, weight_2, bias_2, image_tensor] = fetch_mnist_data()?;
    let weight_1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Layer one weight buffer"),
        contents: bytemuck::cast_slice(&weight_1.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });
    let weight_2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Layer two weight buffer"),
        contents: bytemuck::cast_slice(&weight_2.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });
    let bias_1_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Layer one bias buffer"),
        contents: bytemuck::cast_slice(&bias_1.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });
    let bias_2_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Layer two bias buffer"),
        contents: bytemuck::cast_slice(&bias_2.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });
    let image_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Image value buffer"),
        contents: bytemuck::cast_slice(&image_tensor.data),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
    });
    let hidden_layer_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Hidden layer output buffer"),
        size: (HIDDEN_SIZE * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let output_layer_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Output layer output buffer"),
        size: (OUTPUT_SIZE as u32 * std::mem::size_of::<f32>() as u32) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // metadata for layer 1: [rows, inner dim, columns, _]
    let uniform_1 = Uniforms {
        bounds: [HIDDEN_SIZE as u32, IMAGE_SIZE as u32, 1, 0],
    };
    let uniform_buffer_1 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Layer 1 Uniforms"),
        contents: bytemuck::cast_slice(&[uniform_1]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // metadata for layer 2: [rows, inner dim, columns, _]
    let uniform_2 = Uniforms {
        bounds: [OUTPUT_SIZE as u32, HIDDEN_SIZE as u32, 1, 0],
    };
    let uniform_buffer_2 = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Layer 2 Uniforms"),
        contents: bytemuck::cast_slice(&[uniform_2]),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });

    // BindGroup for (weight_1 * image_tensor) -> hidden_layer_buffer

    let matmul_1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul 1 bind group"),
        layout: &matrix_binary_operation_bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer_1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight_1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: image_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: hidden_layer_buffer.as_entire_binding(),
            },
        ],
    });

    // BindGroup for: (hidden_layer_buffer + bias_1) -> hidden_layer_buffer (In-place Add)
    let matadd_1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matadd 1 bind group"),
        layout: &matrix_unary_operation_bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer_1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bias_1_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: hidden_layer_buffer.as_entire_binding(),
            },
        ],
    });

    // BindGroup for: relu(hidden_layer_buffer)
    let relu_1_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("relu 1 bind group"),
        layout: &matrix_inplace_operation_bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer_1.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: hidden_layer_buffer.as_entire_binding(),
            },
        ],
    });

    // BindGroup for: hidden_layer_buffer * weight_2_buffer -> output_layer_buffer
    let matmul_2_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matmul 2 bind grouo"),
        layout: &matrix_binary_operation_bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: weight_2_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: hidden_layer_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: output_layer_buffer.as_entire_binding(),
            },
        ],
    });

    // BindGroup for: output_layer_buffer + bias_2_buffer
    let matadd_2_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("matadd 2 bind group"),
        layout: &matrix_unary_operation_bind_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer_2.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: bias_2_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: output_layer_buffer.as_entire_binding(),
            },
        ],
    });

    // BindGroup for: softmax(output_layer_buffer)

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MLP Hidden Layer Pass"),
            ..Default::default()
        });

        // layer 1 matmul
        pass.set_pipeline(&matrix_mul_pipeline);
        pass.set_bind_group(0, &matmul_1_bind_group, &[]);
        // Workgroups: Output rows, Output cols, 1
        let wg_rows = u32::div_ceil(HIDDEN_SIZE, 16) as u32;
        pass.dispatch_workgroups(wg_rows, 1, 1);

        // later 1 matadd
        let wg_num = u32::div_ceil(HIDDEN_SIZE, 64) as u32;

        pass.set_pipeline(&matrix_add_pipeline);
        pass.set_bind_group(0, &matadd_1_bind_group, &[]);
        pass.dispatch_workgroups(wg_num, 1, 1);

        // layer 1 relu
        pass.set_pipeline(&relu_pipeline);
        pass.set_bind_group(0, &relu_1_bind_group, &[]);
        pass.dispatch_workgroups(wg_num, 1, 1);
    }

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("MLP Output Layer Pass"),
            ..Default::default()
        });

        // layer 2 matmul
        pass.set_pipeline(&matrix_mul_pipeline);
        pass.set_bind_group(0, &matmul_2_bind_group, &[]);
        // Workgroups: Output rows, Output cols, 1
        let wg_rows = u32::div_ceil(HIDDEN_SIZE, 16) as u32;
        pass.dispatch_workgroups(wg_rows, 1, 1);

        // later 2 matadd
        let wg_num = u32::div_ceil(OUTPUT_SIZE as u32, 64) as u32;

        pass.set_pipeline(&matrix_add_pipeline);
        pass.set_bind_group(0, &matadd_2_bind_group, &[]);
        pass.dispatch_workgroups(wg_num, 1, 1);

        // layer 2 softmax (todo later)
    }

    let temp_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Temp buffer"),
        size: output_layer_buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    encoder.copy_buffer_to_buffer(
        &output_layer_buffer,
        0,
        &temp_buffer,
        0,
        output_layer_buffer.size(),
    );
    queue.submit([encoder.finish()]);

    {
        let (tx, rx) = channel();

        temp_buffer.map_async(wgpu::MapMode::Read, .., move |result| {
            tx.send(result).unwrap()
        });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        rx.recv()??;

        let output_data = temp_buffer.get_mapped_range(..);
        let num_data: &[f32] = bytemuck::cast_slice(&output_data);
        println!("GPU result: {:?}", num_data);
    }

    temp_buffer.unmap();

    Ok(())
}
