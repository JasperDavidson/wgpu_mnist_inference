struct Uniforms {
  mat_dim: vec4<u32>
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read> mat1: array<f32>;
@group(0) @binding(2) var<storage, read> mat2: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_mat: array<f32>;

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let row = global_id.x;
  let col = global_id.y;

  let M = uniforms.mat_dim.x;
  let K = uniforms.mat_dim.y;
  let N = uniforms.mat_dim.z;

  if row >= M || col >= N {
    return;
  }

  var sum = 0.0;
  for (var k = u32(0); k < K; k++) {
    sum += mat1[row * K + k] * mat2[k * N + col];
  }

  output_mat[row * N + col] = sum;
}
