struct Uniforms {
  bounds: vec4<u32>
}

@group(0) @binding(0) var<storage, read> image: array<f32>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;

@group(0) @binding(2) var<storage, read> w1: array<f32>;
@group(0) @binding(3) var<storage, read> b1: array<f32>;
@group(0) @binding(4) var<storage, read> w2: array<f32>;
@group(0) @binding(5) var<storage, read> b2: array<f32>;

fn is_outside_bounds(coord: vec3<u32>, bounds: vec3<f32>) -> bool {
  return coord.x >= u32(bounds.x) || coord.y >= u32(bounds.y) || coord.z >= u32(bounds.z);
}

@compute @workgroup_size(64)
fn compute_probability(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.y * (uniforms.bounds.x) + global_id.x;
}
