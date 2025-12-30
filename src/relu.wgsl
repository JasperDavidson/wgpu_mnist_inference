struct Uniforms {
  bounds: vec4<u32>
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var<storage, read_write> buffer: array<f32>;

fn is_outside_bounds(coord: vec3<u32>, bounds: vec3<u32>) -> bool {
  return coord.x > bounds.x || coord.y > bounds.y || coord.z > bounds.z;
}

@compute @workgroup_size(64)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
  if (is_outside_bounds(global_id, uniforms.bounds.xyz)) {
    return;
  }

  let idx = global_id.y * uniforms.bounds.x + global_id.x;

  if buffer[idx] < 0 {
    buffer[idx] = 0;
  }
}
