#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_color;

layout(set = 0, binding = 0) uniform Camera {
    mat4  view_proj;
    vec4  camera_pos;
    vec4  sun_dir;      // xyz = world-space direction toward light, w = intensity
    vec4  sun_color;    // rgb = color
    vec4  ambient;      // rgb = color, w = strength
} cam;

layout(push_constant) uniform Push {
    mat4 model;
} push;

layout(location = 0) out vec3 frag_normal_ws;
layout(location = 1) out vec4 frag_color;
layout(location = 2) out vec3 frag_pos_ws;
layout(location = 3) out vec2 frag_uv;

void main() {
    vec4 world_pos    = push.model * vec4(in_position, 1.0);
    gl_Position       = cam.view_proj * world_pos;
    frag_normal_ws    = normalize(mat3(push.model) * in_normal);
    frag_color        = in_color;  // rgb = tint, a = texture layer index
    frag_pos_ws       = world_pos.xyz;
    frag_uv           = in_uv;
}
