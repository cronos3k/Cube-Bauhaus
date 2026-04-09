#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec2 in_uv;
layout(location = 3) in vec4 in_color;

layout(set = 0, binding = 0) uniform Camera {
    mat4  view_proj;
    vec4  camera_pos;
    vec4  sun_dir;
    vec4  sun_color;
    vec4  ambient;
} cam;

layout(push_constant) uniform Push {
    mat4 model;
} push;

layout(location = 0) out vec4 frag_color;

void main() {
    vec4 world_pos = push.model * vec4(in_position, 1.0);
    gl_Position    = cam.view_proj * world_pos;
    frag_color     = in_color;
}
