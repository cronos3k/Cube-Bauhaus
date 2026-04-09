#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;  // R8G8B8A8_UNORM, auto-normalized by GPU

layout(push_constant) uniform Push {
    vec2 screen_size;
} push;

layout(location = 0) out vec2 frag_uv;
layout(location = 1) out vec4 frag_color;

void main() {
    frag_uv    = in_uv;
    frag_color = in_color;

    // Convert pixel coordinates to Vulkan NDC [-1, 1].
    // Vulkan clip space: Y points down (top = -1, bottom = +1),
    // which matches egui's top-left origin — no flip needed.
    gl_Position = vec4(
        2.0 * in_pos.x / push.screen_size.x - 1.0,
        2.0 * in_pos.y / push.screen_size.y - 1.0,
        0.0,
        1.0
    );
}
