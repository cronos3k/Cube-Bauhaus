#version 450

layout(location = 0) in vec2 frag_uv;
layout(location = 1) in vec4 frag_color;

layout(set = 0, binding = 0) uniform sampler2D font_tex;

layout(location = 0) out vec4 out_color;

void main() {
    // Font texture is uploaded as R8G8B8A8_UNORM (alpha-only data is expanded
    // to RGBA during upload).  Simple multiply gives correct blending.
    out_color = frag_color * texture(font_tex, frag_uv);
}
