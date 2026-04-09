#version 450

layout(location = 0) in vec3 frag_normal_ws;
layout(location = 1) in vec4 frag_color;
layout(location = 2) in vec3 frag_pos_ws;
layout(location = 3) in vec2 frag_uv;

layout(set = 0, binding = 0) uniform Camera {
    mat4  view_proj;
    vec4  camera_pos;
    vec4  sun_dir;
    vec4  sun_color;
    vec4  ambient;
} cam;

layout(set = 0, binding = 1) uniform sampler2DArray tex_array;

layout(location = 0) out vec4 out_color;

void main() {
    vec3 n = normalize(frag_normal_ws);
    vec3 l = normalize(cam.sun_dir.xyz);

    // Texture layer index is packed in color.a
    float layer = frag_color.a;
    vec3 base_color;

    if (layer >= 0.0) {
        // Sample from texture array
        vec3 tex_coord = vec3(frag_uv, layer);
        vec3 tex_color = texture(tex_array, tex_coord).rgb;
        // Apply VSlot color tint (rgb)
        base_color = tex_color * frag_color.rgb;
    } else {
        // No texture loaded — use vertex color with checkerboard
        float checker = mod(floor(frag_uv.x) + floor(frag_uv.y), 2.0);
        float pattern = mix(0.85, 1.0, checker);
        base_color = frag_color.rgb * pattern;
    }

    // Lambert diffuse
    float ndotl   = max(dot(n, l), 0.0);
    vec3  diffuse = base_color * cam.sun_color.rgb * ndotl * cam.sun_dir.w;

    // Ambient
    vec3  ambient = base_color * cam.ambient.rgb * cam.ambient.w;

    // Subtle back-face dim so concave geometry reads well
    float back    = max(dot(-n, l), 0.0) * 0.08;
    vec3  back_c  = base_color * back;

    out_color = vec4(diffuse + ambient + back_c, 1.0);
}
