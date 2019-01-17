#version 450
#extension GL_ARB_seperate_shader_objects : enable

// FrameBuffer 0, output a color.
layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(1.0, 0.0, 0.0, 1.0); // Output Red, opaque.
}