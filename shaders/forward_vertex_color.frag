#version 450
#extension GL_ARB_seperate_shader_objects : enable

layout(location = 0) out vec4 outColor;

// Inputs are automatically interpolated.
layout(location = 0) in vec3 fragColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}
