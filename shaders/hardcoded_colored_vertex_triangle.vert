#version 450

vec2 triangle_verticies[3] = vec2[](
    vec2(0.0, -0.5), // Top vertex
    vec2(0.5, 0.5), // Bottom Right Vertex
    vec2(0.0, -0.5) // Bottom left Vertex
);
vec3 colors[3] = vec3[](
    vec3(1.0, 0.0, 0.0), // Red
    vec3(0.0, 1.0, 0.0), // Green
    vec3(0.0, 0.0, 1.0)  // Blue
);

layout(location = 0) out vec3 fragColor;

void main() {
    // gl_Position is the built in to output the position of the vertex.
    // gl_VertexIndex is the built in to index into verticies in a buffer (or our global in this case).
    gl_Position = vec4(triangle_verticies[gl_VertexIndex], 0.0, 1.0); // z = 0, w = 1.0, built in divide does nothing.
    fragColor = colors[gl_VertexIndex];
}
