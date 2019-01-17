#version 450
/** This is a very unorthadox vertex sharder with a hardcoded triangle as a global being sent to output.
  * The reason for this is because creating buffers in Vulkan isn't exactly easy, and we want to just draw something. */

// This triangle is for Vulkan, so it is a right handed coordinate system (Positive z is into the screen, and up is
// negtaive for y).
// Normalized Device Coordinates/Clip Coordinates 2d triangle.
vec2 triangle_verticies[3] = vec2[](
    vec2(0.0, -0.5), // Top vertex
    vec2(0.5, 0.5), // Bottom Right Vertex
    vec2(0.0, -0.5) // Bottom left Vertex
);

void main() {
    // gl_Position is the built in to output the position of the vertex.
    // gl_VertexIndex is the built in to index into verticies in a buffer (or our global in this case).
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0); // z = 0, w = 1.0, built in divide does nothing.
}