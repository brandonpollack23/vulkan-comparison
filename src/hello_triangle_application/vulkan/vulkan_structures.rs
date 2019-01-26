use ash::version::DeviceV1_0;
use ash::vk;
use ash::Device;

pub struct VulkanVertexBuffer {
  pub vertex_buffer: vk::Buffer,
  pub buffer_indices: Vec<usize>,
  pub memory_indices: Vec<usize>,
}
