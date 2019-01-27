use ash::vk;

pub struct VulkanVertexBuffer {
  pub vertex_buffer: vk::Buffer,
  pub index_buffer: Option<vk::Buffer>,
  pub buffer_indices: Vec<usize>,
  pub memory_indices: Vec<usize>,
}
