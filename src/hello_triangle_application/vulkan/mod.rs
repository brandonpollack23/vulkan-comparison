mod initialization_helpers;
mod vulkan_context;

pub const TITLE_BYTES: &'static [u8] = b"Vulkan";

pub mod vulkan_structures;

pub use self::vulkan_context::ColoredVertex;
pub use self::vulkan_context::VulkanContext;
