use crate::hello_triangle_application::raw_vulkan_helpers;
use crate::hello_triangle_application::vulkan::initialization_helpers::*;
use ash::{extensions, version::DeviceV1_0, version::InstanceV1_0, vk, Device, Entry, Instance};
use winit::Window;

pub struct VulkanContext {
  inner: VulkanContextInner,
}

pub struct VulkanContextInner {
  pub entry: Entry, // Function loader.
  pub instance: Instance,
  pub callback_structures: Option<CallbackStructures>,
  pub surface_structures: VulkanSurfaceStructures,
  pub logical_device: Device,
  pub graphics_queue: vk::Queue,
  pub present_queue: vk::Queue,
  pub swap_chain_structures: VulkanSwapChainStructures,
  pub swap_chain_images: Vec<vk::Image>,
  pub image_views: Vec<vk::ImageView>,
  pub pipeline_structures: VulkanPipelineStructures,
  pub swap_chain_framebuffers: Vec<vk::Framebuffer>,
  pub command_structures: VulkanCommandStructures,
  pub synchronization: VulkanSynchronization,
  pub in_flight_frame_index: usize,
}

// Each extension is wrapped in it's own struct with it's created structures to
// allow their functions to be accessed more easily in Drop, and logically group
// them together.
pub struct CallbackStructures {
  pub debug_utils_extension: extensions::ext::DebugUtils,
  pub debug_callback_structure: vk::DebugUtilsMessengerEXT,
}

pub struct VulkanSurfaceStructures {
  pub surface_extension: extensions::khr::Surface,
  pub surface: vk::SurfaceKHR,
}

pub struct VulkanSwapChainStructures {
  pub swap_chain_extension: extensions::khr::Swapchain,
  pub swap_chain: vk::SwapchainKHR,
  pub swap_chain_image_format: vk::Format,
  pub swap_chain_extent: vk::Extent2D,
}

pub struct VulkanPipelineStructures {
  pub render_pass: vk::RenderPass,
  pub pipeline_layout: vk::PipelineLayout,
  pub graphics_pipeline: vk::Pipeline,
}

pub struct VulkanCommandStructures {
  pub command_pool: vk::CommandPool,
  pub command_buffers: Vec<vk::CommandBuffer>,
}

pub struct VulkanSynchronization {
  pub image_available_sems: Vec<vk::Semaphore>,
  pub render_finished_sems: Vec<vk::Semaphore>,
  pub in_flight_fences: Vec<vk::Fence>,
}

// Have to manually deallocate (vkDestroyInstance, destroy debug utils) etc.
// Physical Devices are cleaned up when the instance is destroyed, so no need to
// do that manually. Logical devices, however, are not, since they're not part
// of the instance.
// Rule of thumb, anything create was called for, destroy is called for,
// everything else was already "part" of some other created thing (eg queues,
// physical devices).
impl Drop for VulkanContextInner {
  fn drop(&mut self) {
    unsafe {
      self
        .synchronization
        .render_finished_sems
        .iter()
        .for_each(|&sem| {
          self.logical_device.destroy_semaphore(sem, None);
        });
      self
        .synchronization
        .image_available_sems
        .iter()
        .for_each(|&sem| {
          self.logical_device.destroy_semaphore(sem, None);
        });
      self
        .synchronization
        .in_flight_fences
        .iter()
        .for_each(|&fence| {
          self.logical_device.destroy_fence(fence, None);
        });

      self
        .logical_device
        .destroy_command_pool(self.command_structures.command_pool, None);

      for framebuffer in self.swap_chain_framebuffers.iter() {
        self.logical_device.destroy_framebuffer(*framebuffer, None);
      }

      self
        .logical_device
        .destroy_pipeline(self.pipeline_structures.graphics_pipeline, None);

      self
        .logical_device
        .destroy_render_pass(self.pipeline_structures.render_pass, None);

      self
        .logical_device
        .destroy_pipeline_layout(self.pipeline_structures.pipeline_layout, None);

      self.image_views.iter().for_each(|image_view| {
        self.logical_device.destroy_image_view(*image_view, None);
      });

      self
        .swap_chain_structures
        .swap_chain_extension
        .destroy_swapchain(self.swap_chain_structures.swap_chain, None);

      self
        .surface_structures
        .surface_extension
        .destroy_surface(self.surface_structures.surface, None);

      self.logical_device.destroy_device(None);

      // Destroy debug extension.
      if let Some(callback_structures) = self.callback_structures.as_mut() {
        callback_structures
          .debug_utils_extension
          .destroy_debug_utils_messenger(callback_structures.debug_callback_structure, None);
      }

      // Destroy Vulkan instance.
      self.instance.destroy_instance(None);
    }
  }
}

impl VulkanContext {
  pub fn initialize_vulkan(window: &Window) -> VulkanContext {
    // Entry is the vulkan library loader, it loads the vulkan shared object
    // (dll/so/etc), and then loads all the function pointers for vulkan versions
    // 1.0 and 1.1 from that.
    let entry: Entry = Entry::new().expect("Unable to load Vulkan dll and functions!");

    // Create the Vulkan instance (ie instantiate the client side driver).
    let instance = create_instance(&entry);

    // Set up debug callback, so we can get messages through the Vulkan runtime (via
    // Rust FFI).
    let callback_structures = setup_debug_callback(&entry, &instance);

    // Surface is actually created before physical device selection because it can
    // influence it.
    let surface_structures = create_surface_structures(window, &entry, &instance);

    // For now just use the first one, who cares right?  In the future a scoring
    // system could be used, or a user selection, but anything works atm.
    let physical_device =
      get_physical_devices_for_surface_drawing(&instance, &surface_structures)[0];

    // Right now no features are requested, but this will be different by the end of
    // the tutorial. Later this will include things like vertex shader, geometry
    // shader, etc.
    let (logical_device, queue_family_indices) = create_logical_device(
      &instance,
      &physical_device,
      &surface_structures,
      &vk::PhysicalDeviceFeatures::builder().build(),
    );

    // Get the queues that were created with the logical device.
    let graphics_queue = unsafe {
      logical_device.get_device_queue(queue_family_indices.graphics_queue_family.unwrap(), 0u32)
    };
    let present_queue = unsafe {
      logical_device.get_device_queue(queue_family_indices.present_queue_family.unwrap(), 0u32)
    };

    // Load swap chain extension and create all the structures to use it.
    let swap_chain_structures = create_swap_chain_structures(
      &instance,
      &surface_structures,
      &physical_device,
      &logical_device,
    );

    // Now that the swapchain is created, we take the handle to the image out of it.
    let swap_chain_images = unsafe {
      swap_chain_structures
        .swap_chain_extension
        .get_swapchain_images(swap_chain_structures.swap_chain)
        .expect("Could not get images out of swapchain")
    };

    let image_views =
      create_image_views(&swap_chain_images, &swap_chain_structures, &logical_device);

    let render_pass = create_render_pass(&logical_device, &swap_chain_structures);

    // Finally, all the pieces of the Setup and presentation extensions are complete
    // Now we have to create the graphics pipeline and we'll be done.
    // NOTE: A new graphics pipeline needs to be constructed from scratch if it is
    // changed or created to begin with. So every different technique for drawing
    // etc would need to be created here (EG a regular object draw, a shadow map
    // pass, an alpha blend of just textures to create a new texture, would all use
    // slightly different pipeline configurations and need to be recreated).
    let pipeline_structures =
      create_graphics_pipeline(&logical_device, &swap_chain_structures, render_pass);

    let swap_chain_framebuffers = create_framebuffers(
      &logical_device,
      &image_views,
      &pipeline_structures,
      &swap_chain_structures,
    );

    let command_structures = create_command_structures(
      &instance,
      &physical_device,
      &logical_device,
      swap_chain_framebuffers.len(),
      &surface_structures,
    );

    let semaphores = create_sync_objects(&logical_device);

    let vulkan_context = VulkanContextInner {
      entry,
      instance,
      callback_structures,
      surface_structures,
      logical_device,
      graphics_queue,
      present_queue,
      swap_chain_structures,
      swap_chain_images,
      image_views,
      pipeline_structures,
      swap_chain_framebuffers,
      command_structures,
      synchronization: semaphores,
      in_flight_frame_index: 0,
    };

    setup_command_buffers(&vulkan_context);

    VulkanContext {
      inner: vulkan_context,
    }
  }

  pub fn draw_frame(&mut self) {
    let inner = &mut self.inner;
    let device = &inner.logical_device;
    let synchronization = &inner.synchronization;
    let in_flight_frame_index = inner.in_flight_frame_index;
    // Steps to success:
    // 0) Wait for fence of currently in flight frame.  This is to limit resource
    // use on queued draw calls. 1) Aquire image from swapchain
    // 2) Execute command buffer with image as attachment in framebuffer
    // 3) Return image to swapchain for presentation.
    // 4) fucking finally.
    unsafe {
      // 0
      device
        .wait_for_fences(
          &[synchronization.in_flight_fences[in_flight_frame_index]],
          true,
          std::u64::MAX,
        )
        .expect("Error waiting for fences!");
      device
        .reset_fences(&[synchronization.in_flight_fences[in_flight_frame_index]])
        .expect("Error resetting fences");

      // 1;  Ignoring whether swapchain is suboptimal for surface.
      let (image_index, suboptimal_) = inner
        .swap_chain_structures
        .swap_chain_extension
        .acquire_next_image(
          inner.swap_chain_structures.swap_chain,
          std::u64::MAX,
          synchronization.image_available_sems[in_flight_frame_index],
          vk::Fence::null(),
        )
        .expect("Could not acquire image!");

      // 2
      let submit_info = vk::SubmitInfo::builder()
        .wait_semaphores(&[synchronization.image_available_sems[in_flight_frame_index]])
        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT]) // Wait until we can draw colors
        .command_buffers(&[inner.command_structures.command_buffers[image_index as usize]])
        .signal_semaphores(&[synchronization.render_finished_sems[in_flight_frame_index]]) // Signal this when done drawing
        .build();
      device
        .queue_submit(
          inner.graphics_queue,
          &[submit_info],
          synchronization.in_flight_fences[in_flight_frame_index],
        ) //last param is a fence which is signaled when the submitted work is completed.
        .expect("Failed to submit draw command buffer!");

      // 3
      let present_info = vk::PresentInfoKHR::builder()
        .wait_semaphores(&[synchronization.render_finished_sems[in_flight_frame_index]])
        .swapchains(&[inner.swap_chain_structures.swap_chain])
        .image_indices(&[image_index])
        .build();

      inner
        .swap_chain_structures
        .swap_chain_extension
        .queue_present(inner.present_queue, &present_info)
        .expect("Could not present!");
    }

    inner.in_flight_frame_index = (inner.in_flight_frame_index + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  pub fn wait_for_idle(&self) {
    unsafe {
      self
        .inner
        .logical_device
        .device_wait_idle()
        .expect("Could not wait for idle!");
    }
  }
}
