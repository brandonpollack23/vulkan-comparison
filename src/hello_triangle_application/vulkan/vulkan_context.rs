use super::initialization_helpers::*;
use super::vulkan_structures::*;
use ash::util::Align;
use ash::{extensions, version::DeviceV1_0, version::InstanceV1_0, vk, Device, Entry, Instance};
use cgmath::{Vector2, Vector3};
use std::mem::align_of;
use winit::{dpi::LogicalSize, Window};
pub struct VulkanContext {
  inner: VulkanContextInner,
}

pub struct VulkanContextInner {
  pub entry: Entry, // Function loader.
  pub instance: Instance,
  pub callback_structures: Option<CallbackStructures>,
  pub surface_structures: VulkanSurfaceStructures,
  pub physical_device: vk::PhysicalDevice,
  pub logical_device: Device,
  pub graphics_queue: vk::Queue,
  pub present_queue: vk::Queue,
  pub swap_chain_extension: extensions::khr::Swapchain,
  pub swap_chain_structures: VulkanSwapChainStructures,
  pub image_views: Vec<vk::ImageView>,
  pub pipeline_structures: VulkanPipelineStructures,
  pub swap_chain_framebuffers: Vec<vk::Framebuffer>,
  pub command_structures: VulkanCommandStructures,
  pub synchronization: VulkanSynchronization,
  pub in_flight_frame_index: usize,
  pub buffer_allocations: Vec<vk::Buffer>,
  pub memory_allocations: Vec<vk::DeviceMemory>,
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
      self.buffer_allocations.iter().for_each(|&buffer| {
        self.logical_device.destroy_buffer(buffer, None);
      });
      self.memory_allocations.iter().for_each(|&memory| {
        self.logical_device.free_memory(memory, None);
      });

      cleanup_swap_chain(self);

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
    let swap_chain_extension = extensions::khr::Swapchain::new(&instance, &logical_device);
    let swap_chain_structures = create_swap_chain_structures(
      &instance,
      &surface_structures,
      &physical_device,
      &swap_chain_extension,
      window.get_inner_size().unwrap(),
    );

    // Now that the swapchain is created, we take the handle to the image out of it.
    let image_views = create_image_views(
      &swap_chain_extension,
      &swap_chain_structures,
      &logical_device,
    );

    // Finally, all the pieces of the Setup and presentation extensions are complete
    // Now we have to create the graphics pipeline and we'll be done.
    // NOTE: A new graphics pipeline needs to be constructed from scratch if it is
    // changed or created to begin with. So every different technique for drawing
    // etc would need to be created here (EG a regular object draw, a shadow map
    // pass, an alpha blend of just textures to create a new texture, would all use
    // slightly different pipeline configurations and need to be recreated).

    // Render pass is just moved right in, but more fine customization might be
    // needed later so i won't move it into create_render_pass.
    let render_pass = create_render_pass(&logical_device, &swap_chain_structures);
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
      physical_device,
      logical_device,
      graphics_queue,
      present_queue,
      swap_chain_extension,
      swap_chain_structures,
      image_views,
      pipeline_structures,
      swap_chain_framebuffers,
      command_structures,
      synchronization: semaphores,
      in_flight_frame_index: 0,
      buffer_allocations: Vec::new(),
      memory_allocations: Vec::new(),
    };

    VulkanContext {
      inner: vulkan_context,
    }
  }

  pub fn setup_command_buffers(
    &self,
    buffers: &[vk::Buffer],
    space_information: &(u32, Vec<vk::DeviceSize>),
  ) {
    for i in 0..self.inner.command_structures.command_buffers.len() {
      let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE) // Resubmit allowed while pending execution.
        .build();

      unsafe {
        self
          .inner
          .logical_device
          .begin_command_buffer(
            self.inner.command_structures.command_buffers[i],
            &begin_info,
          )
          .expect("Could not begin command buffer");
      }

      let clear_color = vk::ClearValue {
        color: vk::ClearColorValue {
          float32: [0f32, 0f32, 0f32, 1f32],
        },
      };

      let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(self.inner.pipeline_structures.render_pass)
        .framebuffer(self.inner.swap_chain_framebuffers[i])
        .render_area(vk::Rect2D {
          offset: vk::Offset2D { x: 0, y: 0 },
          extent: self.inner.swap_chain_structures.swap_chain_extent,
        })
        .clear_values(&[clear_color])
        .build();

      unsafe {
        self.inner.logical_device.cmd_begin_render_pass(
          self.inner.command_structures.command_buffers[i],
          &render_pass_info,
          vk::SubpassContents::INLINE,
        );

        self.inner.logical_device.cmd_bind_pipeline(
          self.inner.command_structures.command_buffers[i],
          vk::PipelineBindPoint::GRAPHICS,
          self.inner.pipeline_structures.graphics_pipeline,
        );

        self.inner.logical_device.cmd_bind_vertex_buffers(
          self.inner.command_structures.command_buffers[i],
          0,
          buffers,
          &space_information.1,
        );

        self.inner.logical_device.cmd_draw(
          self.inner.command_structures.command_buffers[i],
          space_information.0,
          1,
          0,
          0,
        );

        self
          .inner
          .logical_device
          .cmd_end_render_pass(self.inner.command_structures.command_buffers[i]);

        self
          .inner
          .logical_device
          .end_command_buffer(self.inner.command_structures.command_buffers[i])
          .expect("Failed to record command buffer!");
      }
    }
  }

  /// Returns true if the command buffers require reset.
  pub fn draw_frame(&mut self, window: &Window, framebuffer_resized: &mut bool) -> bool {
    let device = &self.inner.logical_device;
    let synchronization = &self.inner.synchronization;
    let in_flight_frame_index = self.inner.in_flight_frame_index;
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

      // 1;  Ignoring whether swapchain is suboptimal for surface.
      let next_image_result = self.inner.swap_chain_extension.acquire_next_image(
        self.inner.swap_chain_structures.swap_chain,
        std::u64::MAX,
        synchronization.image_available_sems[in_flight_frame_index],
        vk::Fence::null(),
      );

      let (image_index, suboptimal_) = match next_image_result {
        Ok(res) => res,
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
          self.recreate_swap_chain(window.get_inner_size().unwrap());
          return true;
        }
        Err(_) => panic!("Unable to acquire next image!"),
      };

      // Reset the fences before submitting the next queue, but after we're certain
      // we'll draw.
      device
        .reset_fences(&[synchronization.in_flight_fences[in_flight_frame_index]])
        .expect("Error resetting fences");

      // 2
      let submit_info = vk::SubmitInfo::builder()
        .wait_semaphores(&[synchronization.image_available_sems[in_flight_frame_index]])
        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT]) // Wait until we can draw colors
        .command_buffers(&[self.inner.command_structures.command_buffers[image_index as usize]])
        .signal_semaphores(&[synchronization.render_finished_sems[in_flight_frame_index]]) // Signal this when done drawing
        .build();

      device
        .queue_submit(
          self.inner.graphics_queue,
          &[submit_info],
          synchronization.in_flight_fences[in_flight_frame_index],
        ) //last param is a fence which is signaled when the submitted work is completed.
        .expect("Failed to submit draw command buffer!");

      // 3
      let present_info = vk::PresentInfoKHR::builder()
        .wait_semaphores(&[synchronization.render_finished_sems[in_flight_frame_index]])
        .swapchains(&[self.inner.swap_chain_structures.swap_chain])
        .image_indices(&[image_index])
        .build();

      let present_result = self
        .inner
        .swap_chain_extension
        .queue_present(self.inner.present_queue, &present_info);

      let should_recreate_swapchain = match present_result {
        Err(vk::Result::ERROR_OUT_OF_DATE_KHR) | Ok(true) => true,
        Ok(false) => false,
        _ => panic!("Could not present to queue!"),
      };

      if should_recreate_swapchain || *framebuffer_resized {
        *framebuffer_resized = false;
        self.recreate_swap_chain(window.get_inner_size().unwrap());
        return true;
      }
    }

    self.inner.in_flight_frame_index =
      (self.inner.in_flight_frame_index + 1) % MAX_FRAMES_IN_FLIGHT;

    false
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

  pub fn load_colored_vertices(&mut self, vertices: &[ColoredVertex]) -> VulkanVertexBuffer {
    unsafe {
      // 1 create the buffer
      let buffer_info = vk::BufferCreateInfo::builder()
        .size(std::mem::size_of_val(vertices) as u64)
        .usage(vk::BufferUsageFlags::VERTEX_BUFFER)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .build();

      let vertex_buffer = self
        .inner
        .logical_device
        .create_buffer(&buffer_info, None)
        .expect("Could not create vertex buffer!");
      self.inner.buffer_allocations.push(vertex_buffer);

      let memory_requirements = self
        .inner
        .logical_device
        .get_buffer_memory_requirements(vertex_buffer);

      // 2 Allocate memory of the correct type for the buffer.
      let alloc_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(memory_requirements.size)
        .memory_type_index(self.find_memory_type(
          memory_requirements.memory_type_bits,
          vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        ))
        .build();

      let vertex_buffer_memory = self
        .inner
        .logical_device
        .allocate_memory(&alloc_info, None)
        .expect("Could not allocate vertex buffer memory!");
      self.inner.memory_allocations.push(vertex_buffer_memory);

      self
        .inner
        .logical_device
        .bind_buffer_memory(vertex_buffer, vertex_buffer_memory, 0);

      // 3 copy the vertex data into the buffer.
      let mapped_memory = self
        .inner
        .logical_device
        .map_memory(
          vertex_buffer_memory,
          0,
          buffer_info.size,
          vk::MemoryMapFlags::default(),
        )
        .expect("Could not memory map from device!");
      let mut memory_slice = Align::new(
        mapped_memory,
        align_of::<u32>() as u64,
        memory_requirements.size,
      );
      memory_slice.copy_from_slice(vertices); // Guaranteed to finish because requested HOST_COHERENT memory.
      self.inner.logical_device.unmap_memory(vertex_buffer_memory);

      VulkanVertexBuffer {
        vertex_buffer,
        buffer_index: self.inner.buffer_allocations.len() - 1,
        memory_index: self.inner.memory_allocations.len() - 1,
      }
    }
  }

  pub fn free_vertex_buffer(&mut self, buffer: &VulkanVertexBuffer) {
    let vk_buffer = self.inner.buffer_allocations.remove(buffer.buffer_index);
    let vk_memory = self.inner.memory_allocations.remove(buffer.memory_index);
    unsafe {
      self.inner.logical_device.destroy_buffer(vk_buffer, None);
      self.inner.logical_device.free_memory(vk_memory, None);
    }
  }

  unsafe fn find_memory_type(&self, type_filter: u32, properties: vk::MemoryPropertyFlags) -> u32 {
    let mem_properties = self
      .inner
      .instance
      .get_physical_device_memory_properties(self.inner.physical_device);

    for i in 0..mem_properties.memory_type_count {
      if (type_filter & (1 << i)) != 0
        && mem_properties.memory_types[i as usize].property_flags & properties == properties
      {
        return i as u32;
      }
    }

    panic!("Failed to find suitable memory type!");
  }

  unsafe fn recreate_swap_chain(&mut self, new_size: LogicalSize) {
    self.inner.logical_device.device_wait_idle();

    let inner = &mut self.inner;

    cleanup_swap_chain(inner);

    let instance = &inner.instance;
    let surface_structures = &inner.surface_structures;
    let physical_device = &inner.physical_device;
    let logical_device = &inner.logical_device;
    let command_structures = &inner.command_structures;

    inner.swap_chain_structures = create_swap_chain_structures(
      instance,
      surface_structures,
      physical_device,
      &inner.swap_chain_extension,
      new_size,
    );
    inner.image_views = create_image_views(
      &inner.swap_chain_extension,
      &inner.swap_chain_structures,
      logical_device,
    );
    inner.pipeline_structures.render_pass =
      create_render_pass(logical_device, &inner.swap_chain_structures);
    inner.pipeline_structures = create_graphics_pipeline(
      logical_device,
      &inner.swap_chain_structures,
      inner.pipeline_structures.render_pass,
    );
    inner.swap_chain_framebuffers = create_framebuffers(
      logical_device,
      &inner.image_views,
      &inner.pipeline_structures,
      &inner.swap_chain_structures,
    );
    inner.command_structures.command_buffers = create_command_buffers(
      logical_device,
      inner.swap_chain_framebuffers.len(),
      command_structures.command_pool,
    );
  }
}

unsafe fn cleanup_swap_chain(vulkan_context_inner: &mut VulkanContextInner) {
  vulkan_context_inner.logical_device.free_command_buffers(
    vulkan_context_inner.command_structures.command_pool,
    &vulkan_context_inner.command_structures.command_buffers,
  );
  vulkan_context_inner
    .command_structures
    .command_buffers
    .clear();

  for framebuffer in vulkan_context_inner.swap_chain_framebuffers.iter() {
    vulkan_context_inner
      .logical_device
      .destroy_framebuffer(*framebuffer, None);
  }

  vulkan_context_inner.logical_device.destroy_pipeline(
    vulkan_context_inner.pipeline_structures.graphics_pipeline,
    None,
  );

  vulkan_context_inner.logical_device.destroy_pipeline_layout(
    vulkan_context_inner.pipeline_structures.pipeline_layout,
    None,
  );

  vulkan_context_inner
    .logical_device
    .destroy_render_pass(vulkan_context_inner.pipeline_structures.render_pass, None);

  vulkan_context_inner
    .image_views
    .iter()
    .for_each(|image_view| {
      vulkan_context_inner
        .logical_device
        .destroy_image_view(*image_view, None);
    });

  vulkan_context_inner
    .swap_chain_extension
    .destroy_swapchain(vulkan_context_inner.swap_chain_structures.swap_chain, None);
}

// TODO try f64, set format to R64...
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ColoredVertex {
  pub position: Vector2<f32>,
  pub color: Vector3<f32>,
}

impl ColoredVertex {
  pub fn get_binding_description() -> vk::VertexInputBindingDescription {
    vk::VertexInputBindingDescription::builder()
      .binding(0)
      .stride(std::mem::size_of::<ColoredVertex>() as u32)
      .input_rate(vk::VertexInputRate::VERTEX)
      .build()
  }

  pub fn get_attribute_descriptions() -> [vk::VertexInputAttributeDescription; 2] {
    [
      vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(0)
        .format(vk::Format::R32G32_SFLOAT)
        .offset(offset_of!(ColoredVertex, position) as u32)
        .build(),
      vk::VertexInputAttributeDescription::builder()
        .binding(0)
        .location(1)
        .format(vk::Format::R32G32B32_SFLOAT)
        .offset(offset_of!(ColoredVertex, color) as u32)
        .build(),
    ]
  }
}
