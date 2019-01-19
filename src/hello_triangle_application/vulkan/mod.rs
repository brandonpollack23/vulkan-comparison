use crate::hello_triangle_application::raw_vulkan_helpers;
use crate::hello_triangle_application::HEIGHT;
use crate::hello_triangle_application::WIDTH;
use ash::util::*;
use ash::{
  extensions,
  version::DeviceV1_0,   //Needed for methods on Device
  version::EntryV1_0,    // Needed for methods on Entry.
  version::InstanceV1_0, // Needed for methods on Instance.
  vk,
  vk_make_version,
  Device,
  Entry,
  Instance,
};
use std::ffi::CStr;
use std::fs;
use std::os::raw::c_char;
use std::str;
use winit::Window;

pub const TITLE_BYTES: &'static [u8] = b"Vulkan";

const VALIDATION_LAYERS_CSTR: &'static [&'static [u8]] =
  &[b"VK_LAYER_LUNARG_standard_validation\0"];

// If the compiler is in debug mode, enable validation layers so we can do extra
// checks, print debug messages, etc. By default, Vulkan does not do any kind of
// helping and validation, but these plugins will add that back in and be gone
// during a release build. Juicy.
#[cfg(debug_assertions)]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

pub struct VulkanStructures {
  entry: Entry, // Function loader.
  instance: Instance,
  callback_structures: Option<CallbackStructures>,
  surface_structures: VulkanSurfaceStructures,
  logical_device: Device,
  graphics_queue: vk::Queue,
  present_queue: vk::Queue,
  swap_chain_structures: VulkanSwapChainStructures,
  swap_chain_images: Vec<vk::Image>,
  image_views: Vec<vk::ImageView>,
  pipeline_structures: VulkanPipelineStructures,
  swap_chain_framebuffers: Vec<vk::Framebuffer>,
  command_structures: VulkanCommandStructures,
  semaphores: VulkanSemaphores,
}

// Each extension is wrapped in it's own struct with it's created structures to
// allow their functions to be accessed more easily in Drop, and logically group
// them together.
struct CallbackStructures {
  debug_utils_extension: extensions::ext::DebugUtils,
  debug_callback_structure: vk::DebugUtilsMessengerEXT,
}

struct VulkanSurfaceStructures {
  surface_extension: extensions::khr::Surface,
  surface: vk::SurfaceKHR,
}

struct VulkanSwapChainStructures {
  swap_chain_extension: extensions::khr::Swapchain,
  swap_chain: vk::SwapchainKHR,
  swap_chain_image_format: vk::Format,
  swap_chain_extent: vk::Extent2D,
}

struct VulkanPipelineStructures {
  render_pass: vk::RenderPass,
  pipeline_layout: vk::PipelineLayout,
  graphics_pipeline: vk::Pipeline,
}

struct VulkanCommandStructures {
  command_pool: vk::CommandPool,
  command_buffers: Vec<vk::CommandBuffer>,
}

struct VulkanSemaphores {
  image_available_sem: vk::Semaphore,
  render_finished_sem: vk::Semaphore,
}

// A struct of all the indices of the different queue families a vulkan devices
// supports.
#[derive(Default, Debug)]
struct HelloTriangeNeededQueueFamilyIndices {
  graphics_queue_family: Option<u32>,
  present_queue_family: Option<u32>,
}

impl HelloTriangeNeededQueueFamilyIndices {
  // This set of QueueFamilyIndices has each of the tracked types of necessary
  // queues (right now it is only graphics queues, but this will change when I'm
  // not stupid anymore).
  fn is_complete(&self) -> bool {
    self.graphics_queue_family.is_some() && self.present_queue_family.is_some()
  }
}

#[derive(Debug)]
struct SwapChainSupportDetails {
  capabilities: vk::SurfaceCapabilitiesKHR,
  formats: Vec<vk::SurfaceFormatKHR>,
  present_modes: Vec<vk::PresentModeKHR>,
}

// Have to manually deallocate (vkDestroyInstance, destroy debug utils) etc.
// Physical Devices are cleaned up when the instance is destroyed, so no need to
// do that manually. Logical devices, however, are not, since they're not part
// of the instance.
// Rule of thumb, anything create was called for, destroy is called for,
// everything else was already "part" of some other created thing (eg queues,
// physical devices).
impl Drop for VulkanStructures {
  fn drop(&mut self) {
    unsafe {
      self
        .logical_device
        .destroy_semaphore(self.semaphores.render_finished_sem, None);
      self
        .logical_device
        .destroy_semaphore(self.semaphores.image_available_sem, None);

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

impl VulkanStructures {
  pub fn initialize_vulkan(window: &Window) -> VulkanStructures {
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
    let surface = unsafe {
      raw_vulkan_helpers::create_surface(&entry, &instance, window)
        .expect("Could not create surface")
    };
    let surface_structures = VulkanSurfaceStructures {
      surface_extension: extensions::khr::Surface::new(&entry, &instance),
      surface,
    };

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

    let semaphores = create_semaphores(&logical_device);

    let vulkan_structures = VulkanStructures {
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
      semaphores,
    };
    vulkan_structures.setup_command_buffers();

    vulkan_structures
  }

  pub fn draw_frame(&self) {
    // Steps to success:
    // 1) Aquire image from swapchain
    // 2) Execute command buffer with image as attachment in framebuffer
    // 3) Return image to swapchain for presentation.
    // 4) fucking finally.
    unsafe {
      // 1;  Ignoring whether swapchain is suboptimal for surface.
      let (image_index, suboptimal_) = self
        .swap_chain_structures
        .swap_chain_extension
        .acquire_next_image(
          self.swap_chain_structures.swap_chain,
          std::u64::MAX,
          self.semaphores.image_available_sem,
          vk::Fence::null(),
        )
        .expect("Could not acquire image!");

      // 2
      let submit_info = vk::SubmitInfo::builder()
        .wait_semaphores(&[self.semaphores.image_available_sem])
        .wait_dst_stage_mask(&[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT]) // Wait until we can draw colors
        .command_buffers(&[self.command_structures.command_buffers[image_index as usize]])
        .signal_semaphores(&[self.semaphores.render_finished_sem]) // Signal this when done drawing
        .build();
      self
        .logical_device
        .queue_submit(self.graphics_queue, &[submit_info], vk::Fence::null())
        .expect("Failed to submit draw command buffer!");

      // 3
      let present_info = vk::PresentInfoKHR::builder()
        .wait_semaphores(&[self.semaphores.render_finished_sem])
        .swapchains(&[self.swap_chain_structures.swap_chain])
        .image_indices(&[image_index])
        .build();

      self
        .swap_chain_structures
        .swap_chain_extension
        .queue_present(self.present_queue, &present_info)
        .expect("Could not present!");
    }
  }

  fn setup_command_buffers(&self) {
    for i in 0..self.command_structures.command_buffers.len() {
      let begin_info = vk::CommandBufferBeginInfo::builder()
        .flags(vk::CommandBufferUsageFlags::SIMULTANEOUS_USE) // Resubmit allowed while pending execution.
        .build();

      unsafe {
        self
          .logical_device
          .begin_command_buffer(self.command_structures.command_buffers[i], &begin_info)
          .expect("Could not begin command buffer");
      }

      let clear_color = vk::ClearValue {
        color: vk::ClearColorValue {
          float32: [0f32, 0f32, 0f32, 1f32],
        },
      };

      let render_pass_info = vk::RenderPassBeginInfo::builder()
        .render_pass(self.pipeline_structures.render_pass)
        .framebuffer(self.swap_chain_framebuffers[i])
        .render_area(vk::Rect2D {
          offset: vk::Offset2D { x: 0, y: 0 },
          extent: self.swap_chain_structures.swap_chain_extent,
        })
        .clear_values(&[clear_color])
        .build();

      unsafe {
        self.logical_device.cmd_begin_render_pass(
          self.command_structures.command_buffers[i],
          &render_pass_info,
          vk::SubpassContents::INLINE,
        );

        self.logical_device.cmd_bind_pipeline(
          self.command_structures.command_buffers[i],
          vk::PipelineBindPoint::GRAPHICS,
          self.pipeline_structures.graphics_pipeline,
        );

        self
          .logical_device
          .cmd_draw(self.command_structures.command_buffers[i], 3, 1, 0, 0);

        self
          .logical_device
          .cmd_end_render_pass(self.command_structures.command_buffers[i]);

        self
          .logical_device
          .end_command_buffer(self.command_structures.command_buffers[i])
          .expect("Failed to record command buffer!");
      }
    }
  }
}

fn create_instance(entry: &Entry) -> Instance {
  let extension_names_raw = get_extension_names(&entry);

  // Various function calls in here are unsafe, creation of the instance, and
  // working with cstrings unchecked.
  unsafe {
    let name = CStr::from_bytes_with_nul_unchecked(TITLE_BYTES);
    // First create application info. s_type is handled by the builder's defaults in
    // all cases in ash (rust vulkan bindings with a little sauce). next being null
    // is also handled by the builder.
    let app_info = vk::ApplicationInfo::builder()
      .application_name(name)
      .application_version(0)
      .engine_name(name)
      .engine_version(0)
      .api_version(vk_make_version!(1, 0, 0))
      .build();

    // Then specify instance creation info, such as extensions
    // enabled extension length is the length of the above Vec, set by ash
    let create_info = vk::InstanceCreateInfo::builder()
      .application_info(&app_info)
      .enabled_extension_names(&extension_names_raw[..])
      .enabled_layer_names(if ENABLE_VALIDATION_LAYERS {
        std::mem::transmute::<&[&[u8]], &[*const c_char]>(VALIDATION_LAYERS_CSTR)
      } else {
        &[]
      })
      .build();

    let instance = entry
      .create_instance(&create_info, None)
      .expect("Error Creating instance");
    instance
  }
}

fn get_extension_names(entry: &Entry) -> Vec<*const c_char> {
  print_supported_extensions(&entry);

  // In order for Vulkan to render to a window, an extension needs to be loaded
  // specific for the platform. This code is copied from here: https://github.com/MaikKlein/ash/blob/master/examples/src/lib.rs
  // Debug report is included in these extension names.
  let mut extension_names_raw = raw_vulkan_helpers::extension_names();

  // Here we check if we are in debug mode, if so load the extensions for
  // debugging, such as DebugUtilsMessenger.
  if ENABLE_VALIDATION_LAYERS {
    validate_layers_exist(
      &entry
        .enumerate_instance_layer_properties()
        .expect("Could not enumerate layer properties"),
    );

    extension_names_raw.push(extensions::ext::DebugUtils::name().as_ptr());
  }

  extension_names_raw
}

fn print_supported_extensions(entry: &Entry) {
  println!("Supported extensions are:");
  for extension in entry
    .enumerate_instance_extension_properties()
    .expect("Could not enumerate extensions!")
  {
    println!("{}", unsafe {
      // All this ceremony is to treat an unsafe cstr buffer as a printable string.
      str::from_utf8(CStr::from_ptr(extension.extension_name.as_ptr()).to_bytes()).unwrap()
    });
  }
}

fn validate_layers_exist(layer_properties: &[vk::LayerProperties]) {
  for &enabled_layer in VALIDATION_LAYERS_CSTR {
    unsafe {
      let search_layer = CStr::from_ptr(enabled_layer.as_ptr() as *const c_char)
        .to_str()
        .unwrap();
      validate_layer_exists(layer_properties, search_layer);
    }
  }
}

fn validate_layer_exists(layer_properties: &[vk::LayerProperties], search_layer: &str) {
  for &layer in layer_properties.iter() {
    unsafe {
      let extant_layer =
        str::from_utf8(CStr::from_ptr(layer.layer_name.as_ptr()).to_bytes()).unwrap();
      if search_layer == extant_layer {
        return;
      }
    }
  }

  panic!("No such layer {}", search_layer);
}

fn setup_debug_callback(entry: &Entry, instance: &Instance) -> Option<CallbackStructures> {
  if !ENABLE_VALIDATION_LAYERS {
    return None;
  }

  // Load up the function pointers so we can use the functions to create.
  let debug_utils_extension = extensions::ext::DebugUtils::new(entry, instance);

  // I want all the message types and levels here (Except Info).  Also specify the
  // callback function.
  let debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
    .message_severity(
      vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE,
    )
    .message_type(
      vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
    )
    .pfn_user_callback(Some(debug_callback))
    .build();
  unsafe {
    // Build it using the function loaded above.
    let debug_callback_structure = debug_utils_extension
      .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
      .expect("Could not create debug utils message extension callback");

    Some(CallbackStructures {
      debug_utils_extension,
      debug_callback_structure,
    })
  }
}

fn get_physical_devices_for_surface_drawing(
  instance: &Instance,
  surface_structures: &VulkanSurfaceStructures,
) -> Vec<vk::PhysicalDevice> {
  unsafe {
    let devices = instance
      .enumerate_physical_devices()
      .expect("Unable to enumerate devices");

    println!("\nEnumerating your devices...");
    let mut suitable_devices: Vec<vk::PhysicalDevice> = devices
      .into_iter()
      .filter(|device| is_device_suitable(instance, device, surface_structures))
      .collect();

    suitable_devices
      .iter()
      .for_each(|device| print_device_information(instance, device));

    suitable_devices.shrink_to_fit();
    suitable_devices
  }
}

fn is_device_suitable(
  instance: &Instance,
  device: &vk::PhysicalDevice,
  surface_structures: &VulkanSurfaceStructures,
) -> bool {
  // This is such a basic application (read: I have no idea what I'm doing) that
  // anything that supports vulkan and the queue families we need is fine.

  // Presenting to a surface is a queue specific feature, so it's important to
  // be sure one of the queues supports it for the device to be suitable.
  let has_queue_families = find_queue_families(instance, device, surface_structures).is_complete();

  // We also need to verify the necessary extensions are supported by the physical
  // device. Right now that's just vk_khr_swapchain.
  let extensions_supported = unsafe {
    instance
      .enumerate_device_extension_properties(*device)
      .expect("Could not enumerate device extension properties")
      .iter()
      .any(|x| CStr::from_ptr(x.extension_name.as_ptr()) == extensions::khr::Swapchain::name())
  };

  // Just because the device supports the extensions doesn't mean the surface
  // does! Here we check if the surface supports the swapchain extension by
  // making sure there are formats and present modes for it in the surface.
  let swapchain_adequete = if extensions_supported {
    let swap_chain_support = query_swapchain_support(surface_structures, device);
    !swap_chain_support.present_modes.is_empty() && !swap_chain_support.formats.is_empty()
  } else {
    false
  };

  has_queue_families && extensions_supported && swapchain_adequete
}

// In Vulkan, there are different types of queues that come from different types
// of queue families. This function checks what families are supported by a
// vkPhysicalDevice and makes sure the device supports all we need.
fn find_queue_families(
  instance: &Instance,
  device: &vk::PhysicalDevice,
  surface_structures: &VulkanSurfaceStructures,
) -> HelloTriangeNeededQueueFamilyIndices {
  // For now just take queues with the feature I need.  In the future they can be
  // the same queue for performance improvements if they aren't already
  // (unlikely).
  let mut queue_family_indices = HelloTriangeNeededQueueFamilyIndices::default();
  unsafe {
    let queue_families = instance.get_physical_device_queue_family_properties(*device);
    for (i, queue_family) in queue_families.iter().enumerate() {
      if queue_family.queue_count <= 0 {
        continue;
      }

      if queue_family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
        queue_family_indices.graphics_queue_family = Some(i as u32);
      }

      let present_supported = surface_structures
        .surface_extension
        .get_physical_device_surface_support(*device, i as u32, surface_structures.surface);
      if present_supported {
        queue_family_indices.present_queue_family = Some(i as u32);
      }

      // All the needed queues have been found.
      if queue_family_indices.is_complete() {
        break;
      }
    }
  }

  queue_family_indices
}

fn query_swapchain_support(
  surface_structures: &VulkanSurfaceStructures,
  device: &vk::PhysicalDevice,
) -> SwapChainSupportDetails {
  unsafe {
    let capabilities = surface_structures
      .surface_extension
      .get_physical_device_surface_capabilities(*device, surface_structures.surface)
      .expect("Could not query surface capabilities");
    let formats = surface_structures
      .surface_extension
      .get_physical_device_surface_formats(*device, surface_structures.surface)
      .expect("could not query surface formats");
    let present_modes = surface_structures
      .surface_extension
      .get_physical_device_surface_present_modes(*device, surface_structures.surface)
      .expect("Could not query surface present modes");
    SwapChainSupportDetails {
      capabilities,
      formats,
      present_modes,
    }
  }
}

// Used to choose the optimal swap chain settings, format (color depth),
// presentation mode (conditions for swapping), and swap extent (resolution of
// images in swapchain).
fn choose_swap_surface_format(available_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
  if available_formats.len() == 1 && available_formats[0].format == vk::Format::UNDEFINED {
    // Surface has no preferred format, let's just pick BGR all 8 bits unsigned
    // in sRGB color_space
    return vk::SurfaceFormatKHR {
      format: vk::Format::B8G8R8A8_UNORM,
      color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
    };
  }

  // The surface is making us choose a format, so let's pick from what's
  // available.
  for available_format in available_formats {
    if available_format.format == vk::Format::B8G8R8A8_UNORM
      && available_format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
    {
      return available_format.clone();
    }
  }

  // Nothing matches our preferences, we could rank how "good" they are,
  // but whatever lets just take the first one.
  return available_formats[0].clone();
}

// Quite an important setting, defines how the swap chain operates (presents)
// the images to the surface. Can immediately show, may result in tearing.
// (VK_PRESENT_MODE_IMMEDIATE_KHR). Can Queue images to be shown and show them
// on swaps aka vblanks (VK_PRESSENT_MODE_FIFO_KHR). Guaranteed to be available.
// A relaxed version of above that will immediately show an image if the queue
// was empty instead of waiting for a swap, potentially tearing. Finally a queue
// that will replace old images if the FIFO fills up, avoids tearing with less
// latency (triple buffering) (VK_PRESENT_MODE_MAILBOX_KHR).
fn choose_swap_present_mode(available_present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
  // Prefer mailbox.
  for available_present_mode in available_present_modes {
    if available_present_mode.as_raw() == vk::PresentModeKHR::MAILBOX.as_raw() {
      return available_present_mode.clone();
    }
  }

  // Fallback to FIFO
  return vk::PresentModeKHR::FIFO;
}

// TODO forward WIDTH and HEIGHT as params.
// This selects the resolution of the swap chain images, almost always equal to
// the resolution of the window. The range of possible resolutions is defined in
// this structure.
fn choose_swap_extent(capabilites: &vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
  if capabilites.current_extent.width != std::u32::MAX {
    return capabilites.current_extent;
  }
  // Some window managers set current widht and height to max positive value to
  // indicate that we
  // can select any resolution we want.  In that case I just want to put in the
  // resolution within the max extents that matches the window
  vk::Extent2D {
    width: std::cmp::max(
      capabilites.min_image_extent.width,
      std::cmp::min(capabilites.max_image_extent.width, WIDTH),
    ),
    height: std::cmp::max(
      capabilites.min_image_extent.height,
      std::cmp::min(capabilites.max_image_extent.height, HEIGHT),
    ),
  }
}

// Choose number of images in the swap chain, try to do minimum plus 1 for
// triple buffering (TODO parameterize buffering).
fn choose_swap_image_count(capabilities: &vk::SurfaceCapabilitiesKHR) -> u32 {
  let image_count = capabilities.min_image_count + 1;

  // max image count of 0 means unlimited so here's some logic to check for that.
  if capabilities.max_image_count > 0 && image_count > capabilities.max_image_count {
    return capabilities.max_image_count;
  }

  image_count
}

fn print_device_information(instance: &Instance, device: &vk::PhysicalDevice) {
  unsafe {
    let device_properties = instance.get_physical_device_properties(*device);
    let gpu_type = match device_properties.device_type {
      vk::PhysicalDeviceType::DISCRETE_GPU => "Discrete",
      vk::PhysicalDeviceType::INTEGRATED_GPU => "Integrated",
      vk::PhysicalDeviceType::CPU => "CPU-type",
      vk::PhysicalDeviceType::VIRTUAL_GPU => "Virtual",
      _ => "other",
    };

    let device_features = instance.get_physical_device_features(*device);

    println!(
      "Device Found: {}, type: {}, it supports the following features\n{:?}",
      str::from_utf8(CStr::from_ptr(device_properties.device_name.as_ptr()).to_bytes()).unwrap(),
      gpu_type,
      device_features
    );
  }
}

// Return a logical device for the given physical device, with the necessary
// features.
fn create_logical_device(
  instance: &Instance,
  device: &vk::PhysicalDevice,
  surface_structures: &VulkanSurfaceStructures,
  features: &vk::PhysicalDeviceFeatures,
) -> (Device, HelloTriangeNeededQueueFamilyIndices) {
  let queue_family_indices = find_queue_families(instance, device, surface_structures);

  if !queue_family_indices.is_complete() {
    panic!(
      "Trying to create a physical device with queue families that are not supported! {:?}",
      queue_family_indices
    )
  }

  // Here we specify all the queue types we'll need in our device, as of right now
  // that's just graphics and present.
  let mut queue_creation_infos = vec![];

  queue_creation_infos.push(
    vk::DeviceQueueCreateInfo::builder()
      .queue_family_index(queue_family_indices.graphics_queue_family.unwrap())
      .queue_priorities(&[1.0])
      .build(),
  );

  if !queue_vec_already_contains_index(
    &queue_creation_infos,
    queue_family_indices.present_queue_family.unwrap(),
  ) {
    let present_queue_creation_info = vk::DeviceQueueCreateInfo::builder()
      .queue_family_index(queue_family_indices.present_queue_family.unwrap())
      .queue_priorities(&[1.0])
      .build();
  }

  // Here we create the actual device!
  // Needed extensions are swapchain for the logical device.
  let device_create_info = vk::DeviceCreateInfo::builder()
    .queue_create_infos(queue_creation_infos.as_slice())
    .enabled_features(features)
    .enabled_extension_names(&[extensions::khr::Swapchain::name().as_ptr()]) // Add extensions here!
    .enabled_layer_names(if ENABLE_VALIDATION_LAYERS {
      unsafe { std::mem::transmute::<&[&[u8]], &[*const c_char]>(VALIDATION_LAYERS_CSTR) }
    } else {
      &[]
    })
    .build();

  unsafe {
    (
      instance
        .create_device(*device, &device_create_info, None)
        .expect("Could not create logical device!"),
      queue_family_indices,
    )
  }
}

fn queue_vec_already_contains_index(
  queue_creation_infos: &[vk::DeviceQueueCreateInfo],
  index: u32,
) -> bool {
  queue_creation_infos
    .iter()
    .map(|x| x.queue_family_index)
    .any(|x| x == index)
}

fn create_swap_chain_structures(
  instance: &Instance,
  surface_structures: &VulkanSurfaceStructures,
  physical_device: &vk::PhysicalDevice,
  logical_device: &Device,
) -> VulkanSwapChainStructures {
  let swap_chain_extension = extensions::khr::Swapchain::new(instance, logical_device);
  // Create the swap chain so that we have something to use to draw to the
  // surface. There are thrree major properties of a swap chain that we will
  // also configure: Surface Format, Presentation Mode, and Swap Extent,
  // explained in their helper functions.
  let swap_chain_support_details = query_swapchain_support(&surface_structures, &physical_device);
  let surface_format = choose_swap_surface_format(&swap_chain_support_details.formats);
  let present_mode = choose_swap_present_mode(&swap_chain_support_details.present_modes);
  let extent = choose_swap_extent(&swap_chain_support_details.capabilities);
  let image_count = choose_swap_image_count(&swap_chain_support_details.capabilities);

  // If a window is resized etc, the swapchain needs to be recreated.  This is too
  // hard for now and we'll leave it for a TODO. Next we specify what happens if
  // we're using multiple command queues (one for graphics and one for presenting)
  // usually not. Sharing Mode Exclusive, one queue at a time owns the image at
  // a time and ownership is explicitly transferred, most performant.
  // Sharing Mode Concurrent, images can be used across queue families without
  // explicit ownership.
  let indices = find_queue_families(&instance, &physical_device, &surface_structures);

  let image_sharing_mode;
  let queue_family_indices;
  if indices.graphics_queue_family != indices.present_queue_family {
    // Different queues, synchronizing this is not something to do this early on,
    // later I'll learn how to do that.
    image_sharing_mode = vk::SharingMode::CONCURRENT;
    queue_family_indices = vec![
      indices.graphics_queue_family.unwrap(),
      indices.present_queue_family.unwrap(),
    ];
  } else {
    // Same queue.
    image_sharing_mode = vk::SharingMode::EXCLUSIVE;
    queue_family_indices = vec![];
  }

  let swap_chain_create_info_builder = vk::SwapchainCreateInfoKHR::builder()
    .min_image_count(image_count)
    .image_format(surface_format.format)
    .image_color_space(surface_format.color_space)
    .image_extent(extent)
    .image_array_layers(1) // How many layers each image consists of.  This is 1 unless you're doing stereoscopic 3D.
    // Specifies we're using this for drawing directly, there's also TRANSFER_DST, which is for
    // transfering memory to the swap chain, like in post processing etc. (remember that ogl
    // tutorial with the normals, depth, etc that were used for futher calculation?
    .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
    .pre_transform(swap_chain_support_details.capabilities.current_transform) // Don't rotate the screen or anything.
    .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE) // Don't blend with other windows in the window system.
    .present_mode(present_mode)
    .clipped(true) // Set clipping to true.  If pixels are obscured by other windows etc, we dont care about their
    // color and the can be clipped out.
    .old_swapchain(vk::SwapchainKHR::null())
    .image_sharing_mode(image_sharing_mode)
    .queue_family_indices(&queue_family_indices)
    .surface(surface_structures.surface);

  let swap_chain = unsafe {
    swap_chain_extension
      .create_swapchain(&swap_chain_create_info_builder.build(), None)
      .expect("Could not create swapchain")
  };

  VulkanSwapChainStructures {
    swap_chain_extension,
    swap_chain,
    swap_chain_image_format: surface_format.format,
    swap_chain_extent: extent,
  }
}

fn create_image_views(
  swap_chain_images: &[vk::Image],
  swap_chain_structures: &VulkanSwapChainStructures,
  logical_device: &Device,
) -> Vec<vk::ImageView> {
  swap_chain_images
    .iter()
    .map(|&image| {
      // Leave components default, so no swizzling of rgb
      let image_view_create_info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::TYPE_2D)
        .format(swap_chain_structures.swap_chain_image_format)
        .subresource_range(
          vk::ImageSubresourceRange::builder() // This describes image's purpose and what part should be accessed, for us this is a
            // color target with no mipmapping etc since it's the just whats drawing to the screen,
            // not a texture.
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1)
            .build(),
        )
        .build();
      unsafe {
        logical_device
          .create_image_view(&image_view_create_info, None)
          .expect("Error creating image view")
      }
    })
    .collect()
}

// Rust FFI, never thought I'd use this but here's a callback for errors to be
// called from the C vulkan API.
unsafe extern "system" fn debug_callback(
  message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
  message_types: vk::DebugUtilsMessageTypeFlagsEXT,
  p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
  p_user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
  eprintln!(
    "validation layer: {}",
    str::from_utf8(CStr::from_ptr((*p_callback_data).p_message).to_bytes()).unwrap()
  );

  // Always return false, true indicates that validation itself failed, only useful for developing validation layers so as a user of Vulkan I dont use it.
  return 0;
}

fn create_render_pass(
  logical_device: &Device,
  swap_chain_structures: &VulkanSwapChainStructures,
) -> vk::RenderPass {
  let color_attachment = vk::AttachmentDescription::builder()
    .format(swap_chain_structures.swap_chain_image_format)
    .samples(vk::SampleCountFlags::TYPE_1)
    .load_op(vk::AttachmentLoadOp::CLEAR) // Clear framebuffer to black before rendering new frame.
    .store_op(vk::AttachmentStoreOp::STORE) // Store writes.
    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE) // Not using stencil buffer, so dont care what the driver does with loads.
    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE) // or stores.
    .initial_layout(vk::ImageLayout::UNDEFINED) // Don't know layout before operation begins.
    .final_layout(vk::ImageLayout::PRESENT_SRC_KHR) // Layout for presenting to swap chain.
    .build();

  // Only using the one color attachment, since there's only one subpass for now.
  let attachment_reference = vk::AttachmentReference::builder()
    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
    .build();

  let subpass = vk::SubpassDescription::builder()
    .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
    .color_attachments(&[attachment_reference]) // Index here is the layout parameter in outputs in shaders.
    .build();

  // Set up IMPLICIT subpass dependencies that specefy memory and execution
  // dependencies between subpasses.  There are implicit subpasses around our
  // single subpass.
  let dependency = vk::SubpassDependency::builder()
    .src_subpass(vk::SUBPASS_EXTERNAL) // Dependency subpass (implicit subpass before or after render pass, here before.
    .dst_subpass(0) // Dependent subpass (0th one is the only one).
    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
    //    .src_access_mask() // no src access flags.
    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
    .dst_access_mask(
      vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
    ) // Prevent transition of stages until read and write is available on Color attachment stage.
    .build();

  let render_pass_create_info = vk::RenderPassCreateInfo::builder()
    .attachments(&[color_attachment])
    .subpasses(&[subpass])
    .dependencies(&[dependency])
    .build();

  unsafe {
    logical_device
      .create_render_pass(&render_pass_create_info, None)
      .expect("Could nto create render pass")
  }
}

// TODO parameterize, make public?
// TODO submodule and functions
fn create_graphics_pipeline(
  logical_device: &Device,
  swap_chain_structures: &VulkanSwapChainStructures,
  render_pass: vk::RenderPass,
) -> VulkanPipelineStructures {
  unsafe {
    let vert_shader_file = read_spv(&mut fs::File::open("shaders/vert.spv").unwrap())
      .expect("Could not read vertex shader.");
    let frag_shader_file = read_spv(&mut fs::File::open("shaders/frag.spv").unwrap())
      .expect("Could not read frag shader.");

    let vert_shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
      .code(&vert_shader_file)
      .build();
    let vert_shader = logical_device
      .create_shader_module(&vert_shader_module_create_info, None)
      .expect("Failed to create vertex shader module");
    let vert_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
      .stage(vk::ShaderStageFlags::VERTEX)
      .module(vert_shader)
      .name(CStr::from_bytes_with_nul_unchecked(b"main\0")) //shader entry point.
      .build();

    let frag_shader_module_create_info = vk::ShaderModuleCreateInfo::builder()
      .code(&frag_shader_file)
      .build();
    let frag_shader = logical_device
      .create_shader_module(&frag_shader_module_create_info, None)
      .expect("Failed to create fragment shader module");
    let frag_stage_create_info = vk::PipelineShaderStageCreateInfo::builder()
      .stage(vk::ShaderStageFlags::FRAGMENT)
      .module(frag_shader)
      .name(CStr::from_bytes_with_nul_unchecked(b"main\0")) //shader entry point.
      .build();

    let pipeline_stage_create_infos = [vert_stage_create_info, frag_stage_create_info];
    // END SHADER SETUP.

    // BEGIN FIXED FUNCTION SETUP.
    // Set up vertex input state.  This defines stuff like stride etc I think.
    // For this part of the tutirial we're going through, the vertex data is inside
    // the shader as a global, So there will be no vertex buffer, so we'll fill
    // this in in a way to indicate there's no vertex data.
    let vertex_input_state_create_info = vk::PipelineVertexInputStateCreateInfo::builder()
      .vertex_binding_descriptions(&[])
      .vertex_attribute_descriptions(&[])
      .build();

    // Input assembly, what kind of primitive geometry (triangles etc), should
    // primitive restart be enabled? Primitive restart is when you have a strip
    // (triangle or line strip) and a special index in the index buffer to specify
    // the end of a given strip in the vertex buffer.
    let input_assembly_create_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
      .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
      .primitive_restart_enable(false)
      .build();

    // Viewport and scissors
    // what region of the framebuffer to render to? Almost always the whole viewport
    // (until it isnt of course...). This will also include view depth, which
    // goes 0 to 1 in Vulkan.
    let viewport = vk::Viewport::builder()
      .x(0f32)
      .y(0f32)
      .min_depth(0f32)
      .max_depth(1f32)
      .width(swap_chain_structures.swap_chain_extent.width as f32)
      .height(swap_chain_structures.swap_chain_extent.height as f32)
      .build();
    // Scissors are what to discard from the viewport, which is nothing most of the
    // time.
    let viewport_scissors = vk::Rect2D::builder()
      .offset(vk::Offset2D { x: 0i32, y: 0i32 })
      .extent(swap_chain_structures.swap_chain_extent)
      .build();

    let pipeline_viewport_create_info = vk::PipelineViewportStateCreateInfo::builder()
      .viewports(&[viewport])
      .scissors(&[viewport_scissors])
      .build();

    // Rasterization, creates the fragments inside the primitives.  Also does
    // zbuffering, scissor test, and back face culling.
    let rasterizer_create_info = vk::PipelineRasterizationStateCreateInfo::builder()
      .depth_clamp_enable(false) // Instead of culling things otuside view bounds, clamps them to ends, useful for...shadow
      // maps apparently.
      .rasterizer_discard_enable(false) // setting to true disables rasterization.
      .polygon_mode(vk::PolygonMode::FILL) // Line mode, fill mode, etc.
      .line_width(1.0f32) // Default line width in fragments, anything more than 1 requires enabling the GPU feature.
      .cull_mode(vk::CullModeFlags::BACK) // cull back faces.
      .front_face(vk::FrontFace::CLOCKWISE) // Vertex order for a vertex to be considered "front facing" (used for back face culling for example).
      .depth_bias_enable(false) // Depth bias can alter depth values by adding a bias, used for shadow mapping sometimes.
      .depth_bias_constant_factor(0f32)
      .depth_bias_clamp(0f32)
      .depth_bias_slope_factor(0f32)
      .build();

    // Multisampling, aka MSAA, a form of anti aliasing. Works by combining shader
    // results that rasterize to the same pixel, makes edges look smooth.
    // This is cheaper than rendering a higher resolution then downscaling because
    // it only occurs when multiple fragments map to one pixel.
    let multisampling_create_info = vk::PipelineMultisampleStateCreateInfo::builder()
      .sample_shading_enable(false) // Not needed for now
      .rasterization_samples(vk::SampleCountFlags::TYPE_1)
      .min_sample_shading(1f32)
      //.sample_mask()
      .alpha_to_coverage_enable(false)
      .alpha_to_one_enable(false)
      .build();

    // Skipping deptha and stencil tests.  I'm not doing this so I dont need a
    // create_info for them.

    // Color blending. Combining color already in buffer with fragment's output
    // color (like for transparency). We don't need it now so it's off
    let color_blend_attachment = vk::PipelineColorBlendAttachmentState::builder()
      .color_write_mask(
        vk::ColorComponentFlags::R
          | vk::ColorComponentFlags::G
          | vk::ColorComponentFlags::B
          | vk::ColorComponentFlags::A,
      )
      .blend_enable(false) // No need to blend yet.
      .src_color_blend_factor(vk::BlendFactor::ONE) // All the following are unneeded because blending is off, but it'll produce alpha blending.
      .dst_color_blend_factor(vk::BlendFactor::ZERO)
      .color_blend_op(vk::BlendOp::ADD)
      .src_alpha_blend_factor(vk::BlendFactor::ONE)
      .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
      .alpha_blend_op(vk::BlendOp::ADD)
      .build();
    let color_blend_state_create_info = vk::PipelineColorBlendStateCreateInfo::builder()
      .logic_op_enable(false)
      .logic_op(vk::LogicOp::COPY)
      .attachments(&[color_blend_attachment])
      .blend_constants([0.0; 4])
      .build();

    // Dynamic state.  So some things CAN be changed without recreating the entire
    // pipeline, and that is configured here. Things like viewport size, line
    // width, and blend constants.  I'm not doing this so I wont create one.

    // Pipeline layout.
    // Specify uniforms for the shader pipeline here.  This object is referenced
    // outside of here, so it'll be part of VulkanPipelineStructures, but there
    // is nothing to configure so it's empty for now.
    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo::builder()
      .set_layouts(&[])
      .push_constant_ranges(&[]);

    let pipeline_layout = logical_device
      .create_pipeline_layout(&pipeline_layout_create_info, None)
      .expect("Unable to create pipeline layout!");
    // END OF FIXED FUNCTIONS CONFIGURATION!!!

    let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::builder()
      .stages(&pipeline_stage_create_infos)
      .vertex_input_state(&vertex_input_state_create_info)
      .input_assembly_state(&input_assembly_create_info)
      .viewport_state(&pipeline_viewport_create_info)
      .rasterization_state(&rasterizer_create_info)
      .multisample_state(&multisampling_create_info)
      //.depth_stencil_state() default null, unused
      .color_blend_state(&color_blend_state_create_info)
      //.dynamic_state() null, defaut unusued
      .layout(pipeline_layout)
      .render_pass(render_pass)
      .subpass(0)
      //.base_pipeline_handle(vk::Pipeline::default()) // used if creating pipeline out of anohter, more efficient, but there is only one pipeline now so it isn't needed.
      .build();

    // No pipeline cache for now, speeds up pipeline creation but idk how to use
    // this. For now only one, so hard moving the first element out.
    let graphics_pipeline = logical_device
      .create_graphics_pipelines(
        vk::PipelineCache::default(),
        &[graphics_pipeline_create_info],
        None,
      )
      .expect("Could not create graphics pipeline")[0];

    // Cleanup.
    logical_device.destroy_shader_module(vert_shader, None);
    logical_device.destroy_shader_module(frag_shader, None);

    VulkanPipelineStructures {
      render_pass,
      pipeline_layout,
      graphics_pipeline,
    }
  }
}

fn create_framebuffers(
  logical_device: &Device,
  image_views: &[vk::ImageView],
  graphics_pipeline_structures: &VulkanPipelineStructures,
  swap_chain_structures: &VulkanSwapChainStructures,
) -> Vec<vk::Framebuffer> {
  let mut framebuffers: Vec<vk::Framebuffer> = Vec::with_capacity(image_views.len());

  for image_view in image_views {
    let frame_buffer_create_info = vk::FramebufferCreateInfo::builder()
      .render_pass(graphics_pipeline_structures.render_pass)
      .attachments(&[*image_view])
      .width(swap_chain_structures.swap_chain_extent.width)
      .height(swap_chain_structures.swap_chain_extent.height)
      .layers(1)
      .build();

    unsafe {
      let framebuffer = logical_device
        .create_framebuffer(&frame_buffer_create_info, None)
        .expect("Unable to create framebuffer");
      framebuffers.push(framebuffer);
    }
  }

  framebuffers
}

fn create_command_structures(
  instance: &Instance,
  device: &vk::PhysicalDevice,
  logical_device: &Device,
  num_framebuffers: usize,
  surface_structures: &VulkanSurfaceStructures,
) -> VulkanCommandStructures {
  let queue_family_indices = find_queue_families(instance, device, surface_structures);

  let command_pool_create_info = vk::CommandPoolCreateInfo::builder()
    .queue_family_index(queue_family_indices.graphics_queue_family.unwrap())
    .build();

  unsafe {
    let command_pool = logical_device
      .create_command_pool(&command_pool_create_info, None)
      .expect("Could not create command pool");
    let command_buffers = create_command_buffers(logical_device, num_framebuffers, command_pool);

    VulkanCommandStructures {
      command_pool,
      command_buffers,
    }
  }
}

fn create_command_buffers(
  logical_device: &Device,
  num_framebuffers: usize,
  command_pool: vk::CommandPool,
) -> Vec<vk::CommandBuffer> {
  let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::builder()
    .command_pool(command_pool)
    .level(vk::CommandBufferLevel::PRIMARY)
    .command_buffer_count(num_framebuffers as u32)
    .build();

  unsafe {
    logical_device
      .allocate_command_buffers(&command_buffer_allocate_info)
      .expect("Could not allocate command buffer!")
  }
}

fn create_semaphores(logical_device: &Device) -> VulkanSemaphores {
  let sem_create_info = vk::SemaphoreCreateInfo::default();
  unsafe {
    let image_available_sem = logical_device
      .create_semaphore(&sem_create_info, None)
      .expect("Failed to create semaphore!");
    let render_finished_sem = logical_device
      .create_semaphore(&sem_create_info, None)
      .expect("Failed to create semaphore!");

    VulkanSemaphores {
      image_available_sem,
      render_finished_sem,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use winit::dpi::LogicalSize;
  use winit::*;

  #[test]
  fn test_vulkan_initializes_without_crashing() {
    let window = create_window();
    VulkanStructures::initialize_vulkan(&window);
  }

  fn create_window() -> Window {
    let events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
      .with_title("Test Title")
      .with_dimensions(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
      .with_resizable(false)
      .build(&events_loop)
      .expect("Error Creating Window");
    window
  }
}
