mod raw_vulkan_helpers;

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
use std::os::raw::c_char;
use std::str;
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

const TITLE_BYTES: &'static [u8] = b"Vulkan";
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

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

struct HelloTriangleApplication {
  // Window related structures.
  window: Window,
  events_loop: EventsLoop,
  vulkan_structures: VulkanStructures,
}

struct VulkanStructures {
  entry: Entry, // Function loader.
  instance: Instance,
  callback_structures: Option<CallbackStructures>,
  surface_structures: VulkanSurfaceStructures,
  logical_device: Device,
  graphics_queue: vk::Queue,
  present_queue: vk::Queue,
}

// Each extension is wrapped in it's own struct with it's created structures to
// allow their functions to be accessed more easily in Drop, and logically group
// them together.
struct CallbackStructures {
  debug_utils_extension: extensions::DebugUtils,
  debug_callback_structure: vk::DebugUtilsMessengerEXT,
}

struct VulkanSurfaceStructures {
  surface_extension: extensions::Surface,
  surface: vk::SurfaceKHR,
}

impl HelloTriangleApplication {
  pub fn initialize() -> Self {
    // Event loop and window presented by the host platform.
    let events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
      .with_title(str::from_utf8(TITLE_BYTES).unwrap())
      .with_dimensions(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
      .with_resizable(false)
      .build(&events_loop)
      .expect("Error Creating Window");

    let vulkan_structures = Self::initialize_vulkan(&window);

    Self {
      window,
      events_loop,
      vulkan_structures,
    }
  }

  fn initialize_vulkan(window: &Window) -> VulkanStructures {
    // Entry is the vulkan library loader, it loads the vulkan shared object
    // (dll/so/etc), and then loads all the function pointers for vulkan versions
    // 1.0 and 1.1 from that.
    let entry: Entry = Entry::new().expect("Unable to load Vulkan dll and functions!");

    // Create the Vulkan instance (ie instantiate the client side driver).
    let instance = Self::create_instance(&entry);

    // Set up debug callback, so we can get messages through the Vulkan runtime (via
    // Rust FFI).
    let callback_structures = Self::setup_debug_callback(&entry, &instance);

    // Surface is actually created before physical device selection because it can
    // influence it.
    let surface = unsafe {
      raw_vulkan_helpers::create_surface(&entry, &instance, window)
        .expect("Could not create surface")
    };
    let surface_structures = VulkanSurfaceStructures {
      surface_extension: extensions::Surface::new(&entry, &instance),
      surface,
    };

    // For now just use the first one, who cares right?  In the future a scoring
    // system could be used, or a user selection, but anything works atm.
    let physical_device =
      Self::get_physical_devices_for_surface_drawing(&instance, &surface_structures)[0];

    // Right now no features are requested, but this will be different by the end of
    // the tutorial. Later this will include things like vertex shader, geometry
    // shader, etc.
    let (logical_device, queue_family_indices) = Self::create_logical_device(
      &instance,
      &physical_device,
      &surface_structures,
      &vk::PhysicalDeviceFeatures::builder().build(),
    );

    let graphics_queue = unsafe {
      logical_device.get_device_queue(queue_family_indices.graphics_queue_family.unwrap(), 0u32)
    };
    let present_queue = unsafe {
      logical_device.get_device_queue(queue_family_indices.display_queue_family.unwrap(), 0u32)
    };

    let vulkan_structures = VulkanStructures {
      entry,
      instance,
      callback_structures,
      surface_structures,
      logical_device,
      graphics_queue,
      present_queue,
    };
    vulkan_structures
  }

  fn create_instance(entry: &Entry) -> Instance {
    let extension_names_raw = Self::get_extension_names(&entry);

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
    Self::print_supported_extensions(&entry);

    // In order for Vulkan to render to a window, an extension needs to be loaded
    // specific for the platform. This code is copied from here: https://github.com/MaikKlein/ash/blob/master/examples/src/lib.rs
    // Debug report is included in these extension names.
    let mut extension_names_raw = raw_vulkan_helpers::extension_names();

    // Here we check if we are in debug mode, if so load the extensions for
    // debugging, such as DebugUtilsMessenger.
    if ENABLE_VALIDATION_LAYERS {
      Self::validate_layers_exist(
        &entry
          .enumerate_instance_layer_properties()
          .expect("Could not enumerate layer properties"),
      );

      extension_names_raw.push(extensions::DebugUtils::name().as_ptr());
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

  fn validate_layers_exist(layer_properties: &Vec<vk::LayerProperties>) {
    for &enabled_layer in VALIDATION_LAYERS_CSTR {
      unsafe {
        let search_layer = CStr::from_ptr(enabled_layer.as_ptr() as *const c_char)
          .to_str()
          .unwrap();
        Self::validate_layer_exists(layer_properties, search_layer);
      }
    }
  }

  fn validate_layer_exists(layer_properties: &Vec<vk::LayerProperties>, search_layer: &str) {
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
    let debug_utils_extension = extensions::DebugUtils::new(entry, instance);

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
        .create_debug_utils_messenger_ext(&debug_utils_messenger_create_info, None)
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
        .filter(|device| Self::is_device_suitable(instance, device, surface_structures))
        .collect();

      suitable_devices
        .iter()
        .for_each(|device| Self::print_device_information(instance, device));

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

    // Presenting to a surface is a queue specific feature, so it's imporant to
    // be sure one of the queues supports it for the device to be suitable
    Self::find_queue_families(instance, device, surface_structures).is_complete()
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
          .get_physical_device_surface_support_khr(*device, i as u32, surface_structures.surface);
        if present_supported {
          queue_family_indices.display_queue_family = Some(i as u32);
        }

        // All the needed queues have been found.
        if queue_family_indices.is_complete() {
          break;
        }
      }
    }

    queue_family_indices
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
    let queue_family_indices = Self::find_queue_families(instance, device, surface_structures);

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

    if !Self::queue_vec_already_contains_index(
      &queue_creation_infos,
      queue_family_indices.display_queue_family.unwrap(),
    ) {
      let present_queue_creation_info = vk::DeviceQueueCreateInfo::builder()
        .queue_family_index(queue_family_indices.display_queue_family.unwrap())
        .queue_priorities(&[1.0])
        .build();
    }

    // Here we create the actual device!  No extensions (not even VK_KHR_swapchain,
    // which is needed to draw to a window) are needed at this stage, but will
    // be added when needed in the future.
    let device_create_info = vk::DeviceCreateInfo::builder()
      .queue_create_infos(queue_creation_infos.as_slice())
      .enabled_features(features)
      .enabled_extension_names(&[]) // Add extensions here!
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
    queue_creation_infos: &Vec<vk::DeviceQueueCreateInfo>,
    index: u32,
  ) -> bool {
    queue_creation_infos
      .iter()
      .map(|x| x.queue_family_index)
      .any(|x| x == index)
  }

  fn main_loop(&mut self) {
    loop {
      let mut done = false;
      self.events_loop.poll_events(|ev| match ev {
        Event::WindowEvent {
          event: WindowEvent::CloseRequested,
          ..
        } => done = true,
        _ => (),
      });

      if done {
        return;
      }
    }
  }
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
        .surface_structures
        .surface_extension
        .destroy_surface_khr(self.surface_structures.surface, None);

      self.logical_device.destroy_device(None);

      // Destroy debug extension.
      if let Some(callback_structures) = self.callback_structures.as_mut() {
        callback_structures
          .debug_utils_extension
          .destroy_debug_utils_messenger_ext(callback_structures.debug_callback_structure, None);
      }

      // Destroy Vulkan instance.
      self.instance.destroy_instance(None);
    }
  }
}

// A struct of all the indices of the different queue families a vulkan devices
// supports.
#[derive(Default, Debug)]
struct HelloTriangeNeededQueueFamilyIndices {
  graphics_queue_family: Option<u32>,
  display_queue_family: Option<u32>,
}

impl HelloTriangeNeededQueueFamilyIndices {
  // This set of QueueFamilyIndices has each of the tracked types of necessary
  // queues (right now it is only graphics queues, but this will change when I'm
  // not stupid anymore).
  fn is_complete(&self) -> bool {
    self.graphics_queue_family.is_some() && self.display_queue_family.is_some()
  }
}

fn main() {
  println!("See GUI window...");
  let mut application = HelloTriangleApplication::initialize();
  application.main_loop();
}
