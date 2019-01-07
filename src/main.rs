mod raw_vulkan_helpers;

use crate::raw_vulkan_helpers::*;
use ash::{
  version::EntryV1_0,    // Needed for methods on Entry.
  version::InstanceV1_0, // Needed for methods on instance.
  vk,
  vk_make_version,
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

// If the compiler is in debug mode, enable validation layers so we can do extra checks, print debug messages, etc.
// By default, Vulkan does not do any kind of helping and validation, but these plugins will add that back in and be gone during a release build. Juicy.
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
}

struct CallbackStructures {
  debug_utils_extension: ash::extensions::DebugUtils,
  debug_callback_structure: vk::DebugUtilsMessengerEXT,
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

    let vulkan_structures = Self::initialize_vulkan();

    Self {
      window,
      events_loop,
      vulkan_structures,
    }
  }

  fn initialize_vulkan() -> VulkanStructures {
    // Entry is the vulkan library loader, it loads the vulkan shared object
    // (dll/so/etc), and then loads all the function pointers for vulkan versions
    // 1.0 and 1.1 from that.
    let entry: Entry = Entry::new().expect("Unable to load Vulkan dll and functions!");
    // Create the Vulkan instance (ie instantiate the client side driver).
    let instance = Self::create_instance(&entry);
    // Set up debug callback, so we can get messages through the Vulkan runtime (via Rust FFI).
    let callback_structures = Self::setup_debug_callback(&entry, &instance);
    let vulkan_structures = VulkanStructures {
      entry,
      instance,
      callback_structures,
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
    let mut extension_names_raw = extension_names();

    // Here we check if we are in debug mode, if so load the extensions for debugging, such as DebugUtilsMessenger.
    if ENABLE_VALIDATION_LAYERS {
      Self::validate_layers_exist(
        &entry
          .enumerate_instance_layer_properties()
          .expect("Could not enumerate layer properties"),
      );

      extension_names_raw.push(ash::extensions::DebugUtils::name().as_ptr());
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
    let debug_utils_extension = ash::extensions::DebugUtils::new(entry, instance);

    // I want all the message types and levels here (Except Info).  Also specify the callback function.
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

// Have to manually deallocate (vkDestroyInstance, destroy debug utils) etc.
impl Drop for VulkanStructures {
  fn drop(&mut self) {
    unsafe {
      // Destroy debug extension.
      if let Some(inner) = self.callback_structures.as_mut() {
        inner
          .debug_utils_extension
          .destroy_debug_utils_messenger_ext(inner.debug_callback_structure, None);
      }

      // Destroy Vulkan instance.
      self.instance.destroy_instance(None);
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

fn main() {
  println!("See GUI window...");
  let mut application = HelloTriangleApplication::initialize();
  application.main_loop();
}
