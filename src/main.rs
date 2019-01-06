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
use std::str;
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

const TITLE: &[u8] = b"Vulkan";
const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

struct HelloTriangleApplication {
  // Window related structures.
  window: Window,
  events_loop: EventsLoop,
  vulkan_structures: VulkanStructures,
}

struct VulkanStructures {
  entry: Entry, // Function loader.
  instance: Instance,
}

// Have to manually deallocate (vkDestroyInstance) etc.
impl Drop for VulkanStructures {
  fn drop(&mut self) {
    unsafe {
      self.instance.destroy_instance(None);
    }
  }
}

impl HelloTriangleApplication {
  pub fn initialize() -> Self {
    // Event loop and window presented by the host platform.
    let events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
      .with_title(str::from_utf8(TITLE).unwrap())
      .with_dimensions(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
      .with_resizable(false)
      .build(&events_loop)
      .expect("Error Creating Window");

    // Entry is the vulkan library loader, it loads the vulkan shared object
    // (dll/so/etc), and then loads all the function pointers for vulkan versions
    // 1.0 and 1.1 from that.
    let entry = Entry::new().unwrap();
    Self::print_supported_extensions(&entry);

    let instance = Self::create_instance(&entry);

    let vulkan_structures = VulkanStructures { entry, instance };

    Self {
      window,
      events_loop,
      vulkan_structures,
    }
  }

  fn create_instance(entry: &Entry) -> Instance {
    // In order for Vulkan to render to a window, an extension needs to be loaded specific for the
    // platform. This code is copied from here: https://github.com/MaikKlein/ash/blob/master/examples/src/lib.rs
    let extension_names_raw = extension_names();

    // Various function calls in here are unsafe, creation of the instance, and
    // working with cstrings unchecked.
    unsafe {
      let name = CStr::from_bytes_with_nul_unchecked(TITLE);
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
        .build();

      let instance = entry
        .create_instance(&create_info, None)
        .expect("Error Creating instance");
      instance
    }
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

fn main() {
  println!("See GUI window...");
  let mut application = HelloTriangleApplication::initialize();
  application.main_loop();
}
