pub mod raw_vulkan_helpers;
mod vulkan;

use self::vulkan::*;
use std::str;
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

pub struct HelloTriangleApplication {
  // Window related structures.
  window: Window,
  events_loop: EventsLoop,
  vulkan_context: VulkanStructures,
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

// TODO make title, width, height params.
impl HelloTriangleApplication {
  pub fn initialize() -> Self {
    // Event loop and window presented by the host platform.
    let events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
      .with_title(str::from_utf8(vulkan::TITLE_BYTES).unwrap())
      .with_dimensions(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
      .with_resizable(false)
      .build(&events_loop)
      .expect("Error Creating Window");

    let vulkan_structures = vulkan::VulkanStructures::initialize_vulkan(&window);

    Self {
      window,
      events_loop,
      vulkan_context: vulkan_structures,
    }
  }

  pub fn main_loop(&mut self) {
    let mut done = false;
    while !done {
      self.events_loop.poll_events(|ev| match ev {
        Event::WindowEvent {
          event: WindowEvent::CloseRequested,
          ..
        } => done = true,
        _ => (),
      });

      self.vulkan_context.draw_frame();
    }

    // Wait for idle before exiting to prevent stomping currently ongoing
    // drawing/presentation operations.
    self.vulkan_context.wait_for_idle();
  }
}
