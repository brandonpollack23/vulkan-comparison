pub mod raw_vulkan_helpers;
mod vulkan;

use self::vulkan::VulkanContext;
use std::str;
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

pub struct HelloTriangleApplication {
  // Window related structures.
  window: Window,
  events_loop: EventsLoop,
  vulkan_context: VulkanContext,
}

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

// TODO make title, width, height params.
impl HelloTriangleApplication {
  pub fn initialize() -> Self {
    // Event loop and window presented by the host platform.
    let mut events_loop = EventsLoop::new();
    let window = WindowBuilder::new()
      .with_title(str::from_utf8(vulkan::TITLE_BYTES).unwrap())
      .with_dimensions(LogicalSize::new(f64::from(WIDTH), f64::from(HEIGHT)))
      .with_resizable(true)
      .build(&events_loop)
      .expect("Error Creating Window");

    // Swallow initial resize event.
    events_loop.poll_events(|ev| {});

    let vulkan_structures = VulkanContext::initialize_vulkan(&window);

    Self {
      window,
      events_loop,
      vulkan_context: vulkan_structures,
    }
  }

  pub fn main_loop(&mut self) {
    let mut done = false;
    let mut resized = false;
    let window = &self.window;
    let vulkan_context = &mut self.vulkan_context;

    while !done {
      self.events_loop.poll_events(|ev| match ev {
        Event::WindowEvent {
          event: WindowEvent::CloseRequested,
          ..
        } => done = true,
        Event::WindowEvent {
          event: WindowEvent::Resized(new_size),
          ..
        } => resized = true,
        _ => (),
      });

      vulkan_context.draw_frame(window, &mut resized);
    }

    // Wait for idle before exiting to prevent stomping currently ongoing
    // drawing/presentation operations.
    vulkan_context.wait_for_idle();
  }
}
