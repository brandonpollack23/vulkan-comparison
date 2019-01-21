pub mod raw_vulkan_helpers;
mod vulkan;

use self::vulkan::vulkan_structures::*;
use self::vulkan::ColoredVertex;
use self::vulkan::VulkanContext;
use ash::vk;
use cgmath::{Vector2, Vector3};
use std::str;
use winit::{dpi::LogicalSize, Event, EventsLoop, Window, WindowBuilder, WindowEvent};

const WIDTH: u32 = 800;
const HEIGHT: u32 = 600;

const COLORED_VERTICES: [ColoredVertex; 3] = [
  ColoredVertex {
    position: Vector2 {
      x: 0.0f32,
      y: -0.5f32,
    },
    color: Vector3 {
      x: 1.0f32,
      y: 0.0f32,
      z: 0.0f32,
    },
  },
  ColoredVertex {
    position: Vector2 {
      x: 0.5f32,
      y: 0.5f32,
    },
    color: Vector3 {
      x: 0.0f32,
      y: 1.0f32,
      z: 0.0f32,
    },
  },
  ColoredVertex {
    position: Vector2 {
      x: -0.5f32,
      y: 0.5f32,
    },
    color: Vector3 {
      x: 0.0f32,
      y: 0.0f32,
      z: 1.0f32,
    },
  },
];

pub struct HelloTriangleApplication {
  // Window related structures.
  window: Window,
  events_loop: EventsLoop,
  vulkan_context: VulkanContext,
  vertex_buffer: VulkanVertexBuffer,   // TODO make vec?
  offsets: (u32, Vec<vk::DeviceSize>), /* TODO make a descriptive struct.  This is size of the
                                        * buffer and the offsets of the buffer (which is 0 atm
                                        * since there's only one set of data in it) */
}

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

    let mut vulkan_context = VulkanContext::initialize_vulkan(&window);
    let vertex_buffer = vulkan_context.load_colored_vertices(&COLORED_VERTICES[..]);
    let offsets = (3u32, vec![0]);
    vulkan_context.setup_command_buffers(&[vertex_buffer.vertex_buffer], &offsets);

    Self {
      window,
      events_loop,
      vulkan_context,
      vertex_buffer,
      offsets,
    }
  }

  pub fn main_loop(&mut self) {
    let mut done = false;
    let mut resized = false;
    let mut enable_draw = true;

    let window = &self.window;
    let vulkan_context = &mut self.vulkan_context;
    let vertex_buffer = &mut self.vertex_buffer;
    let offsets = &mut self.offsets;

    while !done {
      self.events_loop.poll_events(|ev| match ev {
        Event::WindowEvent {
          event: WindowEvent::CloseRequested,
          ..
        } => {
          done = true;
        }
        Event::WindowEvent {
          event: WindowEvent::Resized(new_size),
          ..
        } => {
          if new_size.width == 0f64 || new_size.height == 0f64 {
            // Minimize.
            enable_draw = false;
          } else {
            enable_draw = true;
          }

          resized = true;
        }
        _ => (),
      });

      if enable_draw {
        if vulkan_context.draw_frame(window, &mut resized) {
          vulkan_context.setup_command_buffers(&[vertex_buffer.vertex_buffer], offsets);
        }
      }
    }

    // Wait for idle before exiting to prevent stomping currently ongoing
    // drawing/presentation operations.
    vulkan_context.wait_for_idle();
  }
}
