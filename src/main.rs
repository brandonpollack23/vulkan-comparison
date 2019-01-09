mod hello_triangle_application;

use crate::hello_triangle_application::HelloTriangleApplication;

fn main() {
  println!("See GUI window...");
  let mut application = HelloTriangleApplication::initialize();
  application.main_loop();
}
