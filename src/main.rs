#[macro_use]
extern crate memoffset;

mod hello_shape_application;

use crate::hello_shape_application::*;

fn main() {
  // TODO replace with library for args.
  let args: Vec<String> = std::env::args().collect();
  let application_type = &args[1];
  println!("See GUI window...");

  match application_type.as_str() {
    "triangle" => {
      let mut application = HelloTriangleApplication::initialize();
      application.main_loop();
    }
    "rect" => {
      let mut application = HelloRectangleApplication::initialize();
      application.main_loop();
    }
    _ => panic!("No such shape!"),
  }
}
