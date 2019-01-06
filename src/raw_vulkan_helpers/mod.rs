#[cfg(target_os = "macos")]
use ash::extensions::MacOSSurface;
#[cfg(target_os = "windows")]
use ash::extensions::Win32Surface;
#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
use ash::extensions::XlibSurface;

use ash::extensions::{DebugReport, Surface};

#[cfg(all(unix, not(target_os = "android"), not(target_os = "macos")))]
pub fn extension_names() -> Vec<*const i8> {
  vec![
    Surface::name().as_ptr().into(),
    XlibSurface::name().as_ptr().into(),
    DebugReport::name().as_ptr().into(),
  ]
}

#[cfg(target_os = "macos")]
pub fn extension_names() -> Vec<*const i8> {
  vec![
    Surface::name().as_ptr().into(),
    MacOSSurface::name().as_ptr().into(),
    DebugReport::name().as_ptr().into(),
  ]
}

#[cfg(all(windows))]
pub fn extension_names() -> Vec<*const i8> {
  vec![
    Surface::name().as_ptr().into(),
    Win32Surface::name().as_ptr().into(),
    DebugReport::name().as_ptr().into(),
  ]
}
