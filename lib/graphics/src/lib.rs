use winit::dpi::LogicalSize;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::event::{Event, WindowEvent};
use winit::window::{Window, WindowBuilder};

struct GraphicsHandler {
    event_loop: EventLoop,
    window: Window,
}

impl GraphicsHandler {
    fn new(title: &str, (width, height): (u32,u32)) -> Self {
        let event_loop = EventLoop::new();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width,height))
            .build(&event_loop)?;

        Self {event_loop, window}
    }
}