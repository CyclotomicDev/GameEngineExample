//use winit::event_loop::{ControlFlow, EventLoop};
use control::*;
use graphics::GraphicsHandler;
use data::DataHandler;
use control::ControlHandler;
use layers::AnyLayer;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::{Window, WindowBuilder},
    platform::pump_events::{PumpStatus, EventLoopExtPumpEvents},
};
use std::time::Duration;
use tokio::{join, sync::Mutex};
use std::sync::Arc;
use winit::dpi::LogicalSize;
use anyhow::{anyhow, Result};
use log::*;

//#[tokio::main(flavor = "current_thread")]
#[tokio::main]
async fn main() -> Result<()> {
    
    simple_logging::log_to_file("test.log", LevelFilter::Info)?;

    let mut window_handler = WindowHandler::new("Project G03Alpha", (1200,800))?;

    let mut graphics = GraphicsHandler::new(&window_handler.window)?;
    let data = Arc::new(Mutex::new(DataHandler::new()?));
    let mut control = ControlHandler::new();

    //let graphics_future = graphics_cycle(InstructionBuffer::default(), &window_handler.window, graphics.clone());
    let data_future = data_cycle(InstructionBuffer::default(), data.clone());
    // tokio::pin!(graphics_future);
    tokio::pin!(data_future);

    let mut buffer_collection = BufferColection::default(); 

    'main: loop {
        // Handle window events
        let window = &window_handler.window;
        let event_loop = &window_handler.event_loop;

        let mut buffer = InstructionBuffer::default();
        let timeout = Some(Duration::ZERO);
        event_loop.set_control_flow(ControlFlow::Poll);
        

        let status = window_handler.event_loop.pump_events( timeout,  |event, elwt| {
            match event {
                Event::WindowEvent { 
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window.id() == window_id => {
                    info!("Window close requested");
                    elwt.exit();
                },
                Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                    debug!("Size: {:?}", size);
                    if size.width == 0 || size.height == 0 {
                        graphics.set_minisized(true);
                    } else {
                        graphics.set_minisized(false);
                        graphics.set_resized();
                    }
                },
                Event::AboutToWait => {
                    // Continuous requests (rendering)
                    graphics.render(window);
                },
                 Event::WindowEvent {
                    event: WindowEvent::RedrawRequested, 
                    ..
                } => {
                    // // Non-continuous requests 
                },
                _ => (),
            }
        });

        if let PumpStatus::Exit(exit_code) = status {
            info!("Main loop exited");
            break 'main;
        }

        // Handle async
        tokio::select! {
            _ = &mut data_future => {
                data_future.set(data_cycle(InstructionBuffer::default(), data.clone()))
            },
            _ = async {} => (),
        }
    }

    join!(data_future).0?;

    Ok(())

}

async fn graphics_cycle(mut instructions: InstructionBuffer, window: &Window, graphics: Arc<Mutex<GraphicsHandler>>) -> Result<InstructionBuffer> {
    let mut graphics = graphics.lock().await;

    instructions.execute_all(graphics.as_any_mut());

    graphics.render(window)?;

    Ok(InstructionBuffer::default())
}

async fn data_cycle(mut instructions: InstructionBuffer, data: Arc<Mutex<DataHandler>>) -> Result<InstructionBuffer> {
    let mut data = data.lock().await;

    instructions.execute_all(data.as_any_mut());

    println!("Data pass");

    tokio::time::sleep(Duration::from_secs(1)).await;

    Ok(InstructionBuffer::default())
}

/// Contains all information about window control
struct WindowHandler {
    event_loop: EventLoop<()>,
    window: Window,
}

/// To do: hnadle errors
impl WindowHandler {
    fn new(title: &str, (width, height): (u32,u32)) -> Result<Self>{
        let event_loop = EventLoop::new()?;
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width,height))
            //.with_fullscreen(Some(Fullscreen::Borderless(None)))
            .build(&event_loop)?;
    

        info!("Window successfully created");

        Ok(Self {event_loop, window})
    }
}

// May need to add error handling
//fn handle_events(event_loop: &mut EventLoop<()>, window: Arc<Window>) -> InstructionBuffer {
async fn handle_events(event_loop: &mut EventLoop<()>, window: &Window) -> Result<InstructionBuffer> {
    let mut buffer = InstructionBuffer::default();
    let timeout = Some(Duration::ZERO);
    let _status = event_loop.pump_events(timeout,  |event, elwt|   {
        match event {
            Event::WindowEvent { 
                event: WindowEvent::CloseRequested, 
                window_id,
            } if window_id == window.id() => {
                elwt.exit();
                buffer.push_back(Instruction::new(
                    LayerType::Control, 
                    Box::new(|control: &mut ControlHandler| {
                        control.quit()
                    }),
                ).unwrap());
            },
            Event::AboutToWait => {
                window.request_redraw();
            },
            Event::WindowEvent { event: WindowEvent::Resized(size), .. } => {
                // Handle window minimized
                if size.width == 0 || size.height == 0 {
                        buffer.push_back(Instruction::new(LayerType::Graphics, Box::new(|graphics: &mut GraphicsHandler| {
                            graphics.set_minisized(true);
                            InstructionBuffer::default()
                    })).unwrap())
                } else {
                    buffer.push_back(Instruction::new(LayerType::Graphics, Box::new(|graphics: &mut GraphicsHandler| {
                        graphics.set_resized();
                        graphics.set_minisized(false);
                        InstructionBuffer::default()
                    })).unwrap())
                }
                
            },
            _ => (),
        }
    });

    Ok(buffer)
    
}

