//use winit::event_loop::{ControlFlow, EventLoop};
use control::*;
use graphics::GraphicsHandler;
use data::DataHandler;
use control::ControlHandler;
use layers::{AnyLayer, Layer};
use state::StateHandler;
use tokio::sync::mpsc::error::SendError;
use winit::platform::pump_events::{PumpStatus, EventLoopExtPumpEvents};
use std::error::Error;
use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;
use std::task::Poll;
use std::mem;
use std::time::Duration;
use tokio::{task::JoinHandle, time::sleep, sync::mpsc, join};
use tokio::sync::{Mutex, RwLock};
use std::sync::{Arc, atomic::{AtomicBool, Ordering::Relaxed}};
use winit::dpi::LogicalSize;
use winit::event_loop::{self, EventLoop};
use winit::event::{Event, WindowEvent};
use winit::window::{self, Window, WindowBuilder};
use anyhow::Result;
use log::*;
use std::any::Any;

//#[tokio::main(flavor = "current_thread")]
#[tokio::main]
async fn main() -> Result<()> {
    
    simple_logging::log_to_file("test.log", LevelFilter::Info)?;


    //let mut app = App::new();

    /*

    let mut window_handler = WindowHandler::new("Project G03Alpha", (1200,800))?;

    let mut graphics: Arc<Mutex<Option<Box<GraphicsHandler>>>> = Arc::new(Mutex::new(None));
    let mut data: Arc<Mutex<Option<Box<DataHandler>>>> = Arc::new(Mutex::new(None));
    //let mut data: Option<Data> = None;

    //let mut current_state: Box<GameState> = Box::new(GameState::new());
    let mut current_state: StateHandler = StateHandler::new();
    let (state_tx, mut state_rx) = mpsc::channel::<InstructionBuffer>(1);
    {state_tx.send(InstructionBuffer::default()).await};
    */

    let mut window_handler = WindowHandler::new("Project G03Alpha", (1200,800))?;

    let graphics = Arc::new(Mutex::new(GraphicsHandler::new(&window_handler.window)?));
    let data = Arc::new(Mutex::new(DataHandler::new()?));
    let mut control = ControlHandler::new();

    let graphics_future = graphics_cycle(InstructionBuffer::default(), &window_handler.window, graphics.clone());
    let data_future = data_cycle(InstructionBuffer::default(), data.clone());
    tokio::pin!(graphics_future);
    tokio::pin!(data_future);

    let mut buffer_collection = BufferColection::default(); 
    

    'main: loop {

        handle_events(&mut window_handler.event_loop, &window_handler.window).await?
            .sort(&mut buffer_collection);

        buffer_collection.control_buffer.execute_all(control.as_any_mut())?;

        if control.quit_recieved() {
            break 'main;
        }
        
        tokio::select! {
            current_graphics_buffer = &mut graphics_future => {
                graphics_future.set(graphics_cycle(InstructionBuffer::default(), &window_handler.window, graphics.clone()))
            }
            _ = &mut data_future => {
                data_future.set(data_cycle(InstructionBuffer::default(), data.clone()))
            }
        }




        {
            /*
        // Goes through state buffer to get current state
        
        //current_state.execute_all(state_rx.recv().await.unwrap());
        state_rx.recv().await.unwrap().execute_all(current_state.as_any_mut())?;
        // Chooses action based on current game state
        match current_state {
            StateHandler::Start => {
                info!("At state start");
                
                // Instruction to change state, returning no additional instructions
                let state_instructions = InstructionBuffer::new(vec![
                    Instruction::new(
                        LayerType::State,
                        Box::new(|operand: &mut StateHandler| {
                            *operand = StateHandler::Init;
                            InstructionBuffer::default()
                    }))?,
                ]);
                state_tx.send(state_instructions).await;
            },
            StateHandler::Init => { // Initializes graphics
                let window = window_handler.window.as_ref();
                let mut graphics = graphics.lock().await;
                *graphics = Some(Box::new(GraphicsHandler::new(&window)?));
                info!("Successful loading of Vulkan");

                // Empty data init
                let mut data = data.lock().await;
                *data = Some(Box::new(DataHandler::new()?));
                
                let state_instructions = InstructionBuffer::new(vec![
                    Instruction::new(
                        LayerType::State,
                        Box::new(|operand: &mut StateHandler| {
                            *operand = StateHandler::Menu;
                            InstructionBuffer::default()
                    }))?,
                ]);
                state_tx.send(state_instructions).await;
            },
            StateHandler::Menu => { // Main menu - controls all main paths

                {
                    let continue_loop_master = Arc::new(AtomicBool::new(true));
                    
                    let mut graphics = graphics.clone();
                    let mut data = data.clone();

                    let window = window_handler.window.clone();

                    let grph_window = window.clone();
                    let ctrl_window = window.clone();

                    // Channel from All to Control
                    let (ctrl_tx, mut ctrl_rx) = mpsc::channel::<InstructionBuffer>(10);

                    // Channel from Control to Grahpics
                    let (grph_tx, mut grph_rx) = mpsc::channel::<InstructionBuffer>(1);
                    
                    let continue_loop = continue_loop_master.clone();
                    //let mut graphics = graphics.as_mut().unwrap();
                    let graphics_loop = async move {
                        println!("Graphics loop start");
                        let mut graphics = graphics.lock().await;
                        let graphics = graphics.as_mut().unwrap();
                        
                        while continue_loop.load(Relaxed) {
                            // Control
                            let mut crtl_instructions = grph_rx.recv().await.unwrap();

                            
                            
                            // Execute instructions
                            crtl_instructions.execute_all(graphics.as_any_mut());
                            
                            

                            // Execute graphics
                            graphics.render(&grph_window);
                        }

                        ()
                    };

                    let continue_loop = continue_loop_master.clone();
                    let (data_tx, mut data_rx) = mpsc::channel::<InstructionBuffer>(1);
                    let data_loop = async move {
                        println!("Data loop start");
                        let mut data = data.lock().await;
                        let data = data.as_mut().unwrap();
                        while continue_loop.load(Relaxed) {
                            // Control
                            let mut data_instructions = data_rx.recv().await.unwrap();


                            // Execute instructions
                            data_instructions.execute_all(data.as_any_mut());

                            // Execute data
                            println!("1s");
                            sleep(Duration::from_secs(1)).await;
                        }

                        ()
                    };
                    let (inpt_tx, mut inpt_rx) = mpsc::channel::<InstructionBuffer>(1);

                    let continue_loop = continue_loop_master.clone();

                    let state_tx = state_tx.clone();

                    // Schedules all instructions and timing
                    let control_loop = async move {

                        let mut buffer_collection = BufferColection::default();
                        let state_tx = state_tx.clone();
                        
                        while continue_loop.load(Relaxed) {
                            // Wait for new InstructionBuffer
                            let mut current_buffer = match ctrl_rx.try_recv() {
                                Ok(buffer) => buffer,
                                Err(_) => InstructionBuffer::default(),
                            };


                            // Seperate into layer-specific instructions
                            current_buffer.sort(&mut buffer_collection);

                            // Handle giving instructions to each layer
                            grph_tx.send(mem::take(&mut buffer_collection.graphics_buffer)).await;
                            state_tx.send(mem::take(&mut buffer_collection.state_buffer)).await;
                            data_tx.send(mem::take(&mut buffer_collection.state_buffer)).await;

                        }
                    };

                    let graphics_loop = tokio::spawn(graphics_loop);
                    let data_loop = tokio::spawn(data_loop);
                    let control_loop = tokio::spawn(control_loop);

                    let continue_loop = continue_loop_master.clone();



                    // Handle window events
                    while continue_loop.load(Relaxed) {
                        ctrl_tx.send(handle_events(&mut window_handler.event_loop, window_handler.window.clone())).await;
                        println!("Handle events");
                        /*
                        if let PumpStatus::Exit(exit_code) = status {
                            info!("Close window; Exit code {}",exit_code);
                            let state_instructions = InstructionBuffer::new(vec![
                                Instruction::new(
                                    LayerType::State,
                                    Box::new(|operand: &mut StateHandler| {
                                        *operand = StateHandler::Init;
                                        InstructionBuffer::default()
                                }))?,
                            ]);
                            state_tx.send(state_instructions);
                            break;
                        }
                        */
                    }

                    join!(graphics_loop, data_loop, control_loop);
                }

                // Handle state transfer
                
            },
            StateHandler::Exit => {
                trace!("At state exit");

                break 'main;
            },
            StateHandler::Game => { //Holds data for normal running of the game
                let mut graphics_loop : Option<JoinHandle<()>> = None;
                let mut data_loop : Option<JoinHandle<()>> = None;
                
                loop {
                    todo!();
                }
            }
        };
        */
    }
    }

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
    window: Arc<Window>,
}

/// To do: hnadle errors
impl WindowHandler {
    fn new(title: &str, (width, height): (u32,u32)) -> Result<Self>{
        let event_loop = EventLoop::new()?;
        let window = Arc::new(WindowBuilder::new()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width,height))
            .build(&event_loop)?);
    

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

