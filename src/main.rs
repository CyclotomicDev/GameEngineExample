//use winit::event_loop::{ControlFlow, EventLoop};
use control::control::*;
use graphics::GraphicsHandler;
use layers::{AnyLayer, Layer};
use state::StateHandler;
use tokio::sync::mpsc::error::SendError;
use winit::platform::pump_events::{PumpStatus, EventLoopExtPumpEvents};
use std::error::Error;
use std::future::Future;
use std::ops::Deref;
use std::pin::Pin;
use std::task::Poll;
use std::time::Duration;
use tokio::{task::JoinHandle, time::sleep, sync::mpsc, join};
use tokio::sync::{Mutex, RwLock};
use std::sync::Arc;
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


    let mut app = App::new();

    let mut window_handler = WindowHandler::new("Project G03Alpha", (1200,800))?;

    let mut graphics: Arc<Mutex<Option<Box<GraphicsHandler>>>> = Arc::new(Mutex::new(None));
    let mut data: Option<Data> = None;

    //let mut current_state: Box<GameState> = Box::new(GameState::new());
    let mut current_state: StateHandler = StateHandler::new();
    let (state_tx, mut state_rx) = mpsc::channel::<InstructionBuffer>(1);
    {state_tx.send(InstructionBuffer::default()).await};

    

    'main: loop {
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
                    let mut graphics = graphics.clone();

                    let window = window_handler.window.clone();

                    let grph_window = window.clone();
                    let ctrl_window = window.clone();

                    // Channel from All to Control
                    let (ctrl_tx, ctrl_rx) = mpsc::channel::<InstructionBuffer>(10);

                    // Channel from Control to Grahpics
                    let (grph_tx, mut grph_rx) = mpsc::channel::<InstructionBuffer>(1);
                    
                    //let mut graphics = graphics.as_mut().unwrap();
                    let graphics_loop = async move {
                        println!("Graphics loop start");
                        let mut graphics = graphics.lock().await;
                        let graphics = graphics.as_mut().unwrap();
                        loop {
                            // Control
                            let mut crtl_instructions = grph_rx.recv().await.unwrap();

                            
                            
                            // Execute instructions
                            
                            crtl_instructions.execute_all(graphics.as_any_mut());
                            
                            

                            // Execute graphics
                            graphics.render(&grph_window);
                        }

                        ()
                    };

                    let (data_in, mut data_out) = mpsc::channel::<InstructionBuffer>(1);
                    let data_loop = async {
                        println!("Data loop start");
                        loop {
                            // Control

                            // Execute data
                            println!("1s");
                            sleep(Duration::from_secs(1)).await;
                        }

                        ()
                    };

                    // Schedules all instructions and timing
                    let control_loop = async move {
                        
                        todo!();
                    };

                    let graphics_loop = tokio::spawn(graphics_loop);
                    let data_loop = tokio::spawn(data_loop);
                    //let control_loop = tokio::spawn(control_loop);

                    // Handle window events
                    loop {
                        let timeout = Some(Duration::ZERO);
                        let status = window_handler.event_loop.pump_events(timeout, |event, elwt| {
                            match event {
                                Event::WindowEvent { 
                                    event: WindowEvent::CloseRequested, 
                                    window_id,
                                } if window_id == window.id() => elwt.exit(),
                                Event::AboutToWait => {
                                    window_handler.window.request_redraw();
                                },
                                _ => (),
                            }
                        });

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
                    }
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
    }

    Ok(())

}

struct App {
    
    network: Option<Network>,
}

struct  Graphics;


impl Graphics {
    
}

struct  Data;
struct Network;

impl App {
    fn new() -> Self {
        let network = None;
        Self {network}
    }
}

struct Settings {


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