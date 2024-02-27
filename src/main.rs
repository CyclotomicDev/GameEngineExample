//use winit::event_loop::{ControlFlow, EventLoop};
use control::control::{Buffer, Instruction, InstructionErr};
use winit::platform::pump_events::{PumpStatus, EventLoopExtPumpEvents};
use std::{collections::VecDeque, fmt::Error, future::Future, process::Output, time::Duration};
use tokio::{task::JoinHandle, time::sleep};
use winit::dpi::LogicalSize;
use winit::event_loop::{self, ControlFlow, EventLoop};
use winit::event::{Event, WindowEvent};
use winit::window::{Window, WindowBuilder};

#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut app = App::new();

    let mut window_handler = WindowHandler::new("Project G03Alpha", (1200,800));

    

    'main: loop {
        // Goes through state buffer to get current state
        app.state.process_state(&mut app.state_buffer);

        // Chooses action based on current game state
        match app.state {
            GameState::Start => {
                println!("At state start");
                app.state_buffer.push_back(StateChange::new(Box::new(|x| {*x = GameState::Init; Ok(())})));
            },
            GameState::Init => { // Initializes graphics

                app.state_buffer.push_back(StateChange::new(Box::new(|x| {*x = GameState::Menu; Ok(())})));
            },
            GameState::Menu => { // Main menu - controls all main paths
                
                let mut graphics_loop : Option<JoinHandle<()>> = None;
                let mut data_loop : Option<JoinHandle<()>> = None;
                
                loop {
                    let timeout = Some(Duration::ZERO);
                    let status = window_handler.event_loop.pump_events(timeout, |event, elwt| {
                        match event {
                            Event::WindowEvent { 
                                event: WindowEvent::CloseRequested, 
                                window_id,
                            } if window_id == window_handler.window.id() => elwt.exit(),
                            Event::AboutToWait => {
                                window_handler.window.request_redraw();
                            },
                            _ => (),
                        }
                    });

                    if let PumpStatus::Exit(exit_code) = status {
                        println!("Close window; Exit code {}",exit_code);
                        break 'main;
                    }

                    retrieve_graphics_cycle(&mut graphics_loop);
                    retrieve_data_cycle(&mut data_loop);
                }
            },
            GameState::Exit => {
                println!("At state exit");
                break;
            },
            GameState::Game => { //Holds data for normal running of the game
                let mut graphics_loop : Option<JoinHandle<()>> = None;
                let mut data_loop : Option<JoinHandle<()>> = None;
                
                loop {
                    retrieve_graphics_cycle(&mut graphics_loop);
                    retrieve_data_cycle(&mut data_loop);
                }
            }
        };
    }

    Ok(())

}

enum GameState {
    Start,
    Init,
    Menu,
    Game,
    Exit,
}

impl GameState {
    fn new() -> Self {
        Self::Start
    }

    fn process_state(&mut self, buffer: &mut Buffer<StateChange>) -> Result<(), InstructionErr> {
        buffer
            .iter()
            .for_each(|x| {x.execute(self);});
        Ok(())
    }
}

struct StateChange {
    function: Box<dyn Fn(&mut GameState) -> Result<(), InstructionErr>>,
}

impl StateChange {
    fn new(function: Box<dyn Fn(&mut GameState) -> Result<(), InstructionErr>>) -> Self {
        Self {function}
    } 
}

impl Instruction for StateChange {
    type Operand = GameState;

    fn execute(&self, operand: &mut GameState) -> Result<(),InstructionErr> {
        self.check_requirements()?;
        (self.function)(operand)?;
        Ok(())
    }

    fn check_requirements(&self) -> Result<(),InstructionErr> {
        Ok(())
    }
}

struct App {
    graphics: Option<Graphics>,
    data: Option<Data>,
    network: Option<Network>,
    state: GameState,
    state_buffer: Buffer::<StateChange>,
}

struct  Graphics;


impl Graphics {
    
}

struct  Data;
struct Network;

impl App {
    fn new() -> Self {
        let graphics = None;
        let data = None;
        let network = None;
        let state = GameState::new();
        let state_buffer = Buffer::<StateChange>::new();
        //let event_loop = EventLoop::new();
        Self {graphics,data,network,state,state_buffer}
    }
}

struct Settings {


}

fn retrieve_graphics_cycle(cycle: &mut Option<JoinHandle<()>>) {

    if let Some(link) = cycle {
        if !link.is_finished() {
            // No new cycle created
            return;
        }
    }

    // New cycle must be created
    let graphics_fn = async {
        sleep(Duration::from_millis(1000)).await;
                    println!("1 sec");
    };

    *cycle = Some(tokio::spawn(graphics_fn));
}


fn retrieve_data_cycle(cycle: &mut Option<JoinHandle<()>>) {

    if let Some(link) = cycle {
        if !link.is_finished() {
            // No new cycle created
            return;
        }
    }

    // New cycle must be created
    let data_fn = async {
        sleep(Duration::from_millis(500)).await;
                    println!("0.5 sec");
    };

    *cycle = Some(tokio::spawn(data_fn));
}


/// Contains all information about window control
struct WindowHandler {
    event_loop: EventLoop<()>,
    window: Window,
}

/// To do: hnadle errors
impl WindowHandler {
    fn new(title: &str, (width, height): (u32,u32)) -> Self{
        let event_loop = EventLoop::new().unwrap();
        let window = WindowBuilder::new()
            .with_title(title)
            .with_inner_size(LogicalSize::new(width,height))
            .build(&event_loop).unwrap();

        Self {event_loop, window}
    }
}