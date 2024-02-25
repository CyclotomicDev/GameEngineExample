//use winit::event_loop::{ControlFlow, EventLoop};
use control::control::{Buffer, Instruction, InstructionErr};
use std::{collections::VecDeque, fmt::Error, future::Future, process::Output, time::Duration};
use tokio::{task::JoinHandle, time::sleep};


#[tokio::main]
async fn main() -> Result<(), Error> {
    let mut app = App::new();

    loop {
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
                    retrieve_graphics_cycle(&mut graphics_loop);
                    retrieve_data_cycle(&mut data_loop);
                }
            },
            GameState::Exit => {
                println!("At state exit");
                break;
            },
        };
    }

    Ok(())

}

enum GameState {
    Start,
    Init,
    Menu,
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