use layers::Layer;

pub enum StateHandler {
    Start,
    Init,
    Menu,
    Game,
    Exit,
}

impl StateHandler {
    pub fn new() -> Self {
        Self::Start
    }
}

impl Layer for StateHandler {
    
}