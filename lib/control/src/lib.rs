use anyhow::{Result, anyhow};
use std::{any::{Any, TypeId}, collections::VecDeque, sync::Arc};
use winit::{event::{Event, WindowEvent}, event_loop::EventLoop, platform::pump_events::EventLoopExtPumpEvents, window::Window};
use std::time::Duration;

use graphics::GraphicsHandler;
use state::StateHandler;
use data::DataHandler;


use layers::Layer;

type Grph = InstructionHelper<GraphicsHandler>;
type State = InstructionHelper<StateHandler>;
type Data = InstructionHelper<DataHandler>;
type Control = InstructionHelper<ControlHandler>;



pub enum Instruction {
    GrahpicsInst(Grph),
    StateInst(State),
    DataInst(Data),
    ControlInst(Control),
    //Null,
}

// Make static lifetime layers

impl Instruction {
    pub fn new<T: Layer>(layer_type: LayerType, func: Box<dyn Fn(&mut T) -> InstructionBuffer + Send + 'static>) -> Result<Self> {           
        match layer_type {
            LayerType::Graphics => {
                Ok(Instruction::GrahpicsInst(convert_layer_type::<GraphicsHandler>(Box::new(InstructionHelper::<T> { func }))?))
            }
            LayerType::State => {
                Ok(Instruction::StateInst(convert_layer_type::<StateHandler>(Box::new(InstructionHelper::<T> { func }))?))
            },
            LayerType::Data => {
                Ok(Instruction::DataInst(convert_layer_type::<DataHandler>(Box::new(InstructionHelper::<T> { func }))?))
            },
            LayerType::Control => {
                Ok(Instruction::ControlInst(convert_layer_type::<ControlHandler>(Box::new(InstructionHelper::<T> { func }))?))
            },
            //_ => Err(anyhow!("Instruction is not for valid layer type"))
        }   
        
    }

    // Execute consumes the instruction
    fn execute(self, layer: &mut dyn Any) -> Result<InstructionBuffer> {
    //fn execute(&self, layer: Box<&'a mut dyn Layer>) -> Result<InstructionBuffer> {
        match self {
            Instruction::GrahpicsInst(inst) => 
                inst.execute(layer
                    .downcast_mut::<GraphicsHandler>()
                    .ok_or_else(|| anyhow!("Instruction type mismatch"))?),
            Instruction::StateInst(inst) =>
                inst.execute(layer
                    .downcast_mut::<StateHandler>()
                    .ok_or_else(|| anyhow!("Instruction type mismatch"))?),
            Instruction::DataInst(inst) =>
                inst.execute(layer
                    .downcast_mut::<DataHandler>()
                    .ok_or_else(|| anyhow!("Instruction type mismatch"))?),
            Instruction::ControlInst(inst) =>
                inst.execute(layer
                    .downcast_mut::<ControlHandler>()
                    .ok_or_else(|| anyhow!("Instruction type mismatch"))?),
        }
    }
}

fn convert_layer_type<T: Layer>(instruction: Box<dyn Any>) -> Result<InstructionHelper<T>> {
    Ok(*instruction.downcast::<InstructionHelper<T>>().map_err(|_| anyhow!("Layer downcast error"))?)
    //InstructionHelper::<T> {func: Box::new(|_| {InstructionBuffer::default()})}
}

pub struct InstructionHelper<T: Layer> {
    func: Box<dyn Fn(&mut T) -> InstructionBuffer + Send>,
}

impl<T: Layer> InstructionHelper<T>
where
    T: Layer
{
    fn execute(self, operand: &mut T) -> Result<InstructionBuffer> {
        Ok((self.func)(operand))
    }
}




pub struct InstructionBuffer {
    buffer: VecDeque<Instruction>,
}

impl InstructionBuffer {
    pub fn new(buffer: Vec<Instruction>) -> Self {
        Self { buffer: VecDeque::from(buffer) }
    }

    pub fn push_back(&mut self, instruction: Instruction) {
        self.buffer.push_back(instruction);
    }

    fn combine(mut self, mut new: InstructionBuffer) -> Self {
        self.buffer.append(&mut new.buffer);
        self
    }

    pub fn execute_all(&mut self, self_boxed: &mut dyn Any) -> Result<InstructionBuffer> {
        Ok(
            self.buffer.drain(..)
                .map(move |instruction: Instruction | {
                    instruction.execute(self_boxed)
                })
                .map_while(Result::ok)
                .fold(InstructionBuffer::new(vec![]), |acc: InstructionBuffer, c: InstructionBuffer| acc.combine( c))
        )
        
    }

    /// Takes unsorted instructions and adds them to existing buffers
    pub fn sort(mut self, buffer_collection: &mut BufferColection) {
        self.buffer.drain(..)
            .for_each(|instruction| {
                match instruction {
                    Instruction::GrahpicsInst(_) => {
                        buffer_collection.graphics_buffer.buffer.push_back(instruction);
                    },
                    Instruction::StateInst(_) => {
                        buffer_collection.state_buffer.buffer.push_back(instruction);
                    },
                    Instruction::DataInst(_) => {
                        buffer_collection.data_buffer.buffer.push_back(instruction);
                    },
                    Instruction::ControlInst(_) => {
                        buffer_collection.control_buffer.buffer.push_back(instruction);
                    },
                }
            })

        // Need to handle size of buffers
    }
}

impl<'a> Default for InstructionBuffer {
    fn default() -> Self {
    Self { buffer: VecDeque::new() }
    }
}

#[derive(Default)]
pub struct BufferColection {
    pub graphics_buffer: InstructionBuffer,
    pub data_buffer: InstructionBuffer,
    pub state_buffer: InstructionBuffer,
    pub control_buffer: InstructionBuffer,
}

pub enum LayerType {
    Graphics,
    Data,
    State,
    Control,
}

pub struct ControlHandler {
    quit_recieved: bool,
}

impl ControlHandler {
    pub fn new() -> Self {
        Self {quit_recieved: false}
    }

    pub fn quit(&mut self) -> InstructionBuffer {
        self.quit_recieved = true;
        InstructionBuffer::default()
    }

    pub fn quit_recieved(&self) -> bool {
        self.quit_recieved
    }
}

impl Layer for ControlHandler {
    
}
