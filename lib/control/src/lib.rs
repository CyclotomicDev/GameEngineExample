pub mod control {

    use anyhow::{Result, anyhow};
    use std::{any::{Any, TypeId}, collections::VecDeque};

    use graphics::GraphicsHandler;
    use state::StateHandler;

    use layers::Layer;

    /*
    impl Layer for GraphicsHandler {
    
    }
    */

    type Grph = InstructionHelper<GraphicsHandler>;
    type State = InstructionHelper<StateHandler>;

   

    pub enum Instruction {
        GrahpicsInst(Grph),
        StateInst(State),
        //Null,
    }

    // Make static lifetime layers

    impl Instruction {
        pub fn new<T: Layer>(layer_type: LayerType, func: Box<dyn Fn(&mut T) -> InstructionBuffer + Send + 'static>) -> Result<Self> {           
            match layer_type {
                LayerType::Graphics => {
                    Ok(Instruction::GrahpicsInst(convert_layer_type::<GraphicsHandler>(Box::new(InstructionHelper::<T> { func }))))
                }
                LayerType::State => {
                    Ok(Instruction::StateInst(convert_layer_type::<StateHandler>(Box::new(InstructionHelper::<T> { func }))))
                },
                _ => Err(anyhow!("Instruction is not for valid layer type"))
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
            }
        }
    }

    fn convert_layer_type<T: Layer>(instruction: Box<dyn Any>) -> InstructionHelper<T> {
        *instruction.downcast::<InstructionHelper<T>>().unwrap()
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

        fn combine(&mut self, mut new: InstructionBuffer) {
            self.buffer.append(&mut new.buffer);
        }

        pub fn execute_all(&mut self, self_boxed: &mut dyn Any) -> Result<InstructionBuffer> {
            let mut result: InstructionBuffer = InstructionBuffer::default();
            while !self.buffer.is_empty() {
                let instruction = self.buffer.pop_front().unwrap();
                let buffer_segment = instruction.execute(self_boxed)?;
                result.combine(buffer_segment);
            }
            Ok(result)
            /*
            Ok(
                self.buffer.drain(..)
                    .map(move |instruction: Instruction<'_> | {
                        instruction.execute(self_boxed)
                    })
                    .map_while(Result::ok)
                    .fold(InstructionBuffer::new(vec![]), |acc: InstructionBuffer, c: InstructionBuffer| acc.combine( c))
            )
            */
        }
    }

    impl<'a> Default for InstructionBuffer {
        fn default() -> Self {
        Self { buffer: VecDeque::new() }
        }
    }
    
    /*
    pub fn execute_all<'a, 'b: 'a>(self_boxed: &'a mut dyn Any, buffer: InstructionBuffer<'b>) -> Result<InstructionBuffer<'a>> {
        Ok(
            buffer.buffer.iter()
                .map(|instruction | {
                    instruction.execute(self_boxed)
                    //InstructionBuffer::default()
                })
                .map_while(Result::ok)
                .fold(InstructionBuffer::new(vec![]), |acc: InstructionBuffer, c: InstructionBuffer| acc.combine( c))
        )
    }
    */

    pub enum LayerType {
        Graphics,
        Data,
        State,
        Control,
    }

}