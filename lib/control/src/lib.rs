


pub mod control {

    pub enum InstructionErr {
        Invalid,
        Decay, //Instruction no longer valid
        Failure, //Instruction tried and failed to be executed
    }


    ///Basic unit for communicating with other crates
    pub trait Instruction {
        type Operand;

        fn execute(&self, operand: &mut <Self as Instruction>::Operand) -> Result<(),InstructionErr>;

        fn check_requirements(&self) -> Result<(),InstructionErr>;

    }
/*
    ///Buffer that handles execution of all contained instructions
    struct InstrtuctionBuffer {
        instructions: Vec<Box<dyn Instruction>>,
    }

    impl Instruction for InstrtuctionBuffer {
        fn execute(&self) -> Result<(),InstructionErr> {
            self.instructions.iter().for_each(|x| {x.execute();});
            Ok(())
        }

        fn check_rquirements(&self) -> bool {
            true
        }
    }

    //mod _helper {
        #[derive(Hash, PartialEq, Eq)]
        enum Location {
            All,
            Indentity, //To same
            Control,
            Data,
            Graphics,
        }
        ///Determines valid route to send instructions
        struct Relation {
            destination: Location,

        }

        impl Relation {
            fn new(destination: Location) -> Self {
                Self {destination}
            }
        }
    //}

    struct ExternalTransferRules {
        transfers: HashMap<Location,Relation>,
    }

    impl ExternalTransferRules {
        fn new() -> Self {
            let mut transfers = HashMap::<Location,Relation>::new();

            use Location::*;

            transfers.insert(Control,Relation::new(All));

            Self {transfers}
        }
    }

    enum InstructionWrapper {
        None,
        Data(Box<dyn Instruction>),
        Graphics(Box<dyn Instruction>),
    }

    struct Node {
        buffer: VecDeque<Box<dyn Instruction>>,
    }

    impl Node {
        fn new() -> Self {
            let buffer = VecDeque::new();
            Self {buffer}
        }
    }

    use std::collections::VecDeque;
    struct ControlHandler {
        data_buffer: Node,
        graphics_buffer: Node,
    }

    impl ControlHandler {
        fn new() -> Self {
            let data_buffer = Node::new();
            let graphics_buffer = Node::new();
            Self {data_buffer, graphics_buffer}
        }

        fn add_instruction(&mut self, instruction: InstructionWrapper) -> Result<(), InstructionErr> {
            match instruction {
                InstructionWrapper::Data(inner) => {
                    self.data_buffer.push_back(inner);
                },
                InstructionWrapper::Graphics(inner) => {
                    self.graphics_buffer.push_back(inner);
                },
                _ => _,
            };
            Ok(())
        }

        fn organisze_buffers(&mut self) {

        }
    }
*/
use std::collections::VecDeque;

    pub struct Buffer<T: Instruction>{
        pub data: VecDeque<Box<T>>,
    }

    impl<T: Instruction> Buffer<T> {
        pub fn new() -> Self {
            let data = VecDeque::new();
            Self {data}
        }

        pub fn push_back(&mut self, value: T) {
            self.data.push_back(Box::<T>::new(value));
        }

        pub fn pop_front(&mut self) -> Option<T> {
            match self.data.pop_front()
            {
                Some(inner) => Some(*inner),
                None => None,
            }
        }

        pub fn iter(&self) -> impl Iterator<Item = &Box<T>> {
            self.data.iter()
        }
    }

    pub enum ModuleType {
        Status,
        Data,
        Graphics,
    }

}