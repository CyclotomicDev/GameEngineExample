pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}


mod control {
    enum InstructionErr {

    }

    trait Instrustion {
        fn execute(&self) -> Result<(),InstructionErr>;

        fn check_rquirements(&self) -> bool;
        
    }
}