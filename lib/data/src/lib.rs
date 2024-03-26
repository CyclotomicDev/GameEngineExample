use anyhow::Result;
use layers::Layer;

pub struct DataHandler {

}

impl DataHandler {
    pub fn new() -> Result<Self> {
        Ok(Self{})
    }
}

impl Layer for DataHandler {
    
}