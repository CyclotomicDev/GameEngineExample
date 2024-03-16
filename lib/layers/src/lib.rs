use std::any::Any;

pub trait Layer: 'static {

}

pub trait AnyLayer: Any + Layer {
    fn as_any(&self) -> &dyn Any;

    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Any + Layer> AnyLayer for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}