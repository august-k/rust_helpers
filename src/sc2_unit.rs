use pyo3::prelude::*;
use pyo3::types::PyAny;

#[derive(Clone, Debug)]
pub struct SC2Unit {
    pub position: (f32, f32)
}

impl<'source> FromPyObject<'source> for SC2Unit {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        let position: (f32, f32) = obj.getattr("position")?.extract()?;
        Ok(Self {
            position: position,
        })
    }
}