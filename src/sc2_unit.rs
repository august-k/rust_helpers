use pyo3::prelude::*;
use pyo3::types::PyAny;

#[derive(Clone, Copy, Debug)]
pub struct SC2Unit {
    pub position: (f32, f32),
    pub tag: u64,
    pub health_percentage: f32,
    pub distance_calculation_index: u64,
}

impl<'source> FromPyObject<'source> for SC2Unit {
    fn extract(obj: &'source PyAny) -> PyResult<Self> {
        Ok(Self {
            position: obj.getattr("position")?.extract()?,
            tag: obj.getattr("tag")?.extract()?,
            health_percentage: obj.getattr("health_percentage")?.extract()?,
            distance_calculation_index: obj.getattr("distance_calculation_index")?.extract()?,
        })
    }
}
