#[allow(dead_code)]
use crate::Float;
pub struct Polynomial {
    pub coeffs: Vec<f64>,
}

impl Polynomial {
    pub fn new(coeffs: Vec<f64>) -> Self {
        Self { coeffs }
    }

    pub fn eval(&self, x: f64) -> f64 {
        self.coeffs
            .iter()
            .enumerate()
            .map(|(i, &c)| c * x.powi(i as i32))
            .sum()
    }

    pub fn derivative(&self) -> Self {
        let coeffs = self
            .coeffs
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, &c)| c * i as f64)
            .collect();
        Self { coeffs }
    }
}
