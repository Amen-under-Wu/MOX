use crate::Float;

#[derive(Debug, Clone, Copy)]
pub struct MaterialData {
    pub sigtr: (Float, Float),
    pub sigma: (Float, Float),
    pub nusigf: (Float, Float),
    pub sigf: (Float, Float),
    pub sigr: Float,
}

impl MaterialData {
    pub fn new(
        sigtr: (Float, Float),
        sigma: (Float, Float),
        nusigf: (Float, Float),
        sigf: (Float, Float),
        sigr: Float,
    ) -> Self {
        Self {
            sigtr,
            sigma,
            nusigf,
            sigf,
            sigr,
        }
    }
    pub fn fuel() -> Self {
        Self {
            sigtr: (0.1665, 0.21745),
            sigma: (5.988e-4, 2.395e-3),
            nusigf: (1.4311e-4, 3.6931e-3),
            sigf: (5.6475e-5, 1.447e-3),
            sigr: 1.9215e-3,
        }
    }
}
