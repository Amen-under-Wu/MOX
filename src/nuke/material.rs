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
            sigtr: (16.65, 21.745),
            sigma: (5.988e-2, 2.395e-1),
            nusigf: (1.4311e-2, 3.6931e-1),
            sigf: (5.6475e-1, 1.447e-2),
            sigr: 1.9215e-1,
        }
    }
    pub fn carbon() -> Self {
        const SIGTR: Float = 41.0 * 17.0 / 18.0;
        Self {
            sigtr: (SIGTR, SIGTR),
            sigma: (0.0, 2.9e-2),
            nusigf: (0.0, 0.0),
            sigf: (0.0, 0.0),
            sigr: 20.0,
        }
    }
}
