use crate::Float;
use crate::poly::Polynomial;

pub struct Section {
    group: i32,
    sigtr: Float,
    sigma: Float,
    nusigf: Float,
    rvau: Float,
    sigf: Float,
    sigr: Float,
}
impl Section {
    pub fn new(
        group: i32,
        sigtr: Float,
        sigma: Float,
        nusigf: Float,
        rvau: Float,
        sigf: Float,
        sigr: Float,
    ) -> Self {
        Self {
            group,
            sigtr,
            sigma,
            nusigf,
            rvau,
            sigf,
            sigr,
        }
    }
}

pub struct CouplingData {
    mod_temp: Float,
    fuel_temp: Float,
}
impl CouplingData {
    pub fn new(mod_temp: Float, fuel_temp: Float) -> Self {
        Self {
            mod_temp,
            fuel_temp,
        }
    }
}
pub struct CoupCoeff {
    pub default: CouplingData,
    pub coeff_mt: Polynomial,
    pub coeff_ft: Polynomial,
}
impl CoupCoeff {
    pub fn new(default: CouplingData, coeff_mt: Polynomial, coeff_ft: Polynomial) -> Self {
        Self {
            default,
            coeff_mt,
            coeff_ft,
        }
    }
    pub fn eval(&self, data: &CouplingData) -> Float {
        self.coeff_mt.eval(data.mod_temp - self.default.mod_temp)
            + self.coeff_ft.eval(data.fuel_temp - self.default.fuel_temp)
    }
}
