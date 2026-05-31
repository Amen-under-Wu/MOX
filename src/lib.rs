pub mod matrix;
pub mod ode;
pub mod pde;

pub mod nuke;
pub mod thermo;

pub type Float = f64;

#[cfg(test)]
mod tests {
    use super::*;
}
