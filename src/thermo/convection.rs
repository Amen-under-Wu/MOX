use crate::Float;
use crate::matrix::{MyVec, SORSolver, SparseMatrix};
use crate::pde::{BorderCond2D, Grid2D, convec_mat_z, diffusion_eqn_rz};

pub struct FluidData {
    pub h: Float,
    pub cp: Float,
    pub lambda: Float,
    pub dmdt: Float,
}
impl FluidData {
    pub fn helium(d_pebble: Float, dmdt: Float) -> Self {
        const NU: Float = 200.0;
        let lambda = 0.3;
        let h = NU * lambda / d_pebble;
        Self {
            h,
            cp: 5.193e3,
            lambda,
            dmdt,
        }
    }
}

pub struct Convection {
    fluid: FluidData,
    grid: Grid2D,
}

impl Convection {
    pub fn new(fluid: FluidData, grid: Grid2D) -> Self {
        Self { fluid, grid }
    }
    fn convec_eqn(&self, t_in: Float, p: &MyVec, t_wall: &Vec<Float>) -> (SparseMatrix, MyVec) {
        // nabla*(lambda*nabla T)-c*u*dT/dz+p=0
        let (grid_rn, grid_zn) = self.grid.grid_n();
        let coeff_vec = vec![vec![self.fluid.lambda; grid_zn.len()]; grid_rn.len()];
        let bdr_r_vec = t_wall
            .iter()
            .map(|&t| (1.0, self.fluid.lambda / self.fluid.h, t))
            .collect();

        let rn = grid_rn.iter().sum();
        let zn = grid_zn.iter().sum();

        let mut src_vec = vec![vec![0.0; zn]; rn];
        for ri in 0..rn {
            for zi in 0..zn {
                let idx = self.grid.idx(ri, zi);
                src_vec[ri][zi] = p[idx];
            }
        }
        let brd_z = (
            BorderCond2D::Value(vec![t_in; zn]),
            BorderCond2D::Deriv(vec![0.0; zn]),
        );
        let eqn = diffusion_eqn_rz(
            &self.grid,
            &BorderCond2D::Comb(bdr_r_vec),
            &brd_z,
            &coeff_vec,
            &src_vec,
        );
        let convec_mat = convec_mat_z(&self.grid, self.fluid.dmdt * self.fluid.cp);
        (eqn.0 + convec_mat, eqn.1)
    }
    pub fn temp_solve(&self, t_in: Float, p: &MyVec, t_wall: &Vec<Float>) -> MyVec {
        let (a, b) = self.convec_eqn(t_in, p, t_wall);
        SORSolver::new(&a, &b, 1.5).solve().unwrap()
    }
}
