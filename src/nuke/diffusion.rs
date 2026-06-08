use crate::Float;
use crate::matrix::{MyVec, SORSolver, SparseMatrix};
use crate::nuke::material::MaterialData;
use crate::pde::{BorderCond2D, Grid2D, diffusion_eqn_rz};

pub struct Diffusion {
    material: Vec<Vec<MaterialData>>,
    grid: Grid2D,
}

impl Diffusion {
    pub fn new(material: Vec<Vec<MaterialData>>, grid: Grid2D) -> Self {
        Self { material, grid }
    }
    fn fast_group_eqn(&self, thermo_group: &MyVec) -> (SparseMatrix, MyVec) {
        let d: Vec<Vec<Float>> = self
            .material
            .iter()
            .map(|row| row.iter().map(|mat| 1.0 / (mat.sigtr.0 * 3.0)).collect())
            .collect();
        let (grid_rn, grid_zn) = self.grid.grid_n();
        let mut bdr_r_vec = Vec::new();
        let mut bdr_z_vec = (Vec::new(), Vec::new());
        for i in 0..grid_rn.len() {
            bdr_r_vec.extend(vec![
                (
                    1.0,
                    self.material[i][self.material[i].len() - 1].sigtr.0 / 0.7104,
                    0.0
                );
                grid_rn[i]
            ]);
        }
        for i in 0..grid_zn.len() {
            bdr_z_vec.0.extend(vec![
                (1.0, -self.material[0][i].sigtr.0 / 0.7104, 0.0);
                grid_zn[i]
            ]);
            bdr_z_vec.1.extend(vec![
                (
                    1.0,
                    self.material[self.material.len() - 1][i].sigtr.0 / 0.7104,
                    0.0
                );
                grid_zn[i]
            ]);
        }
        let rn = grid_rn.iter().sum();
        let zn = grid_zn.iter().sum();
        let mut add_data = Vec::new();
        let mut src_vec = vec![vec![0.0; zn]; rn];
        for ri in 0..rn {
            for zi in 0..zn {
                let idx = self.grid.idx(ri, zi);
                let coarse = self.grid.to_coarse(ri, zi);
                let mat = &self.material[coarse.0][coarse.1];
                src_vec[ri][zi] = thermo_group[idx] * mat.nusigf.1;
                add_data.push((idx, idx, mat.nusigf.0 - mat.sigtr.0));
            }
        }
        let add_mat = SparseMatrix::new_with_data(rn * zn, rn * zn, add_data);
        let res = diffusion_eqn_rz(
            &self.grid,
            &BorderCond2D::Comb(bdr_r_vec),
            &(
                BorderCond2D::Comb(bdr_z_vec.0),
                BorderCond2D::Comb(bdr_z_vec.1),
            ),
            &d,
            &src_vec,
        );
        (res.0 + add_mat, res.1)
    }
    fn thermo_group_eqn(&self, fast_group: &MyVec) -> (SparseMatrix, MyVec) {
        let d: Vec<Vec<Float>> = self
            .material
            .iter()
            .map(|row| row.iter().map(|mat| 1.0 / (mat.sigtr.1 * 3.0)).collect())
            .collect();
        let (grid_rn, grid_zn) = self.grid.grid_n();
        let mut bdr_r_vec = Vec::new();
        let mut bdr_z_vec = (Vec::new(), Vec::new());
        for i in 0..grid_rn.len() {
            bdr_r_vec.extend(vec![
                (
                    1.0,
                    self.material[i][self.material[i].len() - 1].sigtr.1 / 0.7104,
                    0.0
                );
                grid_rn[i]
            ]);
        }
        for i in 0..grid_zn.len() {
            bdr_z_vec.0.extend(vec![
                (1.0, -self.material[0][i].sigtr.1 / 0.7104, 0.0);
                grid_zn[i]
            ]);
            bdr_z_vec.1.extend(vec![
                (
                    1.0,
                    self.material[self.material.len() - 1][i].sigtr.1 / 0.7104,
                    0.0
                );
                grid_zn[i]
            ]);
        }
        let rn = grid_rn.iter().sum();
        let zn = grid_zn.iter().sum();
        let mut add_data = Vec::new();
        let mut src_vec = vec![vec![0.0; zn]; rn];
        for ri in 0..rn {
            for zi in 0..zn {
                let idx = self.grid.idx(ri, zi);
                let coarse = self.grid.to_coarse(ri, zi);
                let mat = &self.material[coarse.0][coarse.1];
                src_vec[ri][zi] = fast_group[idx] * mat.sigr;
                add_data.push((idx, idx, -mat.sigtr.1));
            }
        }
        let add_mat = SparseMatrix::new_with_data(rn * zn, rn * zn, add_data);
        let res = diffusion_eqn_rz(
            &self.grid,
            &BorderCond2D::Comb(bdr_r_vec),
            &(
                BorderCond2D::Comb(bdr_z_vec.0),
                BorderCond2D::Comb(bdr_z_vec.1),
            ),
            &d,
            &src_vec,
        );
        (res.0 + add_mat, res.1)
    }
    pub fn power_src(&self, fast_group: &MyVec, thermo_group: &MyVec, ef: Float) -> MyVec {
        let rn = self.grid.grid_n().0.iter().sum();
        let zn = self.grid.grid_n().1.iter().sum();
        let mut res = vec![0.0; rn * zn];
        for ri in 0..rn {
            for zi in 0..zn {
                let idx = self.grid.idx(ri, zi);
                let coarse = self.grid.to_coarse(ri, zi);
                let mat = &self.material[coarse.0][coarse.1];
                res[idx] = (fast_group[idx] * mat.sigf.0 + thermo_group[idx] * mat.sigf.1) * ef;
            }
        }
        MyVec(res)
    }
    pub fn fast_group_flux(&self, thermo_group: &MyVec) -> MyVec {
        let (mat, src) = self.fast_group_eqn(thermo_group);
        let solver = SORSolver::new(&mat, &src, 1.5);
        solver.solve().unwrap()
    }
    pub fn thermo_group_flux(&self, fast_group: &MyVec) -> MyVec {
        let (mat, src) = self.thermo_group_eqn(fast_group);
        let solver = SORSolver::new(&mat, &src, 1.5);
        solver.solve().unwrap()
    }
}
