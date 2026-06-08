use crate::{
    Float,
    matrix::{MyVec, SparseMatrix},
};

#[derive(Debug, Clone)]
struct Grid1D {
    bdr: Float,
    grid: Vec<(usize, Float)>,
    dx: Vec<Float>,
    n: usize,
    coord: Vec<Float>,
    grid_idx: Vec<usize>,
}
impl Grid1D {
    fn new(bdr: Float, grid: Vec<(usize, Float)>) -> Self {
        let n = grid.iter().map(|x| x.0).sum();
        let (mut coord, mut grid_idx) = (Vec::new(), Vec::new());
        let mut x1 = bdr;
        let mut dx = Vec::new();
        for i in 0..grid.len() {
            let (n, x2) = grid[i];
            coord.extend((0..n).map(|xx| x1 + xx as Float * (x2 - x1) / n as Float));
            grid_idx.extend(std::iter::repeat(i).take(n));
            dx.push((x2 - x1) / n as Float);
            x1 = x2;
        }
        coord.push(x1);
        Self {
            bdr,
            grid,
            dx,
            n,
            coord,
            grid_idx,
        }
    }
    fn x2idx(&self, x: Float) -> usize {
        for i in 0..self.grid.len() {
            if x < self.grid[i].1 {
                return i;
            }
        }
        self.grid.len()
    }
}
#[derive(Debug, Clone)]
pub struct Grid2D {
    grid: (Grid1D, Grid1D),
}
impl Grid2D {
    pub fn new(
        x: Float,
        grid_x: Vec<(usize, Float)>,
        y: Float,
        grid_y: Vec<(usize, Float)>,
    ) -> Self {
        Self {
            grid: (Grid1D::new(x, grid_x), Grid1D::new(y, grid_y)),
        }
    }
    #[inline]
    pub fn idx(&self, i: usize, j: usize) -> usize {
        i * self.grid.1.n + j
    }
    pub fn idx_inv(&self, idx: usize) -> (usize, usize) {
        (idx / self.grid.0.n, idx % self.grid.0.n)
    }
    pub fn to_coarse(&self, i: usize, j: usize) -> (usize, usize) {
        (self.grid.0.grid_idx[i], self.grid.1.grid_idx[j])
    }
    pub fn xy2idx(&self, x: Float, y: Float) -> usize {
        self.idx(self.grid.0.x2idx(x), self.grid.1.x2idx(y))
    }
    pub fn grid_n(&self) -> (Vec<usize>, Vec<usize>) {
        (
            self.grid.0.grid.iter().map(|x| x.0).collect(),
            self.grid.1.grid.iter().map(|x| x.0).collect(),
        )
    }
    pub fn fn2vec(&self, f: &dyn Fn(Float, Float) -> Float) -> Vec<Vec<Float>> {
        let mut v = Vec::new();
        for i in 0..self.grid.0.n {
            let mut v_i = Vec::new();
            for j in 0..self.grid.1.n {
                v_i.push(f(self.grid.0.coord[i], self.grid.1.coord[j]));
            }
            v.push(v_i);
        }
        v
    }
    pub fn fn2myvec(&self, f: &dyn Fn(Float, Float) -> Float) -> MyVec {
        let mut v = Vec::new();
        for i in 0..self.grid.0.n {
            for j in 0..self.grid.1.n {
                v.push(f(self.grid.0.coord[i], self.grid.1.coord[j]));
            }
        }
        MyVec(v)
    }
    pub fn print_vec(&self, v: &MyVec) {
        for j in 0..self.grid.1.n {
            for i in 0..self.grid.0.n {
                print!("{:<8} ", format!("{:.2E}", v.0[self.idx(i, j)]));
            }
            println!();
        }
    }
    pub fn rz_mean(&self, v: &MyVec) -> Float {
        let rs = &self.grid.0.coord;
        let drs: Vec<Float> = (0..self.grid.0.n)
            .map(|i| self.grid.0.dx[self.grid.0.grid_idx[i]])
            .collect();
        let dzs: Vec<Float> = (0..self.grid.1.n)
            .map(|j| self.grid.1.dx[self.grid.1.grid_idx[j]])
            .collect();
        let r_weights: Vec<Float> = rs.iter().zip(drs.iter()).map(|(r, dr)| r * dr).collect();
        let z_weights: Vec<Float> = dzs;
        let total_weight: Float = r_weights.iter().sum::<Float>() * z_weights.iter().sum::<Float>();
        let mut sum = 0.0;
        for i in 0..self.grid.0.n {
            for j in 0..self.grid.1.n {
                sum += v.0[self.idx(i, j)] * r_weights[i] * z_weights[j];
            }
        }
        sum / total_weight
    }
}
pub enum BorderCond2D {
    Value(Vec<Float>),
    Deriv(Vec<Float>),
    Comb(Vec<(Float, Float, Float)>),
}

pub fn diffusion_eqn_rz(
    grid: &Grid2D,
    border_r: &BorderCond2D,
    border_z: &(BorderCond2D, BorderCond2D),
    coeff: &Vec<Vec<Float>>,
    src: &Vec<Vec<Float>>,
) -> (SparseMatrix, MyVec) {
    let size = grid.grid.0.n * grid.grid.1.n;
    let mut data = Vec::new();
    let mut rhs = MyVec(vec![0.0; size]);
    let grid_r = &grid.grid.0.grid;
    let grid_z = &grid.grid.1.grid;
    for rii in 0..grid_r.len() {
        let ri_offset: usize = grid_r[0..rii].iter().map(|x| x.0).sum();
        let r0 = if rii == 0 {
            grid.grid.0.bdr
        } else {
            grid_r[rii - 1].1
        };
        let dr = grid.grid.0.dx[rii];
        for zjj in 0..grid_z.len() {
            let z0 = if zjj == 0 {
                grid.grid.1.bdr
            } else {
                grid_z[zjj - 1].1
            };
            let zj_offset: usize = grid_z[0..zjj].iter().map(|z| z.0).sum();
            let dz = grid.grid.1.dx[zjj];
            let coeff_loc = coeff[rii][zjj];
            for ri in ri_offset..(ri_offset + grid_r[rii].0) {
                let r = 0.5 * (grid.grid.0.coord[ri] + grid.grid.0.coord[ri + 1]);
                for zj in zj_offset..(zj_offset + grid_z[zjj].0) {
                    //let z = 0.5 * (grid.grid.1.coord[zj] + grid.grid.1.coord[zj + 1]);
                    let idx = (
                        grid.idx(ri, zj),
                        grid.idx(ri.saturating_sub(1), zj),
                        grid.idx(ri + 1, zj),
                        grid.idx(ri, zj.saturating_sub(1)),
                        grid.idx(ri, zj + 1),
                    );
                    rhs[idx.0] = -src[ri][zj] * dr * dz;
                    if ri != ri_offset
                        && zj != zj_offset
                        && ri != ri_offset + grid_r[rii].0 - 1
                        && zj != zj_offset + grid_z[zjj].0 - 1
                    {
                        data.push((idx.0, idx.0, -2.0 * coeff_loc * (dz / dr + dr / dz)));
                        data.push((idx.0, idx.1, coeff_loc * (dz / dr + dz * 0.5 / r)));
                        data.push((idx.0, idx.2, coeff_loc * (dz / dr - dz * 0.5 / r)));
                        data.push((idx.0, idx.3, coeff_loc * dr / dz));
                        data.push((idx.0, idx.4, coeff_loc * dr / dz));
                    } else {
                        if ri == ri_offset {
                            // q = 0 at center
                            if rii != 0 {
                                let r_left = if rii == 1 {
                                    grid.grid.0.bdr
                                } else {
                                    grid_r[rii - 2].1
                                };
                                let dr_left = (r0 - r_left) / grid_r[rii - 1].0 as Float;
                                let coeff_left = coeff[rii - 1][zjj];
                                let coeff_mean =
                                    (dr + dr_left) / (dr / coeff_loc + dr_left / coeff_left);
                                let dr_mean = 0.5 * (dr + dr_left);
                                data.push((
                                    idx.0,
                                    idx.0,
                                    -coeff_mean * (dz / dr_mean - dz * 0.5 / r),
                                ));
                                data.push((
                                    idx.0,
                                    idx.1,
                                    coeff_mean * (dz / dr_mean - dz * 0.5 / r),
                                ));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * (dz / dr - dz * 0.5 / r)));
                            data.push((idx.0, idx.1, coeff_loc * (dz / dr - dz * 0.5 / r)));
                        }
                        if ri == ri_offset + grid_r[rii].0 - 1 {
                            if rii == grid_r.len() - 1 {
                                match &border_r {
                                    BorderCond2D::Value(v) => {
                                        data.push((
                                            idx.0,
                                            idx.0,
                                            -2.0 * coeff_loc * (dz / dr + dz * 0.5 / r),
                                        ));
                                        rhs[idx.0] -=
                                            2.0 * coeff_loc * v[zj] * (dz / dr + dz * 0.5 / r);
                                    }
                                    BorderCond2D::Deriv(v) => {
                                        rhs[idx.0] -= coeff_loc * v[zj] * (dz + dz * 0.5 * dr / r);
                                    }
                                    BorderCond2D::Comb(v) => {
                                        // aT +b*dT/dr = c
                                        // -lambda*dT/dr+alpha(T-T_f)=0
                                        let (a, b, c) = v[zj];
                                        let tf = c / a;
                                        let alpha_mean = coeff_loc * 2.0 * a / (a * dr - 2.0 * b);
                                        rhs[idx.0] -= tf * alpha_mean * (dz / dr + dz * 0.5 / r);
                                        data.push((
                                            idx.0,
                                            idx.0,
                                            -alpha_mean * (dz / dr + dz * 0.5 / r),
                                        ));
                                    }
                                }
                            } else {
                                let dr_right = (grid_r[rii + 1].1 - grid_r[rii].1)
                                    / grid_r[rii + 1].0 as Float;
                                let coeff_right = coeff[rii + 1][zjj];
                                let coeff_mean =
                                    (dr + dr_right) / (dr / coeff_loc + dr_right / coeff_right);
                                let dr_mean = 0.5 * (dr + dr_right);
                                data.push((
                                    idx.0,
                                    idx.0,
                                    -coeff_mean * (dz / dr_mean + dz * 0.5 / r),
                                ));
                                data.push((
                                    idx.0,
                                    idx.2,
                                    coeff_mean * (dz / dr_mean + dz * 0.5 / r),
                                ));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * (dz / dr + dz * 0.5 / r)));
                            data.push((idx.0, idx.2, coeff_loc * (dz / dr + dz * 0.5 / r)));
                        }
                        if zj == zj_offset {
                            if zjj == 0 {
                                match &border_z.0 {
                                    BorderCond2D::Value(v) => {
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * dr / dz));
                                        rhs[idx.0] -= 2.0 * coeff_loc * v[ri] * dr / dz;
                                    }
                                    BorderCond2D::Deriv(v) => {
                                        rhs[idx.0] += coeff_loc * v[ri] * dr;
                                    }
                                    BorderCond2D::Comb(v) => {
                                        let (a, b, c) = v[ri];
                                        let b = -b;
                                        let tf = c / a;
                                        let alpha_mean = coeff_loc * 2.0 * a / (a * dz + 2.0 * b);
                                        rhs[idx.0] -= tf * alpha_mean * dr;
                                        data.push((idx.0, idx.0, -alpha_mean * dr));
                                    }
                                }
                            } else {
                                let z_down = if zjj == 1 {
                                    grid.grid.1.bdr
                                } else {
                                    grid_z[zjj - 2].1
                                };
                                let dz_down = (z0 - z_down) / grid_z[zjj - 1].0 as Float;
                                let coeff_down = coeff[rii][zjj - 1];
                                let coeff_mean =
                                    (dz + dz_down) / (dz / coeff_loc + dz_down / coeff_down);
                                let dz_mean = 0.5 * (dz + dz_down);
                                data.push((idx.0, idx.0, -coeff_mean * dr / dz_mean));
                                data.push((idx.0, idx.3, coeff_mean * dr / dz_mean));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * dr / dz));
                            data.push((idx.0, idx.3, coeff_loc * dr / dz));
                        }
                        if zj == zj_offset + grid_z[zjj].0 - 1 {
                            if zjj == grid_z.len() - 1 {
                                match &border_z.1 {
                                    BorderCond2D::Value(v) => {
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * dr / dz));
                                        rhs[idx.0] -= 2.0 * coeff_loc * v[ri] * dr / dz;
                                    }
                                    BorderCond2D::Deriv(v) => {
                                        rhs[idx.0] -= coeff_loc * v[ri] * dr;
                                    }
                                    BorderCond2D::Comb(v) => {
                                        let (a, b, c) = v[ri];
                                        let tf = c / a;
                                        let alpha_mean = coeff_loc * 2.0 * a / (a * dz + 2.0 * b);
                                        rhs[idx.0] -= tf * alpha_mean * dr;
                                        data.push((idx.0, idx.0, -alpha_mean * dr));
                                    }
                                }
                            } else {
                                let dz_up = (grid_z[zjj + 1].1 - grid_z[zjj].1)
                                    / grid_z[zjj + 1].0 as Float;
                                let coeff_up = coeff[rii][zjj + 1];
                                let coeff_mean = (dz + dz_up) / (dz / coeff_loc + dz_up / coeff_up);
                                let dz_mean = 0.5 * (dz + dz_up);
                                data.push((idx.0, idx.0, -coeff_mean * dr / dz_mean));
                                data.push((idx.0, idx.4, coeff_mean * dr / dz_mean));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * dr / dz));
                            data.push((idx.0, idx.4, coeff_loc * dr / dz));
                        }
                    }
                }
            }
        }
    }
    (SparseMatrix::new_with_data(size, size, data), rhs)
}

pub fn convec_mat_z(grid: &Grid2D, u: Float) -> SparseMatrix {
    let size = grid.grid.0.n * grid.grid.1.n;
    let mut data = Vec::new();
    let grid_z = &grid.grid.1.grid;
    let rn = grid.grid.0.n;
    for zjj in 0..grid_z.len() {
        let z0 = if zjj == 0 {
            grid.grid.1.bdr
        } else {
            grid_z[zjj - 1].1
        };
        let zj_offset: usize = grid_z[0..zjj].iter().map(|z| z.0).sum();
        let dz = (grid_z[zjj].1 - z0) / grid_z[zjj].0 as Float;
        for zj in zj_offset..(zj_offset + grid_z[zjj].0) {
            for ri in 0..rn {
                let idx = (grid.idx(ri, zj), grid.idx(ri, zj.saturating_sub(1)));
                if zj == zj_offset {
                    if zjj != 0 {
                        let z_down = if zjj == 1 {
                            grid.grid.1.bdr
                        } else {
                            grid_z[zjj - 2].1
                        };
                        let dz_down = (z0 - z_down) / grid_z[zjj - 1].0 as Float;
                        let dz_mean = 0.5 * (dz + dz_down);
                        data.push((idx.0, idx.0, -u / dz_mean));
                        data.push((idx.0, idx.1, u / dz_mean));
                    }
                } else {
                    data.push((idx.0, idx.0, -u / dz));
                    data.push((idx.0, idx.1, u / dz));
                }
            }
        }
    }
    SparseMatrix::new_with_data(size, size, data)
}
