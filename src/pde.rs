use crate::{
    Float,
    matrix::{MyVec, SparseMatrix},
};
type Fn1 = Box<dyn Fn(Float) -> Float>;
type Fn2 = Box<dyn Fn(Float, Float) -> Float>;
struct Grid1D {
    bdr: Float,
    grid: Vec<(usize, Float)>,
    n: usize,
    coord: Vec<Float>,
    grid_idx: Vec<usize>,
}
impl Grid1D {
    fn new(bdr: Float, grid: Vec<(usize, Float)>) -> Self {
        let n = grid.iter().map(|x| x.0).sum();
        let (mut coord, mut grid_idx) = (Vec::new(), Vec::new());
        let mut x1 = bdr;
        for i in 0..grid.len() {
            let (n, x2) = grid[i];
            coord.extend((0..n).map(|xx| x1 + xx as Float * (x2 - x1) / n as Float));
            grid_idx.extend(std::iter::repeat(i).take(n));
            x1 = x2;
        }
        coord.push(x1);
        Self {
            bdr,
            grid,
            n,
            coord,
            grid_idx,
        }
    }
}
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
        i * self.grid.0.n + j
    }
    pub fn idx_inv(&self, idx: usize) -> (usize, usize) {
        (idx / self.grid.0.n, idx % self.grid.0.n)
    }
}
pub enum BorderCond2D {
    Value(Fn1),
    Deriv(Fn1),
    Comb(Float, Float, Fn1),
}
pub fn diffusion_eqn(
    grid: &Grid2D,
    border_x: (BorderCond2D, BorderCond2D),
    border_y: (BorderCond2D, BorderCond2D),
    coeff: Vec<Vec<Float>>,
    src: Fn2,
) -> (SparseMatrix, MyVec) {
    let size = grid.grid.0.n * grid.grid.1.n;
    let mut data = Vec::new();
    // jx_{i+1/2} = -lambda*(T_{i+1}-T_i)/dx
    // (jx_{i+1/2}-jx_{i-1/2})dy + (jy_{j+1/2}-jy_{j-1/2})dx = pdxdy
    // (lambda*(T_{i+1}-T_i)+lambda*(T_{i-1}-T_i))dy/dx + ... = -pdxdy
    let mut rhs = MyVec(vec![0.0; size]);
    let grid_x = &grid.grid.0.grid;
    let grid_y = &grid.grid.1.grid;
    for xii in 0..grid_x.len() {
        let xi_offset: usize = grid_x[0..xii].iter().map(|x| x.0).sum();
        let x0 = if xii == 0 {
            grid.grid.0.bdr
        } else {
            grid_x[xii - 1].1
        };
        let dx = (grid_x[xii].1 - x0) / grid_x[xii].0 as Float;
        for yjj in 0..grid_y.len() {
            let y0 = if yjj == 0 {
                grid.grid.1.bdr
            } else {
                grid_y[yjj - 1].1
            };
            let yj_offset: usize = grid_y[0..yjj].iter().map(|y| y.0).sum();
            let dy = (grid_y[yjj].1 - y0) / grid_y[yjj].0 as Float;
            let coeff_loc = coeff[xii][yjj];
            for xi in xi_offset..(xi_offset + grid_x[xii].0) {
                let x = 0.5 * (grid.grid.0.coord[xi] + grid.grid.0.coord[xi + 1]);
                for yj in yj_offset..(yj_offset + grid_y[yjj].0) {
                    let y = 0.5 * (grid.grid.1.coord[yj] + grid.grid.1.coord[yj + 1]);
                    let idx = (
                        grid.idx(xi, yj),
                        grid.idx(xi.saturating_sub(1), yj),
                        grid.idx(xi + 1, yj),
                        grid.idx(xi, yj.saturating_sub(1)),
                        grid.idx(xi, yj + 1),
                    );
                    rhs[idx.0] = -src(x, y) * dx * dy;
                    if xi != xi_offset
                        && yj != yj_offset
                        && xi != xi_offset + grid_x[xii].0 - 1
                        && yj != yj_offset + grid_y[yjj].0 - 1
                    {
                        data.push((idx.0, idx.0, -2.0 * coeff_loc * (dy / dx + dx / dy)));
                        data.push((idx.0, idx.1, coeff_loc * dy / dx));
                        data.push((idx.0, idx.2, coeff_loc * dy / dx));
                        data.push((idx.0, idx.3, coeff_loc * dx / dy));
                        data.push((idx.0, idx.4, coeff_loc * dx / dy));
                    } else {
                        // the following method may have greater round-off error
                        if xi == xi_offset {
                            // jx_{i+1/2} = -lambda*(T_{i+1}-T_i)/dx
                            // (jx_{i+1/2}-jx_{i-1/2})dy + (jy_{j+1/2}-jy_{j-1/2})dx = pdxdy
                            // (lambda*(T_{i+1}-T_i)+lambda*(T_{i-1}-T_i))dy/dx + ... = -pdxdy
                            if xii == 0 {
                                match &border_x.0 {
                                    BorderCond2D::Value(f) => {
                                        // (lambda*(T_{i+1}-T_i)+2.0*lambda*(T_{i-1/2}-T_i))dy/dx + ... = -pdxdy
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * dy / dx));
                                        rhs[idx.0] -= 2.0 * coeff_loc * f(y) * dy / dx;
                                    }
                                    BorderCond2D::Deriv(f) => {
                                        // q = -lambda * dT/dx
                                        rhs[idx.0] += coeff_loc * f(y) * dy;
                                    }
                                    BorderCond2D::Comb(a, b, f) => {
                                        // (lambda*(T_{i+1}-T_i)-q*dy + ... = -pdxdy
                                        let c = f(y);
                                        let b = -b;
                                        let tf = c / a;
                                        let alpha_mean = coeff_loc * 2.0 * a / (a * dx + 2.0 * b);
                                        rhs[idx.0] -= tf * alpha_mean * dy;
                                        data.push((idx.0, idx.0, -alpha_mean * dy));
                                    }
                                }
                            } else {
                                let x_left = if xii == 1 {
                                    grid.grid.0.bdr
                                } else {
                                    grid_x[xii - 2].1
                                };
                                let dx_left = (x0 - x_left) / grid_x[xii - 1].0 as Float;
                                let coeff_left = coeff[xii - 1][yjj];
                                let coeff_mean =
                                    (dx + dx_left) / (dx / coeff_loc + dx_left / coeff_left);
                                let dx_mean = 0.5 * (dx + dx_left);
                                data.push((idx.0, idx.0, -coeff_mean * dy / dx_mean));
                                data.push((idx.0, idx.1, coeff_mean * dy / dx_mean));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * dy / dx));
                            data.push((idx.0, idx.1, coeff_loc * dy / dx));
                        }
                        if xi == xi_offset + grid_x[xii].0 - 1 {
                            if xii == grid_x.len() - 1 {
                                match &border_x.1 {
                                    BorderCond2D::Value(f) => {
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * dy / dx));
                                        rhs[idx.0] -= 2.0 * coeff_loc * f(y) * dy / dx;
                                    }
                                    BorderCond2D::Deriv(f) => {
                                        rhs[idx.0] -= coeff_loc * f(y) * dy;
                                    }
                                    BorderCond2D::Comb(a, b, f) => {
                                        // (lambda*(T_{i+1}-T_i)-q*dy + ... = -pdxdy
                                        let c = f(y);
                                        let tf = c / a;
                                        let alpha_mean = coeff_loc * 2.0 * a / (a * dx + 2.0 * b);
                                        rhs[idx.0] -= tf * alpha_mean * dy;
                                        data.push((idx.0, idx.0, -alpha_mean * dy));
                                    }
                                }
                            } else {
                                let dx_right = (grid_x[xii + 1].1 - grid_x[xii].1)
                                    / grid_x[xii + 1].0 as Float;
                                let coeff_right = coeff[xii + 1][yjj];
                                let coeff_mean =
                                    (dx + dx_right) / (dx / coeff_loc + dx_right / coeff_right);
                                let dx_mean = 0.5 * (dx + dx_right);
                                data.push((idx.0, idx.0, -coeff_mean * dy / dx_mean));
                                data.push((idx.0, idx.2, coeff_mean * dy / dx_mean));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * dy / dx));
                            data.push((idx.0, idx.2, coeff_loc * dy / dx));
                        }
                        if yj == yj_offset {
                            if yjj == 0 {
                                match &border_y.0 {
                                    BorderCond2D::Value(f) => {
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * dx / dy));
                                        rhs[idx.0] -= 2.0 * coeff_loc * f(x) * dx / dy;
                                    }
                                    BorderCond2D::Deriv(f) => {
                                        rhs[idx.0] += coeff_loc * f(x) * dx;
                                    }
                                    BorderCond2D::Comb(a, b, f) => {
                                        // (lambda*(T_{i+1}-T_i)-q*dy + ... = -pdxdy
                                        let c = f(x);
                                        let b = -b;
                                        let tf = c / a;
                                        let alpha_mean = coeff_loc * 2.0 * a / (a * dy + 2.0 * b);
                                        rhs[idx.0] -= tf * alpha_mean * dx;
                                        data.push((idx.0, idx.0, -alpha_mean * dx));
                                    }
                                }
                            } else {
                                let y_down = if yjj == 1 {
                                    grid.grid.1.bdr
                                } else {
                                    grid_y[yjj - 2].1
                                };
                                let dy_down = (y0 - y_down) / grid_y[yjj - 1].0 as Float;
                                let coeff_down = coeff[xii][yjj - 1];
                                let coeff_mean =
                                    (dy + dy_down) / (dy / coeff_loc + dy_down / coeff_down);
                                let dy_mean = 0.5 * (dy + dy_down);
                                data.push((idx.0, idx.0, -coeff_mean * dx / dy_mean));
                                data.push((idx.0, idx.3, coeff_mean * dx / dy_mean));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * dx / dy));
                            data.push((idx.0, idx.3, coeff_loc * dx / dy));
                        }
                        if yj == yj_offset + grid_y[yjj].0 - 1 {
                            if yjj == grid_y.len() - 1 {
                                match &border_y.1 {
                                    BorderCond2D::Value(f) => {
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * dx / dy));
                                        rhs[idx.0] -= 2.0 * coeff_loc * f(x) * dx / dy;
                                    }
                                    BorderCond2D::Deriv(f) => {
                                        rhs[idx.0] -= coeff_loc * f(x) * dx;
                                    }
                                    BorderCond2D::Comb(a, b, f) => {
                                        // (lambda*(T_{i+1}-T_i)-q*dy + ... = -pdxdy
                                        let c = f(x);
                                        let tf = c / a;
                                        let alpha_mean = coeff_loc * 2.0 * a / (a * dy + 2.0 * b);
                                        rhs[idx.0] -= tf * alpha_mean * dx;
                                        data.push((idx.0, idx.0, -alpha_mean * dx));
                                    }
                                }
                            } else {
                                let dy_up = (grid_y[yjj + 1].1 - grid_y[yjj].1)
                                    / grid_y[yjj + 1].0 as Float;
                                let coeff_up = coeff[xii][yjj + 1];
                                let coeff_mean = (dy + dy_up) / (dy / coeff_loc + dy_up / coeff_up);
                                let dy_mean = 0.5 * (dy + dy_up);
                                data.push((idx.0, idx.0, -coeff_mean * dx / dy_mean));
                                data.push((idx.0, idx.4, coeff_mean * dx / dy_mean));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * dx / dy));
                            data.push((idx.0, idx.4, coeff_loc * dx / dy));
                        }
                    }
                }
            }
        }
    }
    (SparseMatrix::new_with_data(size, size, data), rhs)
}
pub fn diffusion_eqn_rz(
    grid: &Grid2D,
    border_r: BorderCond2D,
    border_z: (BorderCond2D, BorderCond2D),
    coeff: Vec<Vec<Float>>,
    src: Fn2,
) -> (SparseMatrix, MyVec) {
    const PI: Float = std::f64::consts::PI as Float;
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
        let dr = (grid_r[rii].1 - r0) / grid_r[rii].0 as Float;
        for zjj in 0..grid_z.len() {
            let z0 = if zjj == 0 {
                grid.grid.1.bdr
            } else {
                grid_z[zjj - 1].1
            };
            let zj_offset: usize = grid_z[0..zjj].iter().map(|z| z.0).sum();
            let dz = (grid_z[zjj].1 - z0) / grid_z[zjj].0 as Float;
            let coeff_loc = coeff[rii][zjj];
            for ri in ri_offset..(ri_offset + grid_r[rii].0) {
                let r = 0.5 * (grid.grid.0.coord[ri] + grid.grid.0.coord[ri + 1]);
                for zj in zj_offset..(zj_offset + grid_z[zjj].0) {
                    let z = 0.5 * (grid.grid.1.coord[zj] + grid.grid.1.coord[zj + 1]);
                    let idx = (
                        grid.idx(ri, zj),
                        grid.idx(ri.saturating_sub(1), zj),
                        grid.idx(ri + 1, zj),
                        grid.idx(ri, zj.saturating_sub(1)),
                        grid.idx(ri, zj + 1),
                    );
                    rhs[idx.0] = -src(r, z) * dr * dz;
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
                                data.push((idx.0, idx.0, -coeff_mean * (dz / dr_mean - dz * 0.5 / r)));
                                data.push((idx.0, idx.1, coeff_mean * (dz / dr_mean - dz * 0.5 / r)));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * (dz / dr - dz * 0.5 / r)));
                            data.push((idx.0, idx.1, coeff_loc * (dz / dr - dz * 0.5 / r)));
                        }
                        if ri == ri_offset + grid_r[rii].0 - 1 {
                            if rii == grid_r.len() - 1 {
                                match &border_r {
                                    BorderCond2D::Value(f) => {
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * (dz / dr + dz * 0.5 / r)));
                                        rhs[idx.0] -= 2.0 * coeff_loc * f(z) * (dz / dr + dz * 0.5 / r);
                                    }
                                    BorderCond2D::Deriv(f) => {
                                        rhs[idx.0] -= coeff_loc * f(z) * (dz + dz * 0.5 * dr / r);
                                    }
                                    BorderCond2D::Comb(a, b, f) => {
                                        // aT +b*dT/dr = c
                                        // -lambda*dT/dr+alpha(T-T_f)=0
                                        let c = f(z);
                                        let tf = c / a;
                                        let alpha_mean = coeff_loc * 2.0 * a / (a * dr - 2.0 * b);
                                        rhs[idx.0] -= tf * alpha_mean * (dz / dr + dz * 0.5 / r);
                                        data.push((idx.0, idx.0, -alpha_mean * (dz / dr + dz * 0.5 / r)));
                                    }
                                }
                            } else {
                                let dr_right = (grid_r[rii + 1].1 - grid_r[rii].1)
                                    / grid_r[rii + 1].0 as Float;
                                let coeff_right = coeff[rii + 1][zjj];
                                let coeff_mean =
                                    (dr + dr_right) / (dr / coeff_loc + dr_right / coeff_right);
                                let dr_mean = 0.5 * (dr + dr_right);
                                data.push((idx.0, idx.0, -coeff_mean * (dz / dr_mean + dz * 0.5 / r)));
                                data.push((idx.0, idx.2, coeff_mean * (dz / dr_mean + dz * 0.5 / r)));
                            }
                        } else {
                            data.push((idx.0, idx.0, -coeff_loc * (dz / dr + dz * 0.5 / r)));
                            data.push((idx.0, idx.2, coeff_loc * (dz / dr + dz * 0.5 / r)));
                        }
                        if zj == zj_offset {
                            if zjj == 0 {
                                match &border_z.0 {
                                    BorderCond2D::Value(f) => {
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * dr / dz));
                                        rhs[idx.0] -= 2.0 * coeff_loc * f(r) * dr / dz;
                                    }
                                    BorderCond2D::Deriv(f) => {
                                        rhs[idx.0] += coeff_loc * f(r) * dr;
                                    }
                                    BorderCond2D::Comb(a, b, f) => {
                                        let c = f(r);
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
                                    BorderCond2D::Value(f) => {
                                        data.push((idx.0, idx.0, -2.0 * coeff_loc * dr / dz));
                                        rhs[idx.0] -= 2.0 * coeff_loc * f(r) * dr / dz;
                                    }
                                    BorderCond2D::Deriv(f) => {
                                        rhs[idx.0] -= coeff_loc * f(r) * dr;
                                    }
                                    BorderCond2D::Comb(a, b, f) => {
                                        let c = f(r);
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
