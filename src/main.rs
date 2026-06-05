use reactor::Float;
use reactor::matrix::JacobiSolver;
use reactor::pde::*;

fn main() {
    let grid = Grid2D::new(0.0, vec![(5, 1.0), (5, 2.0)], 0.0, vec![(5, 1.5), (5, 3.0)]);
    let r_bdr = BorderCond2D::Value(Box::new(|_x: Float| 1.0));
    let z_bdr_1 = BorderCond2D::Value(Box::new(|_x: Float| 1.0));
    let z_bdr_2 = BorderCond2D::Value(Box::new(|_x: Float| 1.0));
    let coeff = vec![vec![0.2, 0.2], vec![0.2, 0.2]];
    let src = Box::new(|_x: Float, _y: Float| 2.01 );
    let (a, b) = diffusion_eqn_rz(&grid, r_bdr, (z_bdr_1, z_bdr_2), coeff, src);
    //println!("A:{:?}", a);
    //println!("b:{:?}", b);
    let solver = JacobiSolver::new(&a, &b);
    let res = solver.solve().unwrap();
    for j in 0..10 {
        for i in 0..10 {
            print!("{:<6}", format!("{:.2}", res[grid.idx(i, j)]));
        }
        println!("");
    }
}
