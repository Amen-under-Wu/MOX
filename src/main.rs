use reactor::Float;
use reactor::matrix::JacobiSolver;
use reactor::pde::*;

fn main() {
    let grid = Grid2D::new(0.0, vec![(5, 1.0), (5, 2.0)], 0.0, vec![(5, 1.5), (5, 3.0)]);
    let x_bdr_1 = BorderCond2D::Comb(1.0, -1.0, Box::new(|_| 1.0));
    let x_bdr_2 = BorderCond2D::Comb(1.0, 1.0, Box::new(|_| 1.0));
    let y_bdr_1 = BorderCond2D::Comb(1.0, -1.0, Box::new(|_| 1.0));
    let y_bdr_2 = BorderCond2D::Comb(1.0, 1.0, Box::new(|_| 1.0));
    /*let x_bdr_1 = BorderCond2D::Value(Box::new(|_| 1.0));
    let x_bdr_2 = BorderCond2D::Value(Box::new(|_| 0.0));
    let y_bdr_1 = BorderCond2D::Deriv(Box::new(|_| 1.0));
    let y_bdr_2 = BorderCond2D::Deriv(Box::new(|_| 0.0));*/
    let coeff = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
    let src = Box::new(|x: Float, y: Float| 2.01 * x.sin() * y.cos());
    let (a, b) = diffusion_eqn(&grid, (x_bdr_1, x_bdr_2), (y_bdr_1, y_bdr_2), coeff, src);
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
