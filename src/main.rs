use reactor::matrix::*;
use reactor::nuke::diffusion::Diffusion;
use reactor::nuke::material::MaterialData;
use reactor::pde::*;
use reactor::thermo::convection::*;

fn _test_eqn() {
    let grid = Grid2D::new(0.0, vec![(5, 1.0), (5, 2.0)], 0.0, vec![(5, 1.5), (5, 3.0)]);
    let coeff = vec![vec![0.2, 0.2], vec![0.2, 0.2]];
    let src = vec![vec![2.0; 10]; 10];
    let bdr = vec![0.0; 10];
    let r_bdr = BorderCond2D::Value(bdr.clone());
    let z_bdr_1 = BorderCond2D::Value(bdr.clone());
    let z_bdr_2 = BorderCond2D::Value(bdr.clone());
    let (a, b) = diffusion_eqn_rz(&grid, &r_bdr, &(z_bdr_1, z_bdr_2), &coeff, &src);
    //println!("A:{:?}", a);
    //println!("b:{:?}", b);
    println!("Jacobi:");
    let solver = JacobiSolver::new(&a, &b);
    let res = solver.solve().unwrap();
    for j in 0..10 {
        for i in 0..10 {
            print!("{:<6}", format!("{:.2}", res[grid.idx(i, j)]));
        }
        println!("");
    }
    println!("SOR:");
    let solver = SORSolver::new(&a, &b, 1.5);
    let res2 = solver.solve().unwrap();
    for j in 0..10 {
        for i in 0..10 {
            print!("{:<6}", format!("{:.2}", res2[grid.idx(i, j)]));
        }
        println!("");
    }
    println!("Diff:");
    let rese = (res2 - &res) * 100.0;
    for j in 0..10 {
        for i in 0..10 {
            print!(
                "{:<6}",
                format!("{:.2}", rese[grid.idx(i, j)] / res[grid.idx(i, j)])
            );
        }
        println!("");
    }
}

fn test_nuke() {
    let grid = Grid2D::new(0.0, vec![(5, 1.0), (5, 2.0)], 0.0, vec![(5, 1.5), (5, 3.0)]);
    let fuel = MaterialData::fuel();
    let mat = vec![vec![fuel.clone(); 2]; 2];
    let diff = Diffusion::new(mat, grid.clone());
    let init = MyVec(vec![1e14; 100]);
    let res = diff.fast_group_flux(&init);
    grid.print_vec(&res);
    let res = diff.thermo_group_flux(&res);
    grid.print_vec(&res);
}

fn test_thermo() {
    let grid = Grid2D::new(0.0, vec![(10, 1.5)], 0.0, vec![(10, 11.0)]);
    let helium = FluidData::helium(0.06, 96.0 / (1.5 * 1.5 * 3.14));
    let conv = Convection::new(helium, grid.clone());
    let src = grid.fn2myvec(&|r, z| 20.0 * (3.14 * z / 11.0).sin() * (3.14 * r / 1.5).cos());
    let res = conv.temp_solve(0.0, &src, &vec![0.0; 10]);
    for j in 0..10 {
        for i in 0..10 {
            print!("{:<8}", format!("{:.2E}", res[grid.idx(i, j)]));
        }
        println!("");
    }
}

fn main() {
    // _test_eqn();
    test_nuke();
    test_thermo();
}
