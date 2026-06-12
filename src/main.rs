use reactor::matrix::*;
use reactor::nuke::diffusion::Diffusion;
use reactor::nuke::material::MaterialData;
use reactor::pde::*;
use reactor::thermo::convection::*;
use reactor::input::*;

fn _test_feedback() {
    let grid = Grid2D::new(0.0, vec![(10, 2.5), (5, 2.7)], 0.0, vec![(10, 11.0)]);
    let helium = FluidData::helium(0.06, 96.0 / (1.5 * 1.5 * 3.14));
    let conv = Convection::new(helium, grid.clone());
    let fuel = MaterialData::fuel();
    let carbon = MaterialData::carbon();
    let mat = vec![vec![fuel.clone()], vec![carbon.clone()]];
    let mut diff = Diffusion::new(mat, grid.clone());
    let init = MyVec(vec![1e13; 150]);
    const EF: f64 = 200e6 * 1.6e-19;
    let sigma_orig = diff.material[0][0].sigma;
    let temp_vec = MyVec(vec![273.15; 150]);
    let mut temp_before = temp_vec.clone();
    for _i in 0..10 {
        println!("Iteration: {}", _i);
        let fast = diff.fast_group_flux(&init);
        let thermo = diff.thermo_group_flux(&fast);
        let heat = diff.power_src(&fast, &thermo, EF);
        let temp = conv.temp_solve(273.15, &heat, &vec![273.15; 15]);
        let temp_avg = grid.rz_mean_loc(&temp, 0, 0);
        diff.material[0][0].sigma = (
            sigma_orig.0 * (temp_avg / 273.15).powf(0.5),
            sigma_orig.1 * (temp_avg / 273.15).powf(0.5),
        );
        let temp_err = &temp_before - &temp;
        println!("Temp error norm: {}", temp_err.norm2());
        if temp_err.norm2() < 1e-3 {
            println!("Converged!");
            println!("Phi_1:");
            grid.print_vec(&fast);
            println!("Phi_2:");
            grid.print_vec(&thermo);
            println!("Temp (K):");
            grid.print_vec(&temp);
            //grid.print_vec(&(temp - &temp_vec));
            break;
        }
        temp_before = temp;
    }
}

fn test_input() {
    let tokens = parse_input("IO/input/input.txt");
    let grid = Grid2D::from_tokens((&tokens[0], &tokens[1]));
    let t0 = tokens[2][0].parse::<f64>().unwrap();
    let u = tokens[3][0].parse::<f64>().unwrap();
    let grid_n = grid.grid_n();
    let rn = grid_n.0.iter().sum::<usize>();
    let zn = grid_n.1.iter().sum::<usize>();
    let helium = FluidData::helium(0.06, u);
    let conv = Convection::new(helium, grid.clone());
    let fuel = MaterialData::fuel();
    let carbon = MaterialData::carbon();
    let mat = vec![vec![fuel.clone()], vec![carbon.clone()]];
    let mut diff = Diffusion::new(mat, grid.clone());
    let init = MyVec(vec![1e13; rn * zn]);
    const EF: f64 = 200e6 * 1.6e-19;
    let sigma_orig = diff.material[0][0].sigma;
    let temp_vec = MyVec(vec![t0; rn * zn]);
    let mut temp_before = temp_vec.clone();
    for _i in 0..10 {
        println!("Iteration: {}", _i);
        println!("solving fast");
        let fast = diff.fast_group_flux(&init);
        println!("solving thermo");
        let thermo = diff.thermo_group_flux(&fast);
        let heat = diff.power_src(&fast, &thermo, EF);
        println!("solving temp");
        let temp = conv.temp_solve(t0, &heat, &vec![t0; zn]);
        let temp_avg = grid.rz_mean_loc(&temp, 0, 0);
        diff.material[0][0].sigma = (
            sigma_orig.0 * (temp_avg / t0).powf(0.5),
            sigma_orig.1 * (temp_avg / t0).powf(0.5),
        );
        let temp_err = &temp_before - &temp;
        println!("Temp error norm: {}", temp_err.norm2());
        if temp_err.norm2() < 1e-3 {
            println!("Converged!");
            println!("Phi_1:");
            grid.print_vec(&fast);
            println!("Phi_2:");
            grid.print_vec(&thermo);
            println!("Temp (K):");
            grid.print_vec(&temp);
            break;
        }
        temp_before = temp;
    }
}

fn main() {
    test_input();
}
