use crate::Float;
use crate::matrix::MyVec;
use crate::nuke::neutron::NeutronGroup;

struct BurnUp {
    section: Vec<(usize, usize, Vec<Float>)>,
    decay: Vec<(usize, usize, Float)>,
}

impl BurnUp {
    fn equiv_decay(sec: &Vec<Float>, n: &NeutronGroup) -> Float {
        sec.iter().zip(n.0.iter()).map(|(a, b)| a * b).sum()
    }
    fn dxdt(&self, nuclear: &MyVec, n: &NeutronGroup) -> MyVec {
        let mut res = MyVec(vec![0.0; nuclear.0.len()]);
        for (i, j, sec) in self.section.iter() {
            let decay = Self::equiv_decay(&sec, n);
            res[*i] -= decay;
            res[*j] += decay
        }
        for (i, j, l) in self.decay.iter() {
            res[*i] -= *l;
            res[*j] += *l;
        }
        for i in 0..nuclear.0.len() {
            res[i] *= nuclear[i];
        }
        res
    }
}
