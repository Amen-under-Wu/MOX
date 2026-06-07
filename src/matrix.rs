use crate::Float;
use std::collections::HashMap;
use std::ops::{Add, Index, IndexMut, Mul, Neg, Sub};

#[derive(Debug, Clone)]
pub struct MyVec(pub Vec<Float>);

impl MyVec {
    pub fn norm2(&self) -> Float {
        self.0.iter().map(|x| x * x).sum()
    }
}

impl Index<usize> for MyVec {
    type Output = Float;
    fn index(&self, index: usize) -> &Float {
        &self.0[index]
    }
}
impl IndexMut<usize> for MyVec {
    fn index_mut(&mut self, index: usize) -> &mut Float {
        &mut self.0[index]
    }
}

impl Add for MyVec {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let result = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a + b)
            .collect();
        MyVec(result)
    }
}
impl Add<&MyVec> for MyVec {
    type Output = Self;

    fn add(self, other: &Self) -> Self {
        let result = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a + b)
            .collect();
        MyVec(result)
    }
}
impl Add for &MyVec {
    type Output = MyVec;

    fn add(self, other: Self) -> MyVec {
        let result = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a + b)
            .collect();
        MyVec(result)
    }
}
impl Add<MyVec> for &MyVec {
    type Output = MyVec;

    fn add(self, other: MyVec) -> MyVec {
        other + self
    }
}
impl Sub for MyVec {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let result = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a - b)
            .collect();
        MyVec(result)
    }
}
impl Sub<&MyVec> for MyVec {
    type Output = Self;

    fn sub(self, other: &Self) -> Self {
        let result = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a - b)
            .collect();
        MyVec(result)
    }
}
impl Sub for &MyVec {
    type Output = MyVec;

    fn sub(self, other: Self) -> MyVec {
        let result = self
            .0
            .iter()
            .zip(other.0.iter())
            .map(|(a, b)| a - b)
            .collect();
        MyVec(result)
    }
}

impl Mul<Float> for MyVec {
    type Output = Self;

    fn mul(self, scalar: Float) -> Self {
        let result = self.0.iter().map(|x| x * scalar).collect();
        MyVec(result)
    }
}
impl Mul<Float> for &MyVec {
    type Output = MyVec;

    fn mul(self, scalar: Float) -> MyVec {
        let result = self.0.iter().map(|x| x * scalar).collect();
        MyVec(result)
    }
}
impl Mul<MyVec> for Float {
    type Output = MyVec;

    fn mul(self, vec: MyVec) -> MyVec {
        vec * self
    }
}
impl Mul<&MyVec> for Float {
    type Output = MyVec;

    fn mul(self, vec: &MyVec) -> MyVec {
        vec * self
    }
}

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    pub data: Vec<Vec<Float>>,
}
impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![vec![0.0; cols]; rows],
        }
    }
    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn mul_vec(&self, vec: &MyVec) -> MyVec {
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.data[i][j] * vec.0[j];
            }
        }
        MyVec(result)
    }
    fn regularize(&mut self) {
        while self.data.len() < self.rows {
            self.data.push(vec![0.0; self.cols]);
        }
        if self.data.len() > self.rows {
            self.data.truncate(self.rows);
        }
        for row in &mut self.data {
            while row.len() < self.cols {
                row.push(0.0);
            }
            if row.len() > self.cols {
                row.truncate(self.cols);
            }
        }
    }
    fn swap_row(&mut self, row_i: usize, row_j: usize) {
        if row_i == row_j {
            return;
        }
        if row_i < row_j {
            let (left, right) = self.data.split_at_mut(row_i + 1);
            std::mem::swap(&mut left[row_i], &mut right[row_j - row_i]);
        } else {
            self.swap_row(row_j, row_i);
        }
    }
    pub fn gauss_j(&mut self, b: &mut Matrix) -> Result<(), &'static str> {
        let n = self.rows;
        let m = b.cols;
        let mut ipiv = vec![0; n];
        let mut indxc = vec![0; n];
        let mut indxr = vec![0; n];
        let mut irow = 0;
        let mut icol = 0;
        for i in 0..n {
            let mut big = 0.0;
            for j in 0..n {
                if ipiv[j] != 1 {
                    for k in 0..n {
                        if ipiv[k] == 0 {
                            if self.data[j][k].abs() >= big {
                                big = self.data[j][k].abs();
                                irow = j;
                                icol = k;
                            }
                        }
                    }
                }
            }
            ipiv[icol] += 1;
            if irow != icol {
                self.swap_row(irow, icol);
                b.swap_row(irow, icol);
            }
            indxr[i] = irow;
            indxc[i] = icol;
            let piv = self.data[icol][icol];
            if piv == 0.0 {
                return Err("Singular matrix in gauss_j");
            }
            let pivinv = 1.0 / piv;
            self.data[icol][icol] = 1.0;
            for l in 0..n {
                self.data[icol][l] *= pivinv;
            }
            for l in 0..m {
                b.data[icol][l] *= pivinv;
            }
            for ll in 0..n {
                if ll != icol {
                    let dum = self.data[ll][icol];
                    self.data[ll][icol] = 0.0;
                    for l in 0..n {
                        self.data[ll][l] -= self.data[icol][l] * dum;
                    }
                    for l in 0..m {
                        b.data[ll][l] -= b.data[icol][l] * dum;
                    }
                }
            }
        }
        for l in (0..n).rev() {
            if indxr[l] != indxc[l] {
                for k in 0..n {
                    let tmp = self.data[k][indxr[l]];
                    self.data[k][indxr[l]] = self.data[k][indxc[l]];
                    self.data[k][indxc[l]] = tmp;
                }
            }
        }
        Ok(())
    }
}
impl Index<(usize, usize)> for Matrix {
    type Output = Float;
    fn index(&self, index: (usize, usize)) -> &Float {
        &self.data[index.0][index.1]
    }
}
impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Float {
        &mut self.data[index.0][index.1]
    }
}
impl Neg for Matrix {
    type Output = Self;

    fn neg(mut self) -> Self {
        for row in self.data.iter_mut() {
            for value in row.iter_mut() {
                *value = -*value;
            }
        }
        self
    }
}

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    rows: usize,
    cols: usize,
    data: Vec<(usize, usize, Float)>, // (row, col, value)
}
impl SparseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        SparseMatrix {
            rows,
            cols,
            data: Vec::new(),
        }
    }
    pub fn new_with_data(rows: usize, cols: usize, data: Vec<(usize, usize, Float)>) -> Self {
        let mut res = SparseMatrix { rows, cols, data };
        res.deduplicate();
        res.sort();
        res
    }
    pub fn add_entry(&mut self, row: usize, col: usize, value: Float) {
        self.data.push((row, col, value));
    }
    pub fn mul_vec(&self, vec: &MyVec) -> MyVec {
        let mut result = vec![0.0; self.rows];
        for (row, col, value) in &self.data {
            result[*row] += value * vec.0[*col];
        }
        MyVec(result)
    }
    pub fn transpose(&self) -> SparseMatrix {
        let mut res = SparseMatrix::new(self.cols, self.rows);
        for (i, j, v) in self.data.iter() {
            res.data.push((*j, *i, *v));
        }
        res
    }
    fn sort(&mut self) {
        self.data.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    }
    fn deduplicate(&mut self) {
        let mut map = HashMap::new();
        for (row, col, value) in &self.data {
            *map.entry((*row, *col)).or_insert(0.0) += *value;
        }
        self.data = map
            .into_iter()
            .map(|((row, col), value)| (row, col, value))
            .collect();
        self.data.retain(|(_, _, value)| *value != 0.0);
    }
    fn col_compress(&self) -> Vec<Vec<(usize, Float)>> {
        let mut res = vec![Vec::new(); self.cols];
        for (row, col, value) in &self.data {
            res[*col].push((*row, *value));
        }
        res
    }
    fn row_compress(&self) -> Vec<Vec<(usize, Float)>> {
        let mut res = vec![Vec::new(); self.rows];
        for (row, col, value) in &self.data {
            res[*row].push((*col, *value));
        }
        res
    }
    fn l_inv_on_vec(&self, rhs: &MyVec) -> MyVec {
        let mut res = rhs.clone();
        for (i, j, v) in self.data.iter() {
            if *i == *j {
                res[*i] /= *v;
            } else {
                res[*i] -= *v * res[*j];
            }
        }
        res
    }
    fn l_inv_on_mat(&self, rhs: &SparseMatrix) -> SparseMatrix {
        let mut rcomp = rhs.row_compress();
        for (i, j, v) in self.data.iter() {
            if *i == *j {
                for (_, vv) in rcomp[*i].iter_mut() {
                    *vv /= *v;
                }
            } else {
                let (s1, s2) = rcomp.split_at_mut(*j + 1);
                for (k, vv) in s1[*j].iter() {
                    s2[*i - *j - 1].push((*k, -*vv * *v));
                }
            }
        }
        let mut res = SparseMatrix::new(rhs.rows, rhs.cols);
        for i in 0..rcomp.len() {
            for (j, v) in rcomp[i].iter() {
                res.data.push((i, *j, *v));
            }
        }
        res.deduplicate();
        res.sort();
        res
    }
}
impl Add for SparseMatrix {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let mut result = self;
        result.data.extend(other.data);
        result.deduplicate();
        result.sort();
        result
    }
}
impl Add<&SparseMatrix> for SparseMatrix {
    type Output = Self;

    fn add(mut self, other: &Self) -> Self {
        self.data.extend(other.data.iter().cloned());
        self.deduplicate();
        self.sort();
        self
    }
}
impl Add for &SparseMatrix {
    type Output = SparseMatrix;

    fn add(self, other: Self) -> SparseMatrix {
        let mut result = self.clone();
        result.data.extend(other.data.iter().cloned());
        result.deduplicate();
        result.sort();
        result
    }
}
impl Sub for SparseMatrix {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let mut result = self;
        result.data.extend(
            other
                .data
                .into_iter()
                .map(|(row, col, value)| (row, col, -value)),
        );
        result.deduplicate();
        result.sort();
        result
    }
}
impl Sub<&SparseMatrix> for SparseMatrix {
    type Output = Self;

    fn sub(mut self, other: &Self) -> Self {
        self.data.extend(
            other
                .data
                .iter()
                .cloned()
                .map(|(row, col, value)| (row, col, -value)),
        );
        self.deduplicate();
        self.sort();
        self
    }
}
impl Sub for &SparseMatrix {
    type Output = SparseMatrix;

    fn sub(self, other: Self) -> SparseMatrix {
        let mut result = self.clone();
        result.data.extend(
            other
                .data
                .iter()
                .cloned()
                .map(|(row, col, value)| (row, col, -value)),
        );
        result.deduplicate();
        result.sort();
        result
    }
}
impl Mul<Float> for SparseMatrix {
    type Output = Self;

    fn mul(mut self, scalar: Float) -> Self {
        for (_, _, value) in &mut self.data {
            *value *= scalar;
        }
        self
    }
}
impl Mul<Float> for &SparseMatrix {
    type Output = SparseMatrix;

    fn mul(self, scalar: Float) -> SparseMatrix {
        let mut result = self.clone();
        for (_, _, value) in &mut result.data {
            *value *= scalar;
        }
        result
    }
}
impl Mul<&MyVec> for SparseMatrix {
    type Output = MyVec;

    fn mul(self, vec: &MyVec) -> MyVec {
        self.mul_vec(vec)
    }
}
impl Mul<&MyVec> for &SparseMatrix {
    type Output = MyVec;

    fn mul(self, vec: &MyVec) -> MyVec {
        self.mul_vec(vec)
    }
}
impl Mul for &SparseMatrix {
    // aik*bkj->cij
    type Output = SparseMatrix;

    fn mul(self, rhs: &SparseMatrix) -> SparseMatrix {
        assert_eq!(self.cols, rhs.rows);
        let mut data = Vec::new();
        let ccomp = self.col_compress();
        let rcomp = rhs.row_compress();
        let mut hashmap = HashMap::new();
        for k in 0..self.cols {
            for i in ccomp[k].iter() {
                for j in rcomp[k].iter() {
                    hashmap
                        .entry((i.0, j.0))
                        .and_modify(|x| *x += i.1 * j.1)
                        .or_insert(i.1 * j.1);
                }
            }
        }
        for (k, v) in hashmap {
            data.push((k.0, k.1, v));
        }
        let mut res = SparseMatrix {
            rows: self.rows,
            cols: rhs.cols,
            data,
        };
        res.deduplicate();
        res.sort();
        res
    }
}

#[derive(Debug)]
pub struct JacobiSolver {
    orig_mat: SparseMatrix,
    lpu: SparseMatrix,
    d_inv: SparseMatrix,
    b: MyVec,
    eps: Float,
}
impl JacobiSolver {
    pub fn new(mat: &SparseMatrix, b: &MyVec) -> Self {
        assert_eq!(mat.rows, mat.cols);
        let orig_mat = mat.clone();
        let mut lpu = mat.clone();
        let mut d_inv = SparseMatrix::new(mat.rows, mat.cols);
        for (row, col, value) in lpu.data.iter_mut() {
            if row == col {
                d_inv.add_entry(*row, *col, 1.0 / *value);
                *value = 0.0;
            }
            *value = -*value;
        }
        lpu.deduplicate();
        lpu.sort();
        Self {
            orig_mat,
            lpu,
            d_inv,
            b: b.clone(),
            eps: 1e-5,
        }
    }
    pub fn solve(&self) -> Result<MyVec, &str> {
        let mut res = MyVec(vec![1.0; self.orig_mat.cols]);
        const MAX_ITER: usize = 5000;
        for _ in 0..MAX_ITER {
            res = &self.d_inv * &(&self.lpu * &res + &self.b);
            if (&self.orig_mat * &res - &self.b).norm2() < self.eps {
                return Ok(res);
            }
        }
        Err("Iteration reaches max limit: problem may not converge")
    }
}

#[derive(Debug)]
pub struct SORSolver {
    orig_mat: SparseMatrix,
    iter_mat: SparseMatrix,
    iter_vec: MyVec,
    b: MyVec,
    eps: Float,
}
impl SORSolver {
    pub fn new(mat: &SparseMatrix, b: &MyVec, omega: Float) -> Self {
        assert_eq!(mat.rows, mat.cols);
        let orig_mat = mat.clone();
        let mut ml = SparseMatrix::new(mat.rows, mat.cols);
        let mut mu = SparseMatrix::new(mat.rows, mat.cols);
        for (i, j, v) in mat.data.iter() {
            if *i > *j {
                ml.data.push((*i, *j, *v * omega));
            } else if *i < *j {
                mu.data.push((*i, *j, -*v * omega));
            } else {
                ml.data.push((*i, *j, *v));
                mu.data.push((*i, *j, *v * (1.0 - omega)));
            }
        }
        ml.sort();
        mu.sort();
        let iter_mat = ml.l_inv_on_mat(&mu);
        let iter_vec = ml.l_inv_on_vec(b) * omega;
        let eps = 1e-5;
        Self {
            orig_mat,
            iter_mat,
            iter_vec,
            b: b.clone(),
            eps,
        }
    }
    pub fn solve(&self) -> Result<MyVec, &str> {
        let mut res = MyVec(vec![1.0; self.orig_mat.cols]);
        const MAX_ITER: usize = 5000;
        //const MAX_ITER: usize = 50;
        for _ in 0..MAX_ITER {
            res = &self.iter_mat * &res + &self.iter_vec;
            //println!("{}", (&self.orig_mat * &res - &self.b).norm2());
            if (&self.orig_mat * &res - &self.b).norm2() < self.eps {
                return Ok(res);
            }
        }
        Err("Iteration reaches max limit: problem may not converge")
    }
}

#[derive(Debug)]
struct SSORSolver {
    orig_mat: SparseMatrix,
    iter_mat: SparseMatrix,
    iter_vec: MyVec,
    b: MyVec,
    eps: Float,
}
/*impl SSORSolver {
    pub fn new(mat: &SparseMatrix, b: &MyVec, omega: Float) -> Self {
        assert_eq!(mat.rows, mat.cols);
        let orig_mat = mat.clone();
        let mut ml = SparseMatrix::new(mat.rows, mat.cols);
        let mut mu = SparseMatrix::new(mat.rows, mat.cols);
        let mut d_inv = SparseMatrix::new(mat.rows, mat.cols);
        for (i, j, v) in mat.data.iter() {
            if *i > *j {
                ml.data.push((*i, *j, *v * omega));
            } else if *i < *j {
                mu.data.push((*i, *j, *v * omega));
            } else {
                ml.data.push((*i, *j, *v));
                mu.data.push((*i, *j, *v));
                d_inv.data.push((*i,*j,1.0 / *v));
            }
        }
        ml.sort();
        mu.sort();
        let mlu = 1.0 / (omega * (2.0 - omega)) * (&ml * &d_inv * &mu);
        let nlu = &mlu - mat;
        let eps = 1e-5;
        Self {
            orig_mat,
            iter_mat,
            iter_vec,
            b: b.clone(),
            eps,
        }
    }
    pub fn solve(&self) -> Result<MyVec, &str> {
        let mut res = MyVec(vec![1.0; self.orig_mat.cols]);
        const MAX_ITER: usize = 5000;
        //const MAX_ITER: usize = 50;
        for _ in 0..MAX_ITER {
            res = &self.iter_mat * &res + &self.iter_vec;
            //println!("{}", (&self.orig_mat * &res - &self.b).norm2());
            if (&self.orig_mat * &res - &self.b).norm2() < self.eps {
                return Ok(res);
            }
        }
        Err("Iteration reaches max limit: problem may not converge")
    }
}*/

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matrix() {
        let mut mat = SparseMatrix::new(4, 4);
        for i in 0..4 {
            mat.add_entry(i, i, 1.0);
        }
        for i in 0..3 {
            mat.add_entry(i, i + 1, 0.5);
        }
        mat = mat * 2.0;
        let x = MyVec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = &mat * &x;
        assert_eq!(y[0], 4.0);
        assert_eq!(y[1], 7.0);
        assert_eq!(y[2], 10.0);
        assert_eq!(y[3], 8.0);
        assert_eq!(x.norm2(), 1.0 + 4.0 + 9.0 + 16.0);

        let mut mat = SparseMatrix::new(4, 4);
        for i in 0..4 {
            mat.add_entry(i, i, 4.0);
        }
        for i in 1..4 {
            mat.add_entry(i, i - 1, 3.0);
        }
        for i in 2..4 {
            mat.add_entry(i, i - 2, 2.0);
        }
        mat.sort();
        let mt = &mat.transpose();
        let mat2 = mat.l_inv_on_mat(&mt);
        println!("self inv: {:?}", &mat * &mat2);
    }

    #[test]
    fn test_solver() {
        let mut mat = SparseMatrix::new(4, 4);
        for i in 0..4 {
            mat.add_entry(i, i, 1.0);
        }
        for i in 0..3 {
            mat.add_entry(i, i + 1, 0.25);
            mat.add_entry(i + 1, i, 0.25);
        }
        mat.sort();
        let x = MyVec(vec![1.0, 2.0, 3.0, 4.0]);
        let y = &mat * &x;
        let solver = JacobiSolver::new(&mat, &y);
        println!("{:?}", solver);
        let res = solver.solve().unwrap();
        println!("{:?}", res);
        assert!((res - &x).norm2() < 1e-4);
        let solver = SORSolver::new(&mat, &y, 1.5);
        println!("{:?}", solver);
        let res = solver.solve().unwrap();
        println!("{:?}", res);
        assert!((res - &x).norm2() < 1e-4);
    }
}
