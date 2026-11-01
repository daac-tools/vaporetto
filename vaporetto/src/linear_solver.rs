//! Linear classifiers implemented in Rust.
//!
//! The elastic-net solver uses coordinate descent with a one-variable Newton step and a line
//! search (Yuan et al. 2010), as LIBLINEAR's solver of L1-regularized L2-loss SVC with an L2 term
//! added. The other solvers are ports of LIBLINEAR's dual coordinate descent solvers. All of them
//! keep the training data in single precision and need less memory and time than LIBLINEAR.

use alloc::vec;
use alloc::vec::Vec;

/// Training data in columns (compressed sparse columns).
struct Columns {
    offsets: Vec<usize>,
    rows: Vec<u32>,
    values: Vec<f32>,
}

impl Columns {
    /// Makes the columns of the training data, with a bias column of value 1 as the last feature.
    ///
    /// # Arguments
    ///
    /// * `xs` - The features of each instance: (feature ID, value). The IDs are less than
    ///   `n_features`.
    /// * `n_features` - The number of features.
    fn new(xs: &[Vec<(u32, f64)>], n_features: usize) -> Self {
        let n = n_features + 1;
        let mut counts = vec![0usize; n];
        for x in xs {
            for &(fid, _) in x {
                counts[fid as usize] += 1;
            }
        }
        counts[n_features] = xs.len();
        let mut offsets = vec![0usize; n + 1];
        for j in 0..n {
            offsets[j + 1] = offsets[j] + counts[j];
        }
        let nnz = offsets[n];
        let mut rows = vec![0u32; nnz];
        let mut values = vec![0f32; nnz];
        let mut fill = offsets.clone();
        for (i, x) in xs.iter().enumerate() {
            for &(j, v) in x.iter().chain(&[(u32::try_from(n_features).unwrap(), 1.)]) {
                let j = j as usize;
                rows[fill[j]] = i as u32;
                values[fill[j]] = v as f32;
                fill[j] += 1;
            }
        }
        Self {
            offsets,
            rows,
            values,
        }
    }

    /// The number of columns, including the bias.
    fn n(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The instances that have the feature `j`, with the values.
    fn col(&self, j: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.offsets[j]..self.offsets[j + 1];
        self.rows[r.clone()]
            .iter()
            .zip(&self.values[r])
            .map(|(&i, &v)| (i as usize, f64::from(v)))
    }
}

/// The seed of the random number generator for the order of coordinates.
const SEED: u64 = 0x2545_f491_4f6c_dd1d;

/// A small deterministic random number generator for the order of coordinates.
struct XorShift(u64);

impl XorShift {
    /// Returns a random number less than `n`.
    fn next_below(&mut self, n: usize) -> usize {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x % n as u64) as usize
    }
}

/// Solves elastic-net regularized L2-loss SVC and returns the weights.
///
/// # Arguments
///
/// * `cols` - The training data.
/// * `y` - The label of each instance (+1 or -1).
/// * `eps` - The stopping tolerance.
/// * `cost` - The cost C.
/// * `l2` - The weight λ of the L2 term.
fn solve(cols: &Columns, y: &[i8], eps: f64, cost: f64, l2: f64) -> Vec<f64> {
    let l = y.len();
    let w_size = cols.n();
    let max_iter = 1000;
    let max_num_linesearch = 20;
    let sigma = 0.01;
    let mut gmax_old = f64::INFINITY;
    let mut gnorm1_init = -1.0;
    let mut active_size = w_size;
    let mut index: Vec<usize> = (0..w_size).collect();
    let mut w = vec![0f64; w_size];
    // b = 1 - y w^T x
    let mut b = vec![1f64; l];
    // Σ C x_ij^2 (y^2 = 1)
    let xj_sq: Vec<f64> = (0..w_size)
        .map(|j| cols.col(j).map(|(_, v)| cost * v * v).sum())
        .collect();
    let mut rng = XorShift(SEED);

    let mut iter = 0;
    while iter < max_iter {
        let mut gmax_new = 0f64;
        let mut gnorm1_new = 0f64;
        for j in 0..active_size {
            let i = j + rng.next_below(active_size - j);
            index.swap(i, j);
        }
        let mut s = 0;
        while s < active_size {
            let j = index[s];
            let mut g_loss = 0f64;
            let mut h = 0f64;
            for (i, v) in cols.col(j) {
                if b[i] > 0. {
                    let val = v * f64::from(y[i]);
                    let tmp = cost * val;
                    g_loss -= tmp * b[i];
                    h += tmp * val;
                }
            }
            g_loss *= 2.;
            let g = g_loss + l2 * w[j];
            let h = (h * 2. + l2).max(1e-12);
            let gp = g + 1.;
            let gn = g - 1.;
            let mut violation = 0f64;
            if w[j] == 0. {
                if gp < 0. {
                    violation = -gp;
                } else if gn > 0. {
                    violation = gn;
                } else if gp > gmax_old / l as f64 && gn < -gmax_old / l as f64 {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                }
            } else if w[j] > 0. {
                violation = gp.abs();
            } else {
                violation = gn.abs();
            }
            gmax_new = gmax_new.max(violation);
            gnorm1_new += violation;

            // the Newton direction
            let mut d = if gp < h * w[j] {
                -gp / h
            } else if gn > h * w[j] {
                -gn / h
            } else {
                -w[j]
            };
            if d.abs() < 1e-12 {
                s += 1;
                continue;
            }
            let reg = |d: f64| {
                (w[j] + d).abs() - w[j].abs() + l2 / 2. * ((w[j] + d).powi(2) - w[j].powi(2))
            };
            let mut delta = (w[j] + d).abs() - w[j].abs() + g * d;
            let mut d_old = 0.;
            let mut loss_old = 0.;
            let mut num_linesearch = 0;
            while num_linesearch < max_num_linesearch {
                let d_diff = d_old - d;
                let mut cond = reg(d) - sigma * delta;
                let appxcond = xj_sq[j] * d * d + g_loss * d + cond;
                if appxcond <= 0. {
                    for (i, v) in cols.col(j) {
                        b[i] += d_diff * v * f64::from(y[i]);
                    }
                    break;
                }
                let mut loss_new = 0.;
                for (i, v) in cols.col(j) {
                    if num_linesearch == 0 && b[i] > 0. {
                        loss_old += cost * b[i] * b[i];
                    }
                    let b_new = b[i] + d_diff * v * f64::from(y[i]);
                    b[i] = b_new;
                    if b_new > 0. {
                        loss_new += cost * b_new * b_new;
                    }
                }
                cond += loss_new - loss_old;
                if cond <= 0. {
                    break;
                }
                d_old = d;
                d *= 0.5;
                delta *= 0.5;
                num_linesearch += 1;
            }
            w[j] += d;

            // recompute b if the line search takes too many steps
            if num_linesearch >= max_num_linesearch {
                b.iter_mut().for_each(|x| *x = 1.);
                for (jj, &wj) in w.iter().enumerate() {
                    if wj == 0. {
                        continue;
                    }
                    for (i, v) in cols.col(jj) {
                        b[i] -= wj * v * f64::from(y[i]);
                    }
                }
            }
            s += 1;
        }

        if iter == 0 {
            gnorm1_init = gnorm1_new;
        }
        iter += 1;

        if gnorm1_new <= eps * gnorm1_init {
            if active_size == w_size {
                break;
            }
            active_size = w_size;
            gmax_old = f64::INFINITY;
            continue;
        }
        gmax_old = gmax_new;
    }
    w
}

/// Training data in rows (compressed sparse rows).
struct Rows {
    offsets: Vec<usize>,
    cols: Vec<u32>,
    values: Vec<f32>,
}

impl Rows {
    /// Makes the rows of the training data, with a bias feature of value 1 as the last feature.
    ///
    /// # Arguments
    ///
    /// * `xs` - The features of each instance: (feature ID, value). The IDs are less than
    ///   `n_features`.
    /// * `n_features` - The number of features.
    fn new(xs: &[Vec<(u32, f64)>], n_features: usize) -> Self {
        let nnz = xs.iter().map(|x| x.len() + 1).sum();
        let mut offsets = Vec::with_capacity(xs.len() + 1);
        let mut cols = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        offsets.push(0);
        for x in xs {
            for &(fid, v) in x {
                cols.push(fid);
                values.push(v as f32);
            }
            cols.push(u32::try_from(n_features).unwrap());
            values.push(1.);
            offsets.push(cols.len());
        }
        Self {
            offsets,
            cols,
            values,
        }
    }

    /// The features of the instance `i`, with the values.
    fn row(&self, i: usize) -> impl Iterator<Item = (usize, f64)> + '_ {
        let r = self.offsets[i]..self.offsets[i + 1];
        self.cols[r.clone()]
            .iter()
            .zip(&self.values[r])
            .map(|(&j, &v)| (j as usize, f64::from(v)))
    }

    /// The inner product of the instance `i` and the weights `w`.
    fn dot(&self, i: usize, w: &[f64]) -> f64 {
        self.row(i).map(|(j, v)| w[j] * v).sum()
    }

    /// Adds the instance `i` multiplied by `a` to the weights `w`.
    fn axpy(&self, i: usize, a: f64, w: &mut [f64]) {
        for (j, v) in self.row(i) {
            w[j] += a * v;
        }
    }

    /// The squared norm of the instance `i`.
    fn nrm2_sq(&self, i: usize) -> f64 {
        self.row(i).map(|(_, v)| v * v).sum()
    }
}

/// Solves the dual of L2-regularized L2-loss or L1-loss SVC and returns the weights.
///
/// # Arguments
///
/// * `rows` - The training data.
/// * `y` - The label of each instance (+1 or -1).
/// * `w_size` - The number of weights, including the bias.
/// * `eps` - The stopping tolerance.
/// * `cost` - The cost C.
/// * `l1_loss` - Uses the L1 loss instead of the L2 loss.
fn solve_svc_dual(
    rows: &Rows,
    y: &[i8],
    w_size: usize,
    eps: f64,
    cost: f64,
    l1_loss: bool,
) -> Vec<f64> {
    let l = y.len();
    let max_iter = 1000;
    let (diag, upper_bound) = if l1_loss {
        (0., cost)
    } else {
        (0.5 / cost, f64::INFINITY)
    };
    let mut pg_max_old = f64::INFINITY;
    let mut pg_min_old = f64::NEG_INFINITY;
    let mut alpha = vec![0f64; l];
    let mut w = vec![0f64; w_size];
    let qd: Vec<f64> = (0..l).map(|i| diag + rows.nrm2_sq(i)).collect();
    let mut index: Vec<usize> = (0..l).collect();
    let mut active_size = l;
    let mut rng = XorShift(SEED);

    let mut iter = 0;
    while iter < max_iter {
        let mut pg_max_new = f64::NEG_INFINITY;
        let mut pg_min_new = f64::INFINITY;
        for i in 0..active_size {
            let j = i + rng.next_below(active_size - i);
            index.swap(i, j);
        }
        let mut s = 0;
        while s < active_size {
            let i = index[s];
            let yi = f64::from(y[i]);
            let g = yi * rows.dot(i, &w) - 1. + alpha[i] * diag;
            let mut pg = 0.;
            if alpha[i] == 0. {
                if g > pg_max_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if g < 0. {
                    pg = g;
                }
            } else if alpha[i] == upper_bound {
                if g < pg_min_old {
                    active_size -= 1;
                    index.swap(s, active_size);
                    continue;
                } else if g > 0. {
                    pg = g;
                }
            } else {
                pg = g;
            }
            pg_max_new = pg_max_new.max(pg);
            pg_min_new = pg_min_new.min(pg);
            if pg.abs() > 1e-12 {
                let alpha_old = alpha[i];
                alpha[i] = (alpha[i] - g / qd[i]).max(0.).min(upper_bound);
                rows.axpy(i, (alpha[i] - alpha_old) * yi, &mut w);
            }
            s += 1;
        }
        iter += 1;

        if pg_max_new - pg_min_new <= eps {
            if active_size == l {
                break;
            }
            active_size = l;
            pg_max_old = f64::INFINITY;
            pg_min_old = f64::NEG_INFINITY;
            continue;
        }
        pg_max_old = if pg_max_new <= 0. {
            f64::INFINITY
        } else {
            pg_max_new
        };
        pg_min_old = if pg_min_new >= 0. {
            f64::NEG_INFINITY
        } else {
            pg_min_new
        };
    }
    w
}

/// Solves the dual of L2-regularized logistic regression and returns the weights.
///
/// # Arguments
///
/// * `rows` - The training data.
/// * `y` - The label of each instance (+1 or -1).
/// * `w_size` - The number of weights, including the bias.
/// * `eps` - The stopping tolerance.
/// * `cost` - The cost C.
fn solve_lr_dual(rows: &Rows, y: &[i8], w_size: usize, eps: f64, cost: f64) -> Vec<f64> {
    let l = y.len();
    let max_iter = 1000;
    let max_inner_iter = 100;
    let mut innereps = 1e-2;
    let innereps_min = eps.min(1e-8);
    let eta = 0.1;
    // alpha and C - alpha
    let mut alpha: Vec<(f64, f64)> = (0..l)
        .map(|_| {
            let a = (0.001 * cost).min(1e-8);
            (a, cost - a)
        })
        .collect();
    let mut w = vec![0f64; w_size];
    let xtx: Vec<f64> = (0..l).map(|i| rows.nrm2_sq(i)).collect();
    for (i, &(a, _)) in alpha.iter().enumerate() {
        rows.axpy(i, f64::from(y[i]) * a, &mut w);
    }
    let mut index: Vec<usize> = (0..l).collect();
    let mut rng = XorShift(SEED);

    let mut iter = 0;
    while iter < max_iter {
        for i in 0..l {
            let j = i + rng.next_below(l - i);
            index.swap(i, j);
        }
        let mut newton_iter = 0;
        let mut gmax = 0f64;
        for &i in &index {
            let yi = f64::from(y[i]);
            let a = xtx[i];
            let b = yi * rows.dot(i, &w);
            // Minimizes g_1(z) or g_2(z):
            // g_t(z) = z log z + (C - z) log(C - z) + 0.5 a (z - alpha_old)^2 + sign b (z - alpha_old)
            let (a1, a2) = alpha[i];
            let (alpha_old, sign) = if 0.5 * a * (a2 - a1) + b < 0. {
                (a2, -1.)
            } else {
                (a1, 1.)
            };
            let mut z = alpha_old;
            if cost - z < 0.5 * cost {
                z *= 0.1;
            }
            let mut gp = a * (z - alpha_old) + sign * b + (z / (cost - z)).ln();
            gmax = gmax.max(gp.abs());

            // Newton method on the sub-problem
            let mut inner_iter = 0;
            while inner_iter <= max_inner_iter {
                if gp.abs() < innereps {
                    break;
                }
                let gpp = a + cost / (cost - z) / z;
                let tmpz = z - gp / gpp;
                if tmpz <= 0. {
                    z *= eta;
                } else {
                    z = tmpz;
                }
                gp = a * (z - alpha_old) + sign * b + (z / (cost - z)).ln();
                newton_iter += 1;
                inner_iter += 1;
            }
            if inner_iter > 0 {
                alpha[i] = if sign > 0. {
                    (z, cost - z)
                } else {
                    (cost - z, z)
                };
                rows.axpy(i, sign * (z - alpha_old) * yi, &mut w);
            }
        }
        iter += 1;

        if gmax < eps {
            break;
        }
        if newton_iter <= l / 10 {
            innereps = innereps_min.max(0.1 * innereps);
        }
    }
    w
}

/// The problems solved here.
#[derive(Clone, Copy, Debug)]
pub enum Problem {
    /// The dual of L2-regularized L2-loss SVC.
    L2RegularizedL2LossSvcDual { cost: f64 },
    /// The dual of L2-regularized L1-loss SVC.
    L2RegularizedL1LossSvcDual { cost: f64 },
    /// Elastic-net regularized L2-loss SVC:
    /// `min_w ||w||_1 + (l2 / 2) ||w||_2^2 + C Σ max(0, 1 - y w^T x)^2`.
    ElasticNetL2LossSvc { cost: f64, l2: f64 },
    /// The dual of L2-regularized logistic regression.
    L2RegularizedLogisticDual { cost: f64 },
}

/// A trained linear model, with the same accessors as LIBLINEAR's model.
pub struct LinearModel {
    labels: Vec<i32>,
    n_features: usize,
    /// One weight vector or one per class, each with the bias at the end.
    ws: Vec<Vec<f64>>,
}

impl LinearModel {
    /// Trains a model as LIBLINEAR's `train()` with bias 1.
    ///
    /// # Arguments
    ///
    /// * `ys` - The labels.
    /// * `xs` - The features of each instance: (feature ID, value). The IDs start from 0.
    /// * `eps` - The stopping tolerance.
    /// * `problem` - The problem and its parameters.
    pub fn train(ys: &[f64], xs: Vec<Vec<(u32, f64)>>, eps: f64, problem: Problem) -> Self {
        let n_features = xs
            .iter()
            .flat_map(|x| x.iter().map(|&(fid, _)| fid as usize + 1))
            .max()
            .unwrap_or(0);
        let w_size = n_features + 1;
        // The input rows are not needed once the solver's layout is made (memory for large data).
        let (cols, rows) = match problem {
            Problem::ElasticNetL2LossSvc { .. } => (Some(Columns::new(&xs, n_features)), None),
            _ => (None, Some(Rows::new(&xs, n_features))),
        };
        drop(xs);
        let mut labels: Vec<i32> = vec![];
        let ys: Vec<i32> = ys.iter().map(|&y| y as i32).collect();
        for &y in &ys {
            if !labels.contains(&y) {
                labels.push(y);
            }
        }
        let l = ys.len();
        let binary = |positive: i32| -> Vec<i8> {
            ys.iter()
                .map(|&y| if y == positive { 1 } else { -1 })
                .collect()
        };
        // LIBLINEAR's stopping tolerance for the primal solvers.
        let primal_tol = |y: &[i8]| {
            let pos = y.iter().filter(|&&y| y > 0).count();
            eps * pos.min(l - pos).max(1) as f64 / l as f64
        };
        let solve_binary = |y: &[i8]| match problem {
            Problem::ElasticNetL2LossSvc { cost, l2 } => {
                solve(cols.as_ref().unwrap(), y, primal_tol(y), cost, l2)
            }
            Problem::L2RegularizedL2LossSvcDual { cost } => {
                solve_svc_dual(rows.as_ref().unwrap(), y, w_size, eps, cost, false)
            }
            Problem::L2RegularizedL1LossSvcDual { cost } => {
                solve_svc_dual(rows.as_ref().unwrap(), y, w_size, eps, cost, true)
            }
            Problem::L2RegularizedLogisticDual { cost } => {
                solve_lr_dual(rows.as_ref().unwrap(), y, w_size, eps, cost)
            }
        };
        let ws = if labels.len() <= 2 {
            vec![solve_binary(&binary(labels[0]))]
        } else {
            labels.iter().map(|&c| solve_binary(&binary(c))).collect()
        };
        Self {
            labels,
            n_features,
            ws,
        }
    }

    /// The labels in the order of their first appearance.
    pub fn labels(&self) -> &[i32] {
        &self.labels
    }

    /// The number of features, excluding the bias.
    pub fn num_features(&self) -> usize {
        self.n_features
    }

    /// The weight of the feature `j` for the class at `label_index`.
    fn w(&self, j: usize, label_index: usize) -> f64 {
        if self.ws.len() == 1 {
            match label_index {
                0 => self.ws[0][j],
                1 => -self.ws[0][j],
                _ => 0.,
            }
        } else {
            self.ws.get(label_index).map_or(0., |w| w[j])
        }
    }

    /// The weight of `feature` for the class at `label_index` in `labels()`.
    pub fn feature_coefficient(&self, feature: usize, label_index: usize) -> f64 {
        if feature >= self.n_features {
            return 0.;
        }
        self.w(feature, label_index)
    }

    /// The bias for the class at `label_index` in `labels()`.
    pub fn label_bias(&self, label_index: usize) -> f64 {
        self.w(self.n_features, label_index)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Trains a problem of two instances with their own features. By symmetry, the optimal weights
    /// for label 1 are (a, -a) and the bias is 0, where `a` depends on the problem.
    fn train_symmetric(problem: Problem) -> LinearModel {
        LinearModel::train(&[1., 0.], vec![vec![(0, 1.)], vec![(1, 1.)]], 1e-6, problem)
    }

    fn assert_symmetric(model: &LinearModel, a: f64) {
        assert_eq!(&[1, 0], model.labels());
        assert_eq!(2, model.num_features());
        for (feature, expected) in [(0, a), (1, -a)] {
            let w = model.feature_coefficient(feature, 0);
            assert!(
                (w - expected).abs() < 1e-3,
                "w[{feature}] = {w}, expected {expected}"
            );
            // The weights for label 0 are the opposite.
            assert_eq!(-w, model.feature_coefficient(feature, 1));
        }
        let bias = model.label_bias(0);
        assert!(bias.abs() < 1e-3, "bias = {bias}");
    }

    #[test]
    fn check_l1r_l2loss_svc() {
        // 2 a + 2 C (1 - a)^2 is the minimum at a = 1 - 1 / (2 C).
        let model = train_symmetric(Problem::ElasticNetL2LossSvc { cost: 1., l2: 0. });
        assert_symmetric(&model, 0.5);
    }

    #[test]
    fn check_elastic_net_l2loss_svc() {
        // 2 a + λ a^2 + 2 C (1 - a)^2 is the minimum at a = (2 C - 1) / (λ + 2 C).
        let model = train_symmetric(Problem::ElasticNetL2LossSvc { cost: 1., l2: 1. });
        assert_symmetric(&model, 1. / 3.);
    }

    #[test]
    fn check_l2r_l2loss_svc_dual() {
        // a^2 + 2 C (1 - a)^2 is the minimum at a = 2 C / (1 + 2 C).
        let model = train_symmetric(Problem::L2RegularizedL2LossSvcDual { cost: 1. });
        assert_symmetric(&model, 2. / 3.);
    }

    #[test]
    fn check_l2r_l1loss_svc_dual() {
        // a^2 + 2 C (1 - a) is the minimum at a = C (C < 1).
        let model = train_symmetric(Problem::L2RegularizedL1LossSvcDual { cost: 0.5 });
        assert_symmetric(&model, 0.5);
    }

    #[test]
    fn check_l2r_lr_dual() {
        // a^2 + 2 C log(1 + exp(-a)) is the minimum at a (1 + exp(a)) = C, found by bisection.
        let cost = 1.;
        let mut lo = 0.;
        let mut hi = cost;
        for _ in 0..100 {
            let mid = (lo + hi) / 2.;
            if mid * (1. + f64::exp(mid)) < cost {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let model = train_symmetric(Problem::L2RegularizedLogisticDual { cost });
        assert_symmetric(&model, lo);
    }
}
