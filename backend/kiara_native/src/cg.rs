//! Preconditioned CG with divergence-watchdog parity to the Python
//! `_cg_div_callback`/`_cg_watchdog` path.
//!
//! Semantics replicated exactly:
//! - rtol check `||r|| <= rtol*||b||` after each iterate update
//! - watchdog sampled every 40 iters on the EXPLICIT residual
//!   `||b - A@x||` (the recursive r drifts — same extra SpMV as Python)
//! - abort classes: `xnorm` (|x|>1e6 or non-finite ratio), `blow`
//!   (ratio > max(5*best,5) for 3 straight samples), `flat` (no
//!   improvement for 10 samples with ratio>1)
//! - `maxiter` iterations max; status converged|maxiter|diverged
//!
//! The preconditioner stays a Python callable (pyamg/SSOR/diagonal all
//! expose matvec) invoked once per iteration; the GIL is released around
//! the compute segment so parallel realisation solves still overlap.
//! Sequential dots/axpy keep the FP ordering identical to the scipy path.

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::prelude::*;

const WATCH_SAMPLE: usize = 40;
const XNORM_CAP: f64 = 1e6;
const BLOW_MULT: f64 = 5.0;
const BLOW_STREAK: usize = 3;
const FLAT_STREAK: usize = 10;

fn matvec(ip: &[i64], ix: &[i32], d: &[f64], x: &[f64], y: &mut [f64]) {
    for (r, v) in y.iter_mut().enumerate() {
        let mut acc = 0.0;
        for q in ip[r]..ip[r + 1] {
            acc += d[q as usize] * x[ix[q as usize] as usize];
        }
        *v = acc;
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(u, v)| u * v).sum()
}

#[allow(clippy::too_many_arguments)]
#[pyfunction]
#[pyo3(signature = (indptr, indices, data, b, x0=None, rtol=1e-6, maxiter=4000, m_apply=None, threads=0))]
pub fn cg_solve<'py>(
    py: Python<'py>,
    indptr: PyReadonlyArray1<i64>,
    indices: PyReadonlyArray1<i32>,
    data: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    x0: Option<PyReadonlyArray1<f64>>,
    rtol: f64,
    maxiter: usize,
    m_apply: Option<Py<PyAny>>,
    threads: usize,
) -> PyResult<(Bound<'py, PyArray1<f64>>, usize, &'static str, f64, &'static str)> {
    let ip = indptr.as_slice()?;
    let ix = indices.as_slice()?;
    let d = data.as_slice()?;
    let bs = b.as_slice()?;
    let n = bs.len();
    if ip.len() != n + 1 || ix.len() != d.len() {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "malformed CSR (indptr len / indices:data mismatch)",
        ));
    }
    // int32 column indices wrap silently above 2^31 rows — reject loudly
    // instead of solving a corrupted system
    if ix.iter().any(|&j| j < 0 || j as usize >= n) {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "CSR column index out of range",
        ));
    }
    let mut x = match x0 {
        Some(ref xa) => {
            let s = xa.as_slice()?;
            if s.len() != n {
                return Err(pyo3::exceptions::PyValueError::new_err("x0 length mismatch"));
            }
            s.to_vec()
        }
        None => vec![0.0f64; n],
    };
    let bnorm = dot(bs, bs).sqrt().max(1e-300);
    let thresh = rtol * bnorm;
    let t = threads.max(1);
    let mcall = m_apply.filter(|m| !m.is_none(py));

    // r = b - A@x
    let mut r = vec![0.0f64; n];
    py.detach(|| {
        let mut ax = vec![0.0f64; n];
        matvec(ip, ix, d, &x, &mut ax);
        for i in 0..n {
            r[i] = bs[i] - ax[i];
        }
    });

    let mut z = vec![0.0f64; n];
    let mut p = vec![0.0f64; n];
    let mut q = vec![0.0f64; n];
    let apply_m = |rvec: &[f64], zvec: &mut [f64]| -> PyResult<()> {
        match &mcall {
            None => zvec.copy_from_slice(rvec),
            Some(m) => {
                let rin = PyArray1::from_slice(py, rvec);
                let out = m.call1(py, (rin,))?;
                let arr: Bound<'_, PyArray1<f64>> = out.extract(py)?;
                let sl = unsafe { arr.as_slice()? };
                if sl.len() != n {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "preconditioner output length mismatch",
                    ));
                }
                zvec.copy_from_slice(sl);
            }
        }
        Ok(())
    };
    apply_m(&r, &mut z)?;
    p.copy_from_slice(&z);
    let mut rz = dot(&r, &z);

    let mut best = f64::INFINITY;
    let mut bad = 0usize;
    let mut flat = 0usize;
    let mut it = 0usize;
    let mut status = "maxiter";
    let mut kind = "";
    let mut ratio_out = dot(&r, &r).sqrt() / bnorm;

    while it < maxiter {
        it += 1;
        // one PCG step — GIL released; vectors are plain memory
        let rnorm = py.detach(|| {
            matvec_par(ip, ix, d, &p, &mut q, t);
            let pq = dot(&p, &q);
            let alpha = if pq != 0.0 { rz / pq } else { f64::NAN };
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * q[i];
            }
            dot(&r, &r).sqrt()
        });
        ratio_out = rnorm / bnorm;
        if rnorm <= thresh {
            status = "converged";
            break;
        }
        if it % WATCH_SAMPLE == 0 {
            // explicit residual like the Python watchdog (recursive r drifts)
            let (xnorm, ratio_expl) = py.detach(|| {
                let mut ax = vec![0.0f64; n];
                matvec_par(ip, ix, d, &x, &mut ax, t);
                let mut res = 0.0f64;
                for i in 0..n {
                    let e = bs[i] - ax[i];
                    res += e * e;
                }
                let xn = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                (xn, res.sqrt() / bnorm)
            });
            ratio_out = ratio_expl;
            if xnorm > XNORM_CAP || !ratio_expl.is_finite() {
                status = "diverged";
                kind = "xnorm";
                break;
            }
            if ratio_expl < best * 0.9 {
                best = ratio_expl;
                bad = 0;
                flat = 0;
            } else {
                flat += 1;
                if ratio_expl > (BLOW_MULT * best).max(BLOW_MULT) {
                    bad += 1;
                }
                if bad >= BLOW_STREAK {
                    status = "diverged";
                    kind = "blow";
                    break;
                }
                if flat >= FLAT_STREAK && ratio_expl > 1.0 {
                    status = "diverged";
                    kind = "flat";
                    break;
                }
            }
        }
        apply_m(&r, &mut z)?;
        let rz_new = dot(&r, &z);
        let beta = if rz != 0.0 { rz_new / rz } else { 0.0 };
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        rz = rz_new;
    }
    Ok((
        PyArray1::from_vec(py, x),
        it,
        status,
        ratio_out,
        kind,
    ))
}

fn matvec_par(ip: &[i64], ix: &[i32], d: &[f64], x: &[f64], y: &mut [f64], t: usize) {
    crate::util::par_chunks(y, t, |chunk, off| {
        for (r, v) in chunk.iter_mut().enumerate() {
            let row = off + r;
            let mut acc = 0.0;
            for q in ip[row]..ip[row + 1] {
                acc += d[q as usize] * x[ix[q as usize] as usize];
            }
            *v = acc;
        }
    });
}
