//! Parallel CSR SpMV exposed as a scipy-LinearOperator-compatible matvec.
//!
//! The CG driver, preconditioner (AMG/SSOR/diagonal), watchdog and all
//! solver telemetry stay in Python — only the per-iteration matvec moves
//! native, which is where the single-threaded scipy cost lives. The CSR
//! arrays are copied once at construction (O(nnz), far below the thousands
//! of matvecs a solve runs) so the kernel never borrows GIL-owned memory.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::util::par_chunks;

#[pyclass]
pub struct SpMv {
    // kept alive so the borrowed slices stay valid (zero-copy numpy refs)
    indptr: Vec<i64>,
    indices: Vec<i32>,
    data: Vec<f64>,
    pub n: usize,
}

#[pymethods]
impl SpMv {
    #[new]
    fn new(
        indptr: PyReadonlyArray1<i64>,
        indices: PyReadonlyArray1<i32>,
        data: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let ip = indptr.as_slice()?;
        let ix = indices.as_slice()?;
        let d = data.as_slice()?;
        if ix.len() != d.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "indices/data length mismatch",
            ));
        }
        Ok(SpMv {
            indptr: ip.to_vec(),
            indices: ix.to_vec(),
            data: d.to_vec(),
            n: ip.len().saturating_sub(1),
        })
    }

    /// y = A @ x, parallel over rows — deterministic per-row dot product.
    #[pyo3(signature = (x, threads=None))]
    fn matvec<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray1<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let xs = x.as_slice()?;
        if xs.len() != self.n {
            return Err(pyo3::exceptions::PyValueError::new_err("x length mismatch"));
        }
        let t = threads.unwrap_or(0).max(1);
        let (ip, ix, d, n) = (&self.indptr, &self.indices, &self.data, self.n);
        let y = py.detach(move || {
            let mut y = vec![0.0f64; n];
            par_chunks(&mut y, t, |chunk, off| {
                for (r, v) in chunk.iter_mut().enumerate() {
                    let row = off + r;
                    let mut acc = 0.0;
                    for q in ip[row]..ip[row + 1] {
                        acc += d[q as usize] * xs[ix[q as usize] as usize];
                    }
                    *v = acc;
                }
            });
            y
        });
        Ok(PyArray1::from_vec(py, y))
    }
}
