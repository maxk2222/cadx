//! CSR assembly against a FIXED sparsity pattern.
//!
//! Python builds K via coo_matrix((sK,(iK,jK))).tocsr() + sum_duplicates —
//! for a full-domain SIMP step that's ~nelem*576 triplets allocated and
//! re-sorted every iteration. Here the (iK,jK) pattern is canonicalised
//! ONCE (sort + dedup → indptr/indices + per-entry scatter target), then
//! each `fill(sK)` is a single deterministic scatter pass: every CSR slot
//! accumulates its contributing entries in input (element-major) order.

use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::util::par_chunks;

#[pyclass]
pub struct Assembler {
    pub ndofs: usize,
    indptr: Vec<i64>,
    indices: Vec<i32>,
    // per CSR slot: contiguous range of source entries into sK
    src_start: Vec<i64>,
    // entry ids grouped by slot, in increasing entry order (deterministic sum)
    src_list: Vec<i64>,
    nnz: usize,
}

#[pymethods]
impl Assembler {
    /// `iK`/`jK` are the (entry-major) COO row/col index vectors — e.g.
    /// np.repeat(edof,24).ravel() / np.tile(edof,(1,24)).ravel().
    #[new]
    #[allow(non_snake_case)]
    fn new(iK: PyReadonlyArray1<i64>, jK: PyReadonlyArray1<i64>, ndofs: i64) -> PyResult<Self> {
        let i = iK.as_slice()?;
        let j = jK.as_slice()?;
        if i.len() != j.len() {
            return Err(pyo3::exceptions::PyValueError::new_err("iK/jK length mismatch"));
        }
        let ndofs = ndofs as usize;
        let n = i.len();

        // canonical ordering: sort entry ids by (row, col)
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by_key(|&k| (i[k as usize], j[k as usize]));

        let mut indptr = vec![0i64; ndofs + 1];
        let mut indices: Vec<i32> = Vec::with_capacity(n);
        let mut scatter = vec![0i64; n];
        let mut prev = (-1i64, -1i64);
        for &k in &order {
            let key = (i[k as usize], j[k as usize]);
            if key != prev {
                if key.0 < 0 || key.0 >= ndofs as i64 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "iK entry out of bounds for ndofs",
                    ));
                }
                indices.push(key.1 as i32);
                indptr[key.0 as usize + 1] += 1;
                prev = key;
            }
            scatter[k as usize] = indices.len() as i64 - 1;
        }
        for r in 0..ndofs {
            indptr[r + 1] += indptr[r];
        }
        let nnz = indices.len();

        // transpose scatter -> per-slot source lists in entry order
        let mut src_start = vec![0i64; nnz + 1];
        for &s in &scatter {
            src_start[s as usize + 1] += 1;
        }
        for s in 0..nnz {
            src_start[s + 1] += src_start[s];
        }
        let mut src_list = vec![0i64; n];
        let mut cur = src_start.clone();
        for (k, &s) in scatter.iter().enumerate() {
            src_list[cur[s as usize] as usize] = k as i64;
            cur[s as usize] += 1;
        }

        Ok(Assembler {
            ndofs,
            indptr,
            indices,
            src_start,
            src_list,
            nnz,
        })
    }

    #[getter]
    fn indptr<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i64>> {
        PyArray1::from_vec(py, self.indptr.clone())
    }

    #[getter]
    fn indices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<i32>> {
        PyArray1::from_vec(py, self.indices.clone())
    }

    #[getter]
    fn nnz(&self) -> usize {
        self.nnz
    }

    /// Scatter element-stiffness values into CSR data — `data[p] =
    /// sum(sK[e] for e in slot p's source list)`, parallel over slots.
    /// Deterministic: each slot accumulates in increasing entry order.
    #[pyo3(signature = (sK, threads=None))]
    #[allow(non_snake_case)]
    fn fill<'py>(
        &self,
        py: Python<'py>,
        sK: PyReadonlyArray1<f64>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let s = sK.as_slice()?;
        if s.len() != self.src_list.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "sK length does not match the assembler's pattern",
            ));
        }
        let t = threads.unwrap_or(0).max(1);
        let (src_start, src_list) = (&self.src_start, &self.src_list);
        let nnz = self.nnz;
        let data = py.detach(move || {
            let mut data = vec![0.0f64; nnz];
            par_chunks(&mut data, t, |chunk, off| {
                for (p, v) in chunk.iter_mut().enumerate() {
                    let pos = off + p;
                    let mut acc = 0.0;
                    for q in src_start[pos]..src_start[pos + 1] {
                        acc += s[src_list[q as usize] as usize];
                    }
                    *v = acc;
                }
            });
            data
        });
        Ok(PyArray1::from_vec(py, data))
    }

    /// sK for the elementwise broadcast `elem_E[:,None,None] * ke[None]` or
    /// `elem_E[:,None,None] * ke_table[cls]` — computed in Rust so `fill`
    /// gets one contiguous input with no Python temporaries.
    /// `elem_E` = per-element E (len nelem), `ke` flat (576,) isotropic or
    /// `ke_table` (ncls,24,24) with `cls` (nelem,).
    #[pyo3(signature = (elem_E, ke, cls=None, threads=None))]
    #[allow(non_snake_case)]
    fn values<'py>(
        &self,
        py: Python<'py>,
        elem_E: PyReadonlyArray1<f64>,
        ke: PyReadonlyArray2<f64>,
        cls: Option<PyReadonlyArray1<i64>>,
        threads: Option<usize>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let e = elem_E.as_slice()?;
        let ke_arr = ke.as_array();
        let nelem = e.len();
        let t = threads.unwrap_or(0).max(1);
        let n = nelem * 576;
        if n != self.src_list.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "elem_E length does not match the assembler's pattern",
            ));
        }
        let flat: Vec<f64> = ke_arr.iter().cloned().collect();
        if flat.is_empty() || flat.len() % 576 != 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "ke must be (24,24) or (ncls,24,24)",
            ));
        }
        let ncls = flat.len() / 576;
        let c: Option<Vec<i64>> = match cls {
            None => {
                if ncls != 1 {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "ke (ncls>1) requires cls",
                    ));
                }
                None
            }
            Some(cls) => {
                let c = cls.as_slice()?;
                if c.len() != nelem {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "cls length must equal elem_E",
                    ));
                }
                if c.iter().any(|&k| k < 0 || k as usize >= ncls) {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "cls values out of bounds",
                    ));
                }
                Some(c.to_vec())
            }
        };
        let s = py.detach(move || {
            let mut s = vec![0.0f64; n];
            par_chunks(&mut s, t, |chunk, off| {
                match &c {
                    None => {
                        for (i, v) in chunk.iter_mut().enumerate() {
                            *v = e[(off + i) / 576] * flat[(off + i) % 576];
                        }
                    }
                    Some(c) => {
                        for (i, v) in chunk.iter_mut().enumerate() {
                            let el = (off + i) / 576;
                            *v = e[el] * flat[c[el] as usize * 576 + (off + i) % 576];
                        }
                    }
                }
            });
            s
        });
        Ok(PyArray1::from_vec(py, s))
    }
}
