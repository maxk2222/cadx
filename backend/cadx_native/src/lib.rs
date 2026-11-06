use pyo3::prelude::*;

mod csr;
mod mesh;
mod spmv;
mod util;

/// Optional native kernels for cadx — geometry queries (BVH contains/ray/
/// scanline), CSR assembly, parallel SpMV. Python falls back whenever this
/// module is absent; see cad/cad/native.py.
#[pymodule]
fn cadx_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<mesh::MeshIndex>()?;
    m.add_class::<csr::Assembler>()?;
    m.add_class::<spmv::SpMv>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
