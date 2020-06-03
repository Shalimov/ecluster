use numpy::{IntoPyArray, PyArray2};
use pyo3::prelude::{pymodule, Py, PyModule, PyResult, Python};

mod estimator;
use estimator::ECluster;

#[pymodule]
fn cluster_estimator(_py: Python, m: &PyModule) -> PyResult<()> {
    #[pyfn(m, "estimate")]
    fn estimate_cluster_centers_py(
        py: Python,
        points: &PyArray2<i16>,
        edge_div: u8,
        alpha: f64,
        beta: f64,
    ) -> Py<PyArray2<i16>> {
        let points = points.as_array();
        ECluster::estimate(points, edge_div, alpha, beta)
            .into_pyarray(py)
            .to_owned()
    }

    Ok(())
}
