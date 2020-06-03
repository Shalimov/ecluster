use ndarray::{array, Array, Array1, Array2, ArrayView1, ArrayView2, Zip};

pub struct ECluster<'a> {
    alpha: f64,
    beta: f64,
    edge_divider: u8,
    mid_clusters: ArrayView2<'a, i16>,
    estimated_clusters: Vec<(f64, Array1<i16>)>,
}

impl<'a> ECluster<'a> {
    pub fn euclidean_distance(zh: ArrayView1<i16>, vi: ArrayView1<i16>) -> f64 {
        Zip::from(zh)
            .and(vi)
            .fold(0f64, |acc, x1, x2| acc + ((x1 - x2) as f64).powf(2f64))
            .sqrt()
    }

    fn calculate_bias(&self, vi: ArrayView1<i16>) -> f64 {
        let mut bias = 0.;

        for (potential, location) in &self.estimated_clusters {
            bias += potential
                * (-1. * self.beta * ECluster::euclidean_distance(location.view(), vi)).exp();
        }

        bias
    }

    fn eval_potential(&self, zh: ArrayView1<i16>) -> f64 {
        let mut potential: f64 = 0.;

        for point in self.mid_clusters.genrows() {
            potential += (-1. * self.alpha * ECluster::euclidean_distance(zh, point)).exp();
        }

        potential
    }

    fn evaluate_maximum_potential(&self) -> (f64, Array1<i16>) {
        let mut bias: f64;
        let mut potential: f64;
        let mut maximum_potential = 0.;
        let mut maximum_potential_position: Array1<i16> = array![0];

        for xi in self.mid_clusters.genrows() {
            bias = self.calculate_bias(xi);

            potential = self.eval_potential(xi) - bias;

            if potential > maximum_potential {
                maximum_potential = potential;
                maximum_potential_position = xi.into_owned();
            }
        }

        (maximum_potential, maximum_potential_position)
    }

    fn to_array2d(&self) -> Array2<i16> {
        let points: Vec<i16> = self
            .estimated_clusters
            .iter()
            .map(|(_p, point)| point.iter().cloned())
            .flatten()
            .collect();

        let (_, loc) = &self.estimated_clusters[0];

        let shape = (self.estimated_clusters.len(), loc.len());
        Array::from_shape_vec(shape, points).expect("Failed to convert to Array<i16, Ix2>")
    }

    pub fn estimate(
        points: ArrayView2<'a, i16>,
        edge_div: u8,
        alpha: f64,
        beta: f64,
    ) -> Array2<i16> {
        let mut cluster = ECluster {
            alpha,
            beta,
            mid_clusters: points,
            edge_divider: edge_div,
            estimated_clusters: Vec::new(),
        };

        let mut iterations = 0;

        let (max_potential, max_loc) = cluster.evaluate_maximum_potential();
        let edge: f64 = max_potential / cluster.edge_divider as f64;

        cluster.estimated_clusters.push((max_potential, max_loc));

        loop {
            let (max_potential, max_loc) = cluster.evaluate_maximum_potential();

            if iterations > 99 || edge > max_potential {
                break;
            }

            cluster.estimated_clusters.push((max_potential, max_loc));
            iterations += 1;
        }

        cluster.to_array2d()
    }
}

#[cfg(test)]
mod tests {
    use crate::estimator::*;
    use ndarray::array;

    #[test]
    fn estimated_cluster_centers() {
        let mid_centers: Array2<i16> = array![
            [1, 3, 4],
            [2, 5, 6],
            [75, 34, 12],
            [234, 42, 122],
            [4, 2, 1],
            [99, 12, 40],
            [57, 27, 11],
            [20, 30, 20],
        ];

        let estimated = ECluster::estimate(mid_centers.view(), 2, 0.19, 0.2);

        assert_eq!(
            estimated,
            array![[1, 3, 4], [57, 27, 11], [75, 34, 12], [20, 30, 20]]
        )
    }

    #[test]
    fn equal_points_distance_zero() {
        let zh1 = Array1::ones(3);
        let zh2 = Array1::ones(3);

        assert_eq!(ECluster::euclidean_distance(zh1.view(), zh2.view()), 0.);
    }

    #[test]
    fn basis_points_distance_one() {
        let zh1 = Array1::zeros(4);
        let zh2 = Array1::ones(4);

        assert_eq!(ECluster::euclidean_distance(zh1.view(), zh2.view()), 2.);
    }

    #[test]
    fn random_vectors_distance_test() {
        let zh1 = array![5, 5];
        let zh2 = array![2, 1];

        assert_eq!(ECluster::euclidean_distance(zh1.view(), zh2.view()), 5.);
    }
}
