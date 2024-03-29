use std::collections::HashMap;

use nalgebra::{Const, DMatrix, DVector, Dyn};
use rayon::prelude::*;

use crate::io::{gwas::IGwasResults, gwas::IntermediateResults, matrix::LabeledMatrix};
use crate::stats::sumstats::compute_neg_log_pvalue;
use crate::util::ProcessingStats;

#[derive(Clone)]
pub struct RunningSufficientStats {
    pub beta: DMatrix<f32>,
    pub gpv: DVector<f32>,
    pub sample_sizes: DVector<i32>,

    cov: DMatrix<f32>,  // Partial covariance matrix of the features
    fpv: DVector<f32>,  // Partial variance vector of the features
    proj: DMatrix<f32>, // Matrix of the projection coefficients

    n_covar: usize,
    chunksize: usize,

    n_features: usize,
    n_projections: usize,
    phenotype_id_to_idx: HashMap<String, usize>,

    variant_ids: Option<Vec<String>>,
    projection_ids: Vec<String>,

    n_features_seen: usize,
}

// Add a method on RunningSufficientStats that takes some GWAS summary statistics and updates the
// state
impl RunningSufficientStats {
    pub fn new(
        proj: &LabeledMatrix,
        cov: &LabeledMatrix,
        n_covar: usize,
        chunksize: usize,
    ) -> Self {
        let n_features = proj.matrix.nrows();
        let n_projections = proj.matrix.ncols();

        // Check that cov is n_features x n_features
        assert_eq!(
            cov.matrix.nrows(),
            n_features,
            "Covariance matrix has wrong shape, expected {} x {}, got {} x {}",
            n_features,
            n_features,
            cov.matrix.nrows(),
            cov.matrix.ncols()
        );
        assert_eq!(
            cov.matrix.ncols(),
            n_features,
            "Covariance matrix has wrong shape, expected {} x {}, got {} x {}",
            n_features,
            n_features,
            cov.matrix.nrows(),
            cov.matrix.ncols()
        );

        // Phenotype_id_to_idx is a hashmap basically of an enumeration of the phenotype ids
        let phenotype_id_to_idx = proj
            .row_labels
            .iter()
            .enumerate()
            .map(|(i, x)| (x.clone(), i))
            .collect();

        RunningSufficientStats {
            beta: DMatrix::zeros(chunksize, n_projections),
            gpv: DVector::zeros(chunksize),
            sample_sizes: DVector::zeros(chunksize),
            cov: cov.matrix.clone(),
            fpv: cov.matrix.diagonal(),
            proj: proj.matrix.clone(),
            n_covar,
            n_features,
            n_projections,
            chunksize,
            phenotype_id_to_idx,
            variant_ids: None,
            projection_ids: proj.col_labels.clone(),
            n_features_seen: 0,
        }
    }

    pub fn clear_chunk(&mut self, new_chunksize: usize) {
        if new_chunksize != self.chunksize {
            self.beta = DMatrix::zeros(new_chunksize, self.n_projections);
            self.gpv = DVector::zeros(new_chunksize);
            self.sample_sizes = DVector::zeros(new_chunksize);
            self.chunksize = new_chunksize;
        } else {
            self.beta.fill(0.0);
            self.gpv.fill(0.0);
            self.sample_sizes.fill(0);
        }
        self.n_features_seen = 0;
    }

    pub fn build_processing_stats(&self) -> ProcessingStats {
        ProcessingStats {
            n_variants: self.beta.nrows(),
            proj: self.proj.clone(),
            fpv: self.fpv.clone(),
            phenotype_id_to_idx: self.phenotype_id_to_idx.clone(),
            n_covar: self.n_covar,
        }
    }

    pub fn update(&mut self, gwas_results: &IntermediateResults) {
        if self.n_features_seen == 0 {
            self.sample_sizes = gwas_results.sample_sizes.clone();
            self.variant_ids = Some(gwas_results.variant_ids.clone());
        } else {
            self.sample_sizes = self.sample_sizes.inf(&gwas_results.sample_sizes);
            // Check that the variant ids match. Add a runtime error message if not
            assert_eq!(
                self.variant_ids.clone().unwrap(),
                gwas_results.variant_ids,
                "Mismatched variant ids"
            );
        }

        self.beta += &gwas_results.beta_update;
        self.gpv += &gwas_results.gpv_update;
        self.n_features_seen += 1;
    }

    pub fn compute_final_stats(&mut self) -> IGwasResults {
        if self.n_features_seen != self.n_features {
            panic!(
                "Too few features seen. Expected {}, got {}",
                self.n_features, self.n_features_seen
            );
        }

        self.gpv /= self.n_features_seen as f32;
        let dof = self.sample_sizes.map(|x| x - 2 - self.n_covar as i32);
        let ppv = (self.proj.transpose() * &self.cov * &self.proj).diagonal();
        let mut se = DMatrix::zeros(self.gpv.nrows(), ppv.nrows());
        se.par_column_iter_mut()
            .enumerate()
            .for_each(|(j, mut col)| {
                for i in 0..col.len() {
                    col[i] =
                        ((ppv[j] / self.gpv[i] - self.beta[(i, j)].powi(2)) / dof[i] as f32).sqrt();
                }
            });
        let t_stat = self.beta.component_div(&se);
        let mut p_values = DMatrix::zeros(t_stat.nrows(), t_stat.ncols());
        p_values
            .par_column_iter_mut()
            .enumerate()
            .for_each(|(j, mut col)| {
                for i in 0..col.len() {
                    col[i] = compute_neg_log_pvalue(t_stat[(i, j)], dof[i]);
                }
            });

        let n_elements = self.beta.nrows() * self.beta.ncols();

        let existing_variant_ids = self.variant_ids.clone().unwrap();
        let mut variant_ids = Vec::with_capacity(self.n_projections * existing_variant_ids.len());
        variant_ids.extend(
            std::iter::repeat(existing_variant_ids)
                .take(self.n_projections)
                .flatten(),
        );

        let sample_sizes: DVector<i32> = DVector::from_vec(
            self.sample_sizes
                .as_slice()
                .iter()
                .cycle()
                .take(n_elements)
                .cloned()
                .collect(),
        );

        let projection_ids: Vec<String> = self
            .projection_ids
            .iter()
            .flat_map(|x| std::iter::repeat(x.clone()).take(self.chunksize))
            .collect();

        IGwasResults {
            projection_ids,
            variant_ids,
            beta_values: self
                .beta
                .clone()
                .reshape_generic(Dyn(n_elements), Const::<1>),
            se_values: se.reshape_generic(Dyn(n_elements), Const::<1>),
            t_stat_values: t_stat.reshape_generic(Dyn(n_elements), Const::<1>),
            p_values: p_values.reshape_generic(Dyn(n_elements), Const::<1>),
            sample_sizes,
        }
    }
}
