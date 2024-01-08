use std::cmp;
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, ensure, Context, Result};
use crossbeam_channel::Sender;
use log::info;

use crate::io;
use crate::io::gwas::GwasResults;
use crate::stats::running::RunningSufficientStats;

fn gwas_path_to_phenotype(filename: &str) -> String {
    Path::new(filename)
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string()
}

/// Check that a GWAS result file has been provided for every phenotype in the
/// projection and covariance matrices. Filter out all GWAS result files that
/// are not needed.
fn check_filter_inputs(
    projection_labels: &[String],
    covariance_labels: &[String],
    gwas_result_files: &[String],
) -> Result<Vec<String>> {
    ensure!(
        projection_labels == covariance_labels,
        "Projection and covariance matrices have different labels"
    );

    let mut phenotype_to_gwas_path: HashMap<String, String> = HashMap::new();
    for gwas_path in gwas_result_files {
        let phenotype = gwas_path_to_phenotype(gwas_path);
        if phenotype_to_gwas_path.contains_key(&phenotype) {
            bail!("Multiple GWAS files provided for phenotype {}", phenotype);
        }
        phenotype_to_gwas_path.insert(phenotype, gwas_path.to_string());
    }

    let mut final_gwas_paths = Vec::new();
    for phenotype in projection_labels {
        let path = phenotype_to_gwas_path.remove(phenotype).ok_or(anyhow!(
            "No GWAS result file provided for phenotype {}",
            phenotype
        ))?;
        final_gwas_paths.push(path);
    }

    Ok(final_gwas_paths)
}

pub struct RuntimeConfig {
    pub num_threads: usize,
    pub chunksize: usize,
}

fn gwas_reader(
    gwas_result_files: &[String],
    column_names: io::gwas::ColumnSpec,
    start_line: usize,
    end_line: usize,
    num_lines: usize,
    output: Sender<(String, io::gwas::GwasResults)>,
) -> Result<()> {
    for filename in gwas_result_files {
        let phenotype_name = gwas_path_to_phenotype(filename);
        info!(
            "Reading lines {} to {} of {} in {}. Interpreted phenotype name: {}",
            start_line, end_line, num_lines, filename, phenotype_name
        );

        let gwas_results =
            io::gwas::read_gwas_results(filename, &column_names, start_line, end_line)
                .with_context(|| format!("Error reading GWAS results from file: {}", &filename))
                .unwrap();

        output.send((phenotype_name, gwas_results))?;
    }

    Ok(())
}

fn process_chunk(
    gwas_result_files: Vec<String>,
    column_names: io::gwas::ColumnSpec,
    start_line: usize,
    end_line: usize,
    num_lines: usize,
    output_file: &str,
    num_threads: usize,
    running: Arc<Mutex<RunningSufficientStats>>,
) -> Result<()> {
    let (sender, receiver) = crossbeam_channel::unbounded::<(String, GwasResults)>();

    let mut workers = Vec::new();
    for _ in 0..num_threads {
        let receiver = receiver.clone();
        let running = running.clone();
        workers.push(std::thread::spawn(move || {
            for (phenotype_name, gwas_results) in receiver.iter() {
                running
                    .lock()
                    .unwrap()
                    .update(&phenotype_name, &gwas_results);
            }
        }));
    }

    let reader = std::thread::spawn({
        let gwas_result_files = gwas_result_files.clone();
        let column_names = column_names.clone();
        let sender = sender.clone();
        move || {
            gwas_reader(
                &gwas_result_files,
                column_names,
                start_line,
                end_line,
                num_lines,
                sender,
            )
        }
    });

    reader.join().unwrap()?;
    drop(sender);

    for worker in workers {
        worker.join().unwrap();
    }
    info!("Finished reading chunk, computing statistics");

    let final_stats = running.lock().unwrap().compute_final_stats();
    let include_header = start_line == 0;
    io::gwas::write_gwas_results(final_stats, output_file, include_header)
        .with_context(|| format!("Error writing GWAS results to file: {}", output_file))?;

    Ok(())
}

pub fn run(
    projection_matrix_path: &str,
    covariance_matrix_path: &str,
    gwas_result_files: &[String],
    output_file: &str,
    num_covar: usize,
    runtime_config: RuntimeConfig,
    column_names: io::gwas::ColumnSpec,
) -> Result<()> {
    let projection_matrix =
        io::matrix::read_labeled_matrix(projection_matrix_path).with_context(|| {
            format!(
                "Error reading projection matrix: {}",
                projection_matrix_path
            )
        })?;

    let cov_matrix =
        io::matrix::read_labeled_matrix(covariance_matrix_path).with_context(|| {
            format!(
                "Error reading covariance matrix: {}",
                covariance_matrix_path
            )
        })?;

    info!("Projection shape {:?}", projection_matrix.matrix.shape());
    info!("Covariance shape {:?}", cov_matrix.matrix.shape());
    info!("Covariance labels {:?}", cov_matrix.col_labels);
    info!("Projection labels {:?}", projection_matrix.row_labels);

    let gwas_result_files = check_filter_inputs(
        &projection_matrix.row_labels,
        &cov_matrix.col_labels,
        gwas_result_files,
    )?;

    let running = Arc::new(Mutex::new(RunningSufficientStats::new(
        &projection_matrix,
        &cov_matrix,
        num_covar,
        runtime_config.chunksize,
    )));

    let num_lines = io::gwas::count_lines(&gwas_result_files[0])?;
    let mut start_line = 0;
    let mut end_line = 0;
    while start_line < num_lines {
        end_line = cmp::min(num_lines, end_line + runtime_config.chunksize);

        let new_chunksize = end_line - start_line;
        running.lock().unwrap().clear_chunk(new_chunksize);

        process_chunk(
            gwas_result_files.clone(),
            column_names.clone(),
            start_line,
            end_line,
            num_lines,
            output_file,
            runtime_config.num_threads,
            running.clone(),
        )?;

        start_line = end_line;
    }

    let final_stats = running.lock().unwrap().compute_final_stats();
    io::gwas::write_gwas_results(final_stats, output_file, false)
        .with_context(|| format!("Error writing GWAS results to file: {}", output_file))?;

    Ok(())
}
