use std::fs::File;
use std::io::{Write, Read, BufWriter, BufReader};
use std::f64;
use std::time::Instant;

use serde_json;
use rand;
use rand::{Rng};
use failure::Error;
use rayon::prelude::*;

use network::{Network};
use network_configurations;

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DriverConfiguration {
    #[serde(default="DriverConfiguration::num_of_input_nodes")]
    pub num_of_input_nodes: usize,
    #[serde(default="DriverConfiguration::num_of_output_nodes")]
    pub num_of_output_nodes: usize,
    #[serde(default="DriverConfiguration::initial_network_size")]
    pub initial_network_size: usize,
    #[serde(default="DriverConfiguration::max_network_size")]
    pub max_network_size: usize,
    #[serde(default="DriverConfiguration::num_of_networks")]
    pub num_of_networks: usize,
    #[serde(default="DriverConfiguration::num_of_node_mutation")]
    pub num_of_node_mutation: usize,
    #[serde(default="DriverConfiguration::num_of_iterations")]
    pub num_of_iterations: usize,
    #[serde(default="DriverConfiguration::num_of_batch_iterations")]
    pub num_of_batch_iterations: usize,
    #[serde(default="DriverConfiguration::batch_size")]
    pub batch_size: usize,
    #[serde(default="DriverConfiguration::num_of_cycles")]
    pub num_of_cycles: usize,
    #[serde(default="DriverConfiguration::use_trained_networks")]
    pub use_trained_networks: bool,
    #[serde(default="DriverConfiguration::node_threshold")]
    pub node_threshold: f64,
    #[serde(default="DriverConfiguration::desired_error")]
    pub desired_error: f64,
    #[serde(default="DriverConfiguration::num_of_threads")]
    pub num_of_threads: usize,
}

impl DriverConfiguration {
    fn num_of_input_nodes() -> usize {5}
    fn num_of_output_nodes() -> usize {5}
    fn initial_network_size() -> usize {1}
    fn max_network_size() -> usize {100}
    fn num_of_networks() -> usize {20}
    fn num_of_node_mutation() -> usize {100}
    fn num_of_iterations() -> usize {1000}
    fn num_of_batch_iterations() -> usize {100}
    fn batch_size() -> usize {10}
    fn num_of_cycles() -> usize {2}
    fn use_trained_networks() -> bool {true}
    fn node_threshold() -> f64 {0.1}
    fn desired_error() -> f64 {0.01}
    fn num_of_threads() -> usize {1}
}

#[derive(Debug, Clone)]
pub struct Driver {
    configuration: DriverConfiguration,
    networks: Vec<Network>,
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct TrainingData {
    pub provided_input: Vec<Vec<f64>>,
    pub expected_output: Vec<Vec<f64>>,
}

impl Driver {
    pub fn new_from_config(configuration: DriverConfiguration) -> Driver {
        assert!(configuration.num_of_input_nodes > 0);
        assert!(configuration.num_of_output_nodes > 0);
        assert!(configuration.initial_network_size > 0);
        assert!(configuration.max_network_size > configuration.initial_network_size);
        assert!(configuration.num_of_networks > 1);
        assert!(configuration.num_of_node_mutation > 0);
        assert!(configuration.batch_size > 0);
        assert!(configuration.num_of_cycles > 0);
        assert!(configuration.node_threshold > 0.0 && configuration.node_threshold < 1.0);

        let mut networks = Vec::with_capacity(configuration.num_of_networks);

        if configuration.use_trained_networks {
            networks.push(network_configurations::xor01(configuration.clone()));
            networks.push(network_configurations::xor02(configuration.clone()));
            networks.push(network_configurations::xor03(configuration.clone()));
            networks.push(network_configurations::xor04(configuration.clone()));
            networks.push(network_configurations::iris01(configuration.clone()));
            networks.push(network_configurations::iris02(configuration.clone()));
            networks.push(network_configurations::iris03(configuration.clone()));
            networks.push(network_configurations::iris04(configuration.clone()));
            networks.push(network_configurations::adder01(configuration.clone()));
            networks.push(network_configurations::adder02(configuration.clone()));
        } else {
            for _ in 0..configuration.num_of_networks {
                networks.push(Network::new(configuration.clone()));
            }
        }

        Driver {
            configuration,
            networks,
        }
    }

    pub fn new_from_json(data: &str) -> Result<Driver, Error> {
        let configuration: DriverConfiguration = serde_json::from_str(&data)?;

        Ok(Self::new_from_config(configuration))
    }

    pub fn new_from_file(filename: &str) -> Result<Driver, Error> {
        let mut data = String::new();
        let f = File::open(filename)?;
        let mut f = BufReader::new(f);
        f.read_to_string(&mut data)?;

        Self::new_from_json(&data)
    }

    pub fn train_from_file(&mut self, filename: &str) -> Result<(), Error> {
        info!("Load training data from file: {}", filename);

        let mut data = String::new();
        let f = File::open(filename)?;
        let mut f = BufReader::new(f);
        f.read_to_string(&mut data)?;

        let training_data: TrainingData = serde_json::from_str(&data)?;

        info!("File loaded successfully");

        self.train(&training_data);

        Ok(())
    }

    pub fn train(&mut self, training_data: &TrainingData) {
        let start_time = Instant::now();
        info!("Begin training");

        let input_len = training_data.provided_input.len();
        let output_len = training_data.expected_output.len();

        assert_eq!(input_len, output_len, "Error in training data: input and output length do not match: {} != {}", input_len, output_len);
        assert!(self.configuration.batch_size <= input_len, "Error in training data: batch size must be <= {} (input length), given: {}", input_len, self.configuration.batch_size);

        info!("Number of entries: {}", input_len);

        let mut input_batch: Vec<Vec<f64>>;
        let mut output_batch: Vec<Vec<f64>>;

        let change_batch =  if self.configuration.batch_size == input_len {
            input_batch = training_data.provided_input.clone();
            output_batch = training_data.expected_output.clone();
            false
        } else {
            input_batch = Vec::with_capacity(self.configuration.batch_size);
            output_batch = Vec::with_capacity(self.configuration.batch_size);
            for _ in 0..self.configuration.batch_size {
                input_batch.push(Vec::new());
                output_batch.push(Vec::new());
            }
            true
        };

        let mut rng = rand::thread_rng();

        let num_of_iterations = self.configuration.num_of_iterations;

        for i in 0..self.configuration.num_of_batch_iterations {
            if change_batch {
                for j in 0..self.configuration.batch_size {
                    let index = rng.gen_range::<usize>(0, input_len);
                    // TODO: store index in batch, and provide training_data to optimize_batch
                    // let indices: Vec<usize> = (0..input_len).collect();
                    // rng.shuffle(indices);
                    // batch[i] = index
                    input_batch[j] = training_data.provided_input[index].clone();
                    output_batch[j] = training_data.expected_output[index].clone();
                }
            }

            // info!("input_batch: {:?}", input_batch);
            // info!("output_batch: {:?}", output_batch);

            for network in &mut self.networks {
                network.reseed([
                    rand::random::<u32>(),
                    rand::random::<u32>(),
                    rand::random::<u32>(),
                    rand::random::<u32>()
                ]);
            }

            self.networks.par_iter_mut().for_each(|network| {
                // Reset best error for this batch
                network.reset_best_error(&input_batch, &output_batch);
                for j in 0..num_of_iterations {
                    network.optimize_batch(&input_batch, &output_batch);

                    if network.is_good_enough() {
                        // No more training needed for this network
                        info!("Good enough after {} iterations", j);
                        break;
                    }
                }
            });

            self.networks.sort_unstable_by(|n1, n2| n1.best_error.partial_cmp(&n2.best_error).unwrap());
            self.networks.truncate(self.configuration.num_of_networks); // Get rid of worst solutions
            // Give the last network a chance to improve:
            let last_network_index = self.networks.len() - 1;
            self.networks[last_network_index].maybe_add_node();

            // Try to avoid cloning local optimum over and over again
            if self.networks[0].best_error != self.networks[1].best_error {
                // Clone the best solution:
                let mut new_network = self.networks[0].clone();
                new_network.first_place_counter = 0;
                // And add it to the list of networks:
                self.networks.push(new_network);
            }

            self.networks[0].first_place_counter += 1;

            /*
            if num_of_networks > 5 {
                for j in 0..(num_of_networks - 5) {
                    self.networks[j].set_node_mutation(self.configuration.num_of_node_mutation);
                }
                self.networks[num_of_networks - 1].set_node_mutation(1);
                self.networks[num_of_networks - 2].set_node_mutation(5);
                self.networks[num_of_networks - 3].set_node_mutation(10);
                self.networks[num_of_networks - 4].set_node_mutation(20);
                self.networks[num_of_networks - 5].set_node_mutation(50);
            }
            */

            /*
            if num_of_networks > 5 {
                self.networks[0].set_node_mutation(1);
                self.networks[1].set_node_mutation(5);
                self.networks[2].set_node_mutation(10);
                self.networks[3].set_node_mutation(20);
                self.networks[4].set_node_mutation(50);
                for j in 5..num_of_networks {
                    self.networks[j].set_node_mutation(self.configuration.num_of_node_mutation);
                }
            }
            */

            info!("Batch iteration: {} of {}", i, self.configuration.num_of_batch_iterations);
            for network in &self.networks {
                info!("Best error: {}, num. of nodes: {}, id: {}, first place: {}", network.best_error, network.num_of_nodes(), network.id, network.first_place_counter);
            }
            info!("-------------------------------------------");
        }

        let duration = start_time.elapsed();
        let duration = (duration.as_secs() as f64) + ((duration.subsec_nanos() as f64) * 1e-9);
        info!("End training");
        info!("Time taken: {} seconds", duration);
        info!("Best error: {}, desired error: {}", self.networks[0].best_error, self.configuration.desired_error);
    }

    pub fn test(&mut self, provided_input: &[f64], expected_output: &[f64]) -> (f64, Vec<f64>) {
        self.networks[0].calculate(provided_input);
        let error = self.networks[0].calculate_error(expected_output);

        let output_values = self.networks[0].get_output();

        (error, output_values)
    }

    pub fn predict(&mut self, provided_input: &[f64]) -> Vec<f64> {
        self.networks[0].calculate(provided_input);
        self.networks[0].get_output()
    }

    pub fn set_network(&mut self, mut network: Network) {
        network.set_configuration(self.configuration.clone());
        network.fix();
        self.networks.push(network);
    }

    pub fn load_network(&mut self, filename: &str) -> Result<(), Error>  {
        let mut data = String::new();
        let f = File::open(filename)?;
        let mut f = BufReader::new(f);
        f.read_to_string(&mut data)?;

        let mut new_network = Network::new(self.configuration.clone());
        new_network.set_property(serde_json::from_str(&data)?);
        new_network.fix();

        self.networks.push(new_network);

        Ok(())
    }

    pub fn save_network(&self, filename: &str) -> Result<(), Error>  {
        let serialized = serde_json::to_string(&self.networks[0].get_property())?;

        let f = File::create(filename)?;
        let mut f = BufWriter::new(f);
        f.write_all(serialized.as_bytes())?;

        Ok(())
    }
}

#[cfg(test)]
mod test {
    // use super::*;
}
