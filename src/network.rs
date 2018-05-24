use std::f64;

use rand;
use rand::{XorShiftRng, SeedableRng};

use driver::{DriverConfiguration};
use property::{Property};
use node::{Node};

#[derive(Debug, Clone)]
pub struct Network {
    configuration: DriverConfiguration,
    property: Property,
    undo_property: Property,
    nodes_output_values: Vec<f64>,
    rng: XorShiftRng,
    pub best_error: f64,
    pub id: String,
    pub first_place_counter: u64,
}

impl Network {
    pub fn new(configuration: DriverConfiguration) -> Network {
        let mut nodes = Vec::with_capacity(configuration.initial_network_size);
        let mut rng = XorShiftRng::from_seed([
            rand::random::<u32>(),
            rand::random::<u32>(),
            rand::random::<u32>(),
            rand::random::<u32>()
        ]);

        for _ in 0..configuration.initial_network_size {
            nodes.push(Node::new_simple(&mut rng));
        }

        let output_indices = vec![0; configuration.num_of_output_nodes];

        let property = Property {
            nodes,
            output_indices,
        };

        Self::new_with_property(configuration, property, "untrained")
    }

    pub fn new_with_property(configuration: DriverConfiguration, property: Property, id: &str) -> Network {
        let undo_property = Property {
            nodes: Vec::new(),
            output_indices: Vec::new(),
        };

        let nodes_output_values = vec![0.0; configuration.num_of_input_nodes + property.nodes.len()];

        Network {
            configuration,
            property,
            undo_property,
            nodes_output_values,
            rng: XorShiftRng::from_seed([
                rand::random::<u32>(),
                rand::random::<u32>(),
                rand::random::<u32>(),
                rand::random::<u32>()
            ]),
            best_error: f64::MAX,
            id: id.to_string(),
            first_place_counter: 0,
        }
    }

    pub fn set_configuration(&mut self, configuration: DriverConfiguration) {
        self.configuration = configuration;
    }

    fn reset_values(&mut self) {
        for value in &mut self.nodes_output_values {
            *value = 0.0;
        }
    }

    fn calculate_once(&mut self, provided_input: &[f64]) {
        for i in 0..self.configuration.num_of_input_nodes {
            self.nodes_output_values[i] = provided_input[i];
        }

        for i in 0..self.property.nodes.len() {
            self.nodes_output_values[self.configuration.num_of_input_nodes + i] = self.property.nodes[i].calculate(&self.nodes_output_values);
        }
    }

    pub fn calculate(&mut self, provided_input: &[f64]) {
        self.reset_values();
        for _ in 0..self.configuration.num_of_cycles {
            self.calculate_once(&provided_input);
        }
    }

    pub fn calculate_error(&mut self, expected_output: &[f64]) -> f64 {
        self.property.output_indices.iter().zip(expected_output).fold(0.0, |error, (index, expected_value)| {
            error + (expected_value - self.nodes_output_values[*index]).abs()
        })
    }

    fn calculate_batch_and_error(&mut self, provided_input: &[Vec<f64>], expected_output: &[Vec<f64>]) -> f64 {
        provided_input.iter().zip(expected_output).fold(0.0, |error, (input, output)| {
            self.calculate(input);
            error + self.calculate_error(output)
        })
    }

    pub fn get_output(&self) -> Vec<f64> {
        self.property.output_indices.iter().map(|index| self.nodes_output_values[*index]).collect()
    }

    pub fn maybe_add_node(&mut self) {
        if self.property.nodes.len() >= self.configuration.max_network_size {
            return
        }

        if self.property.has_unused_nodes() {
            // There are still unused noded in the network, so no need to add more!
            return
        }

        self.property.nodes.push(Node::new_simple(&mut self.rng));
        self.nodes_output_values.push(0.0);
    }

    fn mutate(&mut self) {
        self.property.mutate(&mut self.rng, self.nodes_output_values.len(), self.configuration.node_threshold);
    }

    pub fn optimize_batch(&mut self, provided_input: &[Vec<f64>], expected_output: &[Vec<f64>]) {
        // Initialize
        self.undo_property = self.property.clone();

        for _ in 0..self.configuration.num_of_node_mutation {
            self.mutate();

            let batch_error = self.calculate_batch_and_error(provided_input, expected_output);

            if batch_error < self.best_error {
                // Better solution found
                self.best_error = batch_error;
                self.undo_property = self.property.clone();
            }
        }

        // Revert to previous best solution
        self.property = self.undo_property.clone();
    }

    pub fn set_property(&mut self, property: Property) {
        self.property = property;
    }

    pub fn get_property(&self) -> Property {
        self.property.clone()
    }

    pub fn fix(&mut self) {
        self.property.fix(&mut self.rng, &self.configuration);

        let num_of_nodes = self.property.nodes.len();
        let max_connection_index = num_of_nodes + self.configuration.num_of_input_nodes;

        self.nodes_output_values.resize(max_connection_index, 0.0);
    }

    pub fn reset_best_error(&mut self, provided_input: &[Vec<f64>], expected_output: &[Vec<f64>]) {
        self.best_error = self.calculate_batch_and_error(provided_input, expected_output);
    }

    pub fn is_good_enough(&mut self) -> bool {
        self.best_error <= self.configuration.desired_error
    }

    pub fn num_of_nodes(&self) -> usize {
        self.property.nodes.len()
    }

    pub fn reseed(&mut self, seed: [u32; 4]) {
        self.rng.reseed(seed);
    }

    pub fn move_nodes(&mut self, property: &Property, provided_input: &[Vec<f64>], expected_output: &[Vec<f64>], direction: f64) {
        let num_of_nodes = self.property.nodes.len();
        if num_of_nodes == property.nodes.len() {
            self.undo_property = self.property.clone();

            for i in 0..num_of_nodes {
                let node = property.nodes[i].clone();
                if !self.property.nodes[i].move_node_if_equal(node, direction) {
                    return
                }
            }

            let batch_error = self.calculate_batch_and_error(provided_input, expected_output);

            if batch_error < self.best_error {
                // Better solution found
                self.best_error = batch_error;
            } else {
                self.property = self.undo_property.clone();
            }
        }
    }
}

#[cfg(test)]
mod test {
    // use super::*;
}
