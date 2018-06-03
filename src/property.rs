use rand::{Rng};
use fnv::FnvHashSet;

use driver::{DriverConfiguration};
use node::{Node};

#[derive(Debug, Copy, Clone, PartialEq)]
enum MutatePropertyOperation {
    SwapNodes,
    SwapOutput,
    RandomOutputOne,
    RandomOutputAll,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Property {
    pub nodes: Vec<Node>,
    pub output_indices: Vec<usize>,
}

const PROPERTY_OPERATIONS : [MutatePropertyOperation; 4] = [
    MutatePropertyOperation::SwapNodes,
    MutatePropertyOperation::SwapOutput,
    MutatePropertyOperation::RandomOutputOne,
    MutatePropertyOperation::RandomOutputAll,
];

impl Property {
    pub fn mutate<T: Rng>(&mut self, rng: &mut T, max_connection_index: usize, node_threshold: f64) {
        let value = rng.gen_range::<f64>(0.0, 1.0);

        if value < node_threshold {
            let num_of_nodes = self.nodes.len();
            let node_index = rng.gen_range::<usize>(0, num_of_nodes);
            self.nodes[node_index].mutate_node(rng, max_connection_index);
        } else {
            self.mutate_property(rng, max_connection_index);
        }
    }

    fn mutate_property<T: Rng>(&mut self, rng: &mut T, max_connection_index: usize) {
        let num_of_nodes = self.nodes.len();
        let num_of_outputs = self.output_indices.len();

        use self::MutatePropertyOperation::*;
        match rng.choose(&PROPERTY_OPERATIONS).unwrap() {
            &SwapNodes => {
                let index1 = rng.gen_range::<usize>(0, num_of_nodes);
                let index2 = rng.gen_range::<usize>(0, num_of_nodes);
                self.nodes.swap(index1, index2);
            }
            &SwapOutput => {
                let index1 = rng.gen_range::<usize>(0, num_of_outputs);
                let index2 = rng.gen_range::<usize>(0, num_of_outputs);
                self.output_indices.swap(index1, index2);
            }
            &RandomOutputOne => {
                let index1 = rng.gen_range::<usize>(0, num_of_outputs);
                let index2 = rng.gen_range::<usize>(0, max_connection_index);
                self.output_indices[index1] = index2;
            }
            &RandomOutputAll => {
                for index in &mut self.output_indices {
                    *index = rng.gen_range::<usize>(0, max_connection_index);
                }
            }
        }
    }

    pub fn has_unused_nodes(&self) -> bool {
        let mut set_of_used_nodes = FnvHashSet::default();

        for index in 0..self.output_indices.len() {
            set_of_used_nodes.insert(index);
        }

        for index in 0..self.nodes.len() {
            self.nodes[index].add_used_nodes(index, &mut set_of_used_nodes);
        }

        for index in 0..self.nodes.len() {
            if !set_of_used_nodes.contains(&index) {
                return true
            }
        }

        return false
    }

    pub fn fix<T: Rng>(&mut self, rng: &mut T, configuration: &DriverConfiguration) {
        if self.nodes.is_empty() {
            self.nodes.push(Node::new_simple(rng));
        } else {
            self.nodes.truncate(configuration.max_network_size);
        }

        let num_of_nodes = self.nodes.len();

        let max_connection_index = num_of_nodes + configuration.num_of_input_nodes;

        self.output_indices.resize(configuration.num_of_output_nodes, 0);

        for index in &mut self.output_indices {
            if *index >= max_connection_index {
                *index = rng.gen_range::<usize>(0, max_connection_index);
            }
        }

        for node in &mut self.nodes {
            node.fix(max_connection_index);
        }
    }
}

#[cfg(test)]
mod test {
    // use super::*;
}
