
use crate::gn_node::GNNode;
use crate::gn_output_node::GNOutputNode;

use nanorand::{WyRand, Rng};
use darwin_rs::{DWIndividual};

pub trait GNDataProvider {
    fn get_input_output(&self) -> Option<(&[f32], &[f32])>;
}

pub struct GNNetwork<T> {
    input_len: usize,
    output_len: usize,
    normal_nodes: Vec<GNNode>,
    output_nodes: Vec<GNOutputNode>,
    min_network_size: usize,
    max_network_size: usize,
    add_node_probability: f32,
    remove_node_probability: f32,
    calculation_steps: u8,
    data_provider: T,
}

impl<T: GNDataProvider> GNNetwork<T> {
    pub(crate) fn new(input_len: usize, output_len: usize, data_provider: T) -> Self {
        let mut output_nodes = Vec::new();

        for _ in 0..output_len {
            output_nodes.push(GNOutputNode::new());
        }

        let mut normal_nodes = Vec::new();

        normal_nodes.push(GNNode::new());
        normal_nodes.push(GNNode::new());

        Self {
            input_len,
            output_len,
            normal_nodes,
            output_nodes,
            min_network_size: 2,
            max_network_size: 20,
            add_node_probability: 0.1,
            remove_node_probability: 0.1,
            calculation_steps: 2,
            data_provider,
        }
    }
    fn add_node(&mut self) -> bool {
        for node in self.normal_nodes.iter_mut() {
            if node.is_unused() {
                node.set_unused(false);
                return true;
            }
        }

        if self.normal_nodes.len() < self.max_network_size {
            self.normal_nodes.push(GNNode::new());
            true
        } else {
            false
        }
    }
    fn remove_node(&mut self, rng: &mut WyRand) -> bool {
        let mut nodes_len = 0;

        for node in self.normal_nodes.iter() {
            if !node.is_unused() {
                nodes_len += 1;
            }
        }

        if nodes_len > self.min_network_size {
            let mut index = rng.generate_range(0_usize..nodes_len);
            while self.normal_nodes[index].is_unused() {
                index = rng.generate_range(0_usize..nodes_len);
            }

            self.normal_nodes[index].set_unused(true);

            for node in self.normal_nodes.iter_mut() {
                node.remove_connection_with_index(index);
            }

            true
        } else {
            false
        }
    }
    fn mutate_normal_node(&mut self, rng: &mut WyRand) {
        let nodes_len = self.normal_nodes.len();
        let mut index = rng.generate_range(0_usize..nodes_len);
        while self.normal_nodes[index].is_unused() {
            index = rng.generate_range(0_usize..nodes_len);
        }

        self.normal_nodes[index].mutate(self.input_len, nodes_len, rng);
    }
    fn mutate_output_node(&mut self, rng: &mut WyRand) {
        let nodes_len = self.output_nodes.len();
        let index = rng.generate_range(0_usize..nodes_len);
        self.output_nodes[index].mutate(nodes_len, rng);
    }
    fn mutate_node(&mut self, rng: &mut WyRand) {
        let normal_node = rng.generate::<bool>();

        if normal_node {
            self.mutate_normal_node(rng);
        } else {
            self.mutate_output_node(rng);
        }
    }
    fn roll_dice(&self, rng: &mut WyRand) -> f32 {
        (rng.generate::<u16>() as f32) / (u16::MAX as f32)
    }
    fn mutate(&mut self) {
        let mut rng = WyRand::new();

        loop {
            let operation = rng.generate_range(0_u8..3);

            match operation {
                0 => {
                    let dice = self.roll_dice(&mut rng);
    
                    if dice < self.add_node_probability {
                        if self.add_node() {
                            break;
                        }
                    }
                }
                1 => {
                    let dice = self.roll_dice(&mut rng);
    
                    if dice < self.remove_node_probability {
                        if self.remove_node(&mut rng) {
                            break;
                        }
                    }
                }
                2 => {
                    self.mutate_node(&mut rng);
                    break;
                }
                _ => {
                    panic!("Unknown operation in GNNetwork::mutate: '{}'", operation);
                }
            }
        }

    }
    fn calculate_single(&self, input_values: &[f32], node_values: &mut [f32]) {
        for i in 0..self.normal_nodes.len() {
            let result = self.normal_nodes[i].calculate_value(input_values, node_values);
            node_values[i] = result;
        }
    }
    fn calculate(&self, input_values: &[f32], node_values: &mut [f32]) {
        for _ in 0..self.calculation_steps {
            self.calculate_single(input_values, node_values);
        }
    }
    fn error(&self, expected_values: &[f32], node_values: &[f32]) -> f32 {
        let mut error = 0.0;

        for i in 0..self.output_len {
            let value = self.output_nodes[i].calculate_value(node_values);
            error += (value - expected_values[i]).abs();
        }

        error
    }
}

impl<T: GNDataProvider> DWIndividual for GNNetwork<T> {
    fn mutate(&mut self) {
        self.mutate()
    }

    fn calculate_fitness(&self) -> f64 {
        let mut error = 0.0;
        let mut node_values= vec![0.0; self.normal_nodes.len()];

        while let Some((input_values, expected_values)) = self.data_provider.get_input_output() {
            self.calculate(input_values, &mut node_values);
            error += self.error(expected_values, &node_values);
        }

        error as f64
    }

    fn get_additional_fitness(&self) -> f64 {
        self.normal_nodes.len() as f64
    }
}
