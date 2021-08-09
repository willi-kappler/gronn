
use crate::gn_node::GNNode;
use crate::gn_output_node::GNOutputNode;

use nanorand::{WyRand, Rng};

pub struct GNNetwork {
    input_len: usize,
    output_len: usize,
    normal_nodes: Vec<GNNode>,
    normal_values: Vec<f32>,
    output_nodes: Vec<GNOutputNode>,
    min_network_size: usize,
    max_network_size: usize,
    add_node_probability: f32,
    remove_node_probability: f32,
    calculation_steps: u8,
}

impl GNNetwork {
    pub(crate) fn new(input_len: usize, output_len: usize) -> Self {
        let mut output_nodes = Vec::new();

        for _ in 0..output_len {
            output_nodes.push(GNOutputNode::new());
        }

        let mut normal_nodes = Vec::new();
        let mut normal_values = Vec::new();

        normal_nodes.push(GNNode::new());
        normal_values.push(0.0);
        normal_nodes.push(GNNode::new());
        normal_values.push(0.0);

        Self {
            input_len,
            output_len,
            normal_nodes,
            normal_values,
            output_nodes,
            min_network_size: 2,
            max_network_size: 20,
            add_node_probability: 0.1,
            remove_node_probability: 0.1,
            calculation_steps: 2,
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
            self.normal_values.push(0.0);
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
    pub(crate) fn mutate(&mut self) {
        let mut rng = WyRand::new();

        let operation = rng.generate_range(0_u8..3);

        // TODO: use early return and move mutate_node to bottom

        match operation {
            0 => {
                let dice = self.roll_dice(&mut rng);

                if dice < self.add_node_probability {
                    if self.add_node() {
                        return
                    }
                }
            }
            1 => {
                let dice = self.roll_dice(&mut rng);

                if dice < self.remove_node_probability {
                    if self.remove_node(&mut rng) {
                        return
                    }
                }
            }
            2 => {
                // Mutate node, see below
            }
            _ => {
                panic!("Unknown operation in GNNetwork::mutate: '{}'", operation);
            }
        }

        self.mutate_node(&mut rng);
    }
    fn calculate_once(&mut self, input_values: &[f32]) {
        for i in 0..self.normal_nodes.len() {
            let result = self.normal_nodes[i].calculate_value(input_values, &self.normal_values);
            self.normal_values[i] = result;
        }
    }
    pub(crate) fn calculate(&mut self, input_values: &[f32]) {
        for _ in 0..self.calculation_steps {
            self.calculate_once(input_values);
        }
    }
    pub(crate) fn error_single(&self, expected_values: &[f32]) -> f32 {
        let mut error = 0.0;

        for i in 0..self.output_len {
            let value = self.output_nodes[i].calculate_value(&self.normal_values);
            error += (value - expected_values[i]).abs();
        }

        error
    }
}
