
use nanorand::{WyRand, Rng};


// TODO: Add special output node

pub struct GNNode {
    input_nodes: Vec<usize>,
    input_weights: Vec<f32>,
    normal_nodes: Vec<usize>,
    normal_weights: Vec<f32>,
    unused: bool,
}

impl GNNode {
    pub(crate) fn new() -> Self {
        Self {
            input_nodes: Vec::new(),
            input_weights: Vec::new(),
            normal_nodes: Vec::new(),
            normal_weights: Vec::new(),
            unused: false,
        }
    }
    pub(crate) fn calculate_value(&self, input_values: &[f32], normal_values: &[f32]) -> f32 {
        if self.unused {
            0.0
        } else {
            let mut result = 0.0;

            for (input_index, input_weight) in self.input_nodes.iter().zip(&self.input_weights) {
                result += input_values[*input_index] * input_weight;
            }
    
            for (normal_index, normal_weight) in self.normal_nodes.iter().zip(&self.normal_weights) {
                result += normal_values[*normal_index] * normal_weight;
            }
    
            // Leaky ReLU
            if result < 0.0 {
                result * 0.01
            } else {
                result
            }    
        }
    }
    fn add_input_connection(&mut self, index: usize, weight: f32) {
        if !self.input_nodes.contains(&index) {
            self.input_nodes.push(index);
            self.input_weights.push(weight);    
        }
    }
    fn add_normal_connection(&mut self, index: usize, weight: f32) {
        if !self.normal_nodes.contains(&index) {
            self.normal_nodes.push(index);
            self.normal_weights.push(weight);    
        }
    }
    fn remove_input_connection(&mut self, index: usize) {
        self.input_nodes.swap_remove(index);
        self.input_weights.swap_remove(index);
    }
    fn remove_normal_connection(&mut self, index: usize) {
        self.normal_nodes.swap_remove(index);
        self.normal_weights.swap_remove(index);
    }
    fn replace_input_weight(&mut self, index: usize, weight: f32) {
        self.input_weights[index] = weight;
    }
    fn replace_normal_weight(&mut self, index: usize, weight: f32) {
        self.normal_weights[index] = weight;
    }
    fn change_input_weight(&mut self, index: usize, amount: f32) {
        self.input_weights[index] += amount;
    }
    fn change_normal_weight(&mut self, index: usize, amount: f32) {
        self.normal_weights[index] += amount;
    }
    fn gen_weight(&self, rng: &mut WyRand) -> f32 {
        ((rng.generate_range(0_u16..2000) as f32) - 1000.0) / 100.0
    }
    fn gen_amount(&self, rng: &mut WyRand) -> f32 {
        ((rng.generate_range(0_u16..2000) as f32) - 1000.0) / 10000.0
    }
    fn gen_input_index(&self, rng: &mut WyRand) -> usize {
        rng.generate_range(0_usize..self.input_nodes.len())
    }
    fn gen_normal_index(&self, rng: &mut WyRand) -> usize {
        rng.generate_range(0_usize..self.normal_nodes.len())
    }
    pub(crate) fn mutate(&mut self, input_len: usize, nodes_len: usize, rng: &mut WyRand) {
        if !self.unused {
            let operation = rng.generate_range(0_u8..8);

            match operation {
                0 => {
                    let index = rng.generate_range(0_usize..input_len);
                    let weight = self.gen_weight(rng);
                    self.add_input_connection(index, weight);
                }
                1 => {
                    let index = rng.generate_range(0_usize..nodes_len);
                    let weight = self.gen_weight(rng);
                    self.add_normal_connection(index, weight);
                }
                2 => {
                    let index = self.gen_input_index(rng);
                    self.remove_input_connection(index);
                }
                3 => {
                    let index = self.gen_normal_index(rng);
                    self.remove_normal_connection(index);
                }
                4 => {
                    let index = self.gen_input_index(rng);
                    let weight = self.gen_weight(rng);
                    self.replace_input_weight(index, weight);
                }
                5 => {
                    let index = self.gen_normal_index(rng);
                    let weight = self.gen_weight(rng);
                    self.replace_normal_weight(index, weight);
                }
                6 => {
                    let index = self.gen_input_index(rng);
                    let amount = self.gen_amount(rng);
                    self.change_input_weight(index, amount);
                }
                7 => {
                    let index = self.gen_normal_index(rng);
                    let amount = self.gen_amount(rng);
                    self.change_normal_weight(index, amount);
                }
                _ => {
                    panic!("Unknown operation in GNNode::mutate: '{}'", operation);
                }
            }
        }
    }
    pub(crate) fn mutate_as_output(&mut self, nodes_len: usize, rng: &mut WyRand) {
        let operation = rng.generate_range(0_u8..4);

        match operation {
            0 => {
                let index = rng.generate_range(0_usize..nodes_len);
                let weight = self.gen_weight(rng);
                self.add_normal_connection(index, weight);
            }
            1 => {
                let index = self.gen_normal_index(rng);
                self.remove_normal_connection(index);
            }
            2 => {
                let index = self.gen_normal_index(rng);
                let weight = self.gen_weight(rng);
                self.replace_normal_weight(index, weight);
            }
            3 => {
                let index = self.gen_normal_index(rng);
                let amount = self.gen_amount(rng);
                self.change_normal_weight(index, amount);
            }
            _ => {
                panic!("Unknown operation in GNNode::mutate: '{}'", operation);
            }
        }
    }
    pub(crate) fn set_unused(&mut self, unused: bool) {
        self.unused = unused;

        if unused {
            self.input_nodes.clear();
            self.input_weights.clear();
            self.normal_nodes.clear();
            self.normal_weights.clear();
        }
    }
    pub(crate) fn remove_connection_with_index(&mut self, index: usize) {
        for i in 0..self.normal_nodes.len() {
            if self.normal_nodes[i] == index {
                self.remove_normal_connection(i);
                break;
            }
        }
    }
    pub(crate) fn is_unused(&self) -> bool {
        self.unused
    }
}
