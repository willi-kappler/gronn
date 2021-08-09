
use nanorand::{WyRand, Rng};


// TODO: Add special output node

pub struct GNOutputNode {
    normal_nodes: Vec<usize>,
    normal_weights: Vec<f32>,
}

impl GNOutputNode {
    pub(crate) fn new() -> Self {
        Self {
            normal_nodes: Vec::new(),
            normal_weights: Vec::new(),
        }
    }
    pub(crate) fn calculate_value(&self, normal_values: &[f32]) -> f32 {
        let mut result = 0.0;

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
    fn add_normal_connection(&mut self, index: usize, weight: f32) -> bool {
        if !self.normal_nodes.contains(&index) {
            self.normal_nodes.push(index);
            self.normal_weights.push(weight);
            true
        } else {
            false
        }
    }
    fn remove_normal_connection(&mut self, index: usize) -> bool {
        if self.normal_nodes.len() > 1 {
            self.normal_nodes.swap_remove(index);
            self.normal_weights.swap_remove(index);
            true
        } else {
            false
        }
    }
    fn replace_normal_weight(&mut self, index: usize, weight: f32) {
        self.normal_weights[index] = weight;
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
    fn gen_normal_index(&self, rng: &mut WyRand) -> usize {
        rng.generate_range(0_usize..self.normal_nodes.len())
    }
    pub(crate) fn mutate(&mut self, nodes_len: usize, rng: &mut WyRand) {
        loop {
            let operation = rng.generate_range(0_u8..4);

            match operation {
                0 => {
                    let index = rng.generate_range(0_usize..nodes_len);
                    let weight = self.gen_weight(rng);
                    if self.add_normal_connection(index, weight) {
                        break;
                    }
                }
                1 => {
                    let index = self.gen_normal_index(rng);
                    if self.remove_normal_connection(index) {
                        break;
                    }
                }
                2 => {
                    let index = self.gen_normal_index(rng);
                    let weight = self.gen_weight(rng);
                    self.replace_normal_weight(index, weight);
                    break;
                }
                3 => {
                    let index = self.gen_normal_index(rng);
                    let amount = self.gen_amount(rng);
                    self.change_normal_weight(index, amount);
                    break;
                }
                _ => {
                    panic!("Unknown operation in GNOutputNode::mutate: '{}'", operation);
                }
            }
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
}
