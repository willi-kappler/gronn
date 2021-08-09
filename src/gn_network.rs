
use crate::gn_node::GNNode;

use std::rc::Rc;

pub struct GNNetwork {
    pub(crate) input_values: Rc<Vec<f32>>,
    pub(crate) normal_nodes: Vec<GNNode>,
    pub(crate) normal_values: Vec<f32>,
    pub(crate) output_node: Vec<GNNode>,
    pub(crate) output_values: Vec<f32>,
    pub(crate) expected_output: Rc<Vec<f32>>,
}

impl GNNetwork {
    pub fn new(input_values: &Rc<Vec<f32>>, expected_output: &Rc<Vec<f32>>) -> Self {
        Self {
            input_values: input_values.clone(),
            normal_nodes: Vec::new(),
            normal_values: Vec::new(),
            output_node: Vec::new(),
            output_values: Vec::new(),
            expected_output: expected_output.clone(),
        }
    }
}
