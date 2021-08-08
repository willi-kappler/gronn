
use crate::gn_node::GNNode;
pub struct GNNetwork {
    input: Vec<f32>,
    normal_nodes: Vec<GNNode>,
    output_node: Vec<GNNode>,
}

impl GNNetwork {
    pub fn new() -> Self {
        Self {
            input: Vec::new(),
            normal_nodes: Vec::new(),
            output_node: Vec::new(),
        }
    }
}
