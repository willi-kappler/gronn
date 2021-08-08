
pub struct GNNode {
    in_nodes: Vec<usize>,
    in_weights: Vec<f32>,
    normal_nodes: Vec<usize>,
    normal_weights: Vec<f32>,
    value: f32,
}

impl GNNode {
    pub fn new() -> Self {
        Self {
            in_nodes: Vec::new(),
            in_weights: Vec::new(),
            normal_nodes: Vec::new(),
            normal_weights: Vec::new(),
            value: 0.0,
        }
    }
}