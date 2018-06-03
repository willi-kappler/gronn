use std::f64;

use rand::{Rng};
use fnv::FnvHashSet;

const LIMIT1 : f64 = 1.0;
const LIMIT2 : f64 = 0.001;
const LIMIT3 : f64 = 0.00001;
const LIMIT4 : f64 = 0.0000001;

#[derive(Debug, Copy, Clone, PartialEq)]
enum MutateNodeOperation {
    SwapConnections,
    AddConnection,
    RemoveConnection,
    RandomConnectionOne,
    RandomConnectionAll,
    DeltaBias1,
    DeltaBias2,
    DeltaBias3,
    DeltaBias4,
    RandomBias,
    DeltaWeightOne1,
    DeltaWeightOne2,
    DeltaWeightOne3,
    DeltaWeightOne4,
    RandomWeightOne,
    DeltaWeightAll1,
    DeltaWeightAll2,
    DeltaWeightAll3,
    DeltaWeightAll4,
    RandomWeightAll,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct Connection {
    index: usize,
    weight: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Node {
    bias: f64,
    connections: Vec<Connection>,
}

const NODE_OPERATIONS : [MutateNodeOperation; 20] = [
    MutateNodeOperation::SwapConnections,
    MutateNodeOperation::AddConnection,
    MutateNodeOperation::RemoveConnection,
    MutateNodeOperation::RandomConnectionOne,
    MutateNodeOperation::RandomConnectionAll,
    MutateNodeOperation::DeltaBias1,
    MutateNodeOperation::DeltaBias2,
    MutateNodeOperation::DeltaBias3,
    MutateNodeOperation::DeltaBias4,
    MutateNodeOperation::RandomBias,
    MutateNodeOperation::DeltaWeightOne1,
    MutateNodeOperation::DeltaWeightOne2,
    MutateNodeOperation::DeltaWeightOne3,
    MutateNodeOperation::DeltaWeightOne4,
    MutateNodeOperation::RandomWeightOne,
    MutateNodeOperation::DeltaWeightAll1,
    MutateNodeOperation::DeltaWeightAll2,
    MutateNodeOperation::DeltaWeightAll3,
    MutateNodeOperation::DeltaWeightAll4,
    MutateNodeOperation::RandomWeightAll,
];

impl Node {
    pub fn new_simple<T: Rng>(rng: &mut T) -> Node {
        Node {
            bias: rng.gen_range::<f64>(-10.0, 10.0),
            connections: vec![
                Connection {
                    index: 0,
                    weight: rng.gen_range::<f64>(-10.0, 10.0),
                }
            ],
        }
    }

    pub fn calculate(&self, node_values: &[f64]) -> f64 {
        let value = self.connections.iter().fold(self.bias, |sum, connection| {
            sum + (connection.weight * node_values[connection.index])
        });

        // Leaky ReLU
        if value < 0.0 {
            value * 0.01
        } else {
            value
        }
    }

    pub fn mutate_node<T: Rng>(&mut self, rng: &mut T, max_connection_index: usize) {
        let num_of_connections = self.connections.len();

        use self::MutateNodeOperation::*;
        match rng.choose(&NODE_OPERATIONS).unwrap() {
            &SwapConnections => {
                if num_of_connections > 1 {
                    let index1 = rng.gen_range::<usize>(0, num_of_connections);
                    let mut index2 = rng.gen_range::<usize>(0, num_of_connections);

                    // Ensure we really have two different connections
                    while index1 == index2 {
                        index2 = rng.gen_range::<usize>(0, num_of_connections);
                    }

                    let con_index1 = self.connections[index1].index;
                    let con_index2 = self.connections[index2].index;

                    self.connections[index1].index = con_index2;
                    self.connections[index2].index = con_index1;
                } else {
                    // No swap possible, try a different mutation
                    self.mutate_node(rng, max_connection_index);
                }
            }
            &AddConnection => {
                // This is faster than using FnvHashSet
                let possible_connections: Vec<usize> = (0..max_connection_index).filter(
                    |index| !self.connections.iter().any(|ref connection| connection.index == *index)).collect();

                if possible_connections.is_empty() {
                    // No more connections available, try a different mutation
                    self.mutate_node(rng, max_connection_index);
                } else {
                    let index = rng.gen_range::<usize>(0, possible_connections.len());
                    self.connections.push(Connection {
                        index: possible_connections[index],
                        weight: rng.gen_range::<f64>(-10.0, 10.0),
                    });
                }
            }
            &RemoveConnection => {
                if num_of_connections > 1 {
                    let index = rng.gen_range::<usize>(0, num_of_connections);
                    self.connections.remove(index);
                } else {
                    // Keep at least one connection, try a different mutation
                    self.mutate_node(rng, max_connection_index);
                }
            }
            &RandomConnectionOne => {
                // This is faster than using FnvHashSet
                let possible_connections: Vec<usize> = (0..max_connection_index).filter(
                    |index| !self.connections.iter().any(|ref connection| connection.index == *index)).collect();

                if possible_connections.is_empty() {
                    // No more connections available, try a different mutation
                    self.mutate_node(rng, max_connection_index);
                } else {
                    let index1 = rng.gen_range::<usize>(0, possible_connections.len());
                    let index2 = rng.gen_range::<usize>(0, num_of_connections);
                    self.connections[index2].index = possible_connections[index1];
                }
            }
            &RandomConnectionAll => {
                let mut possible_connections: Vec<usize> = (0..max_connection_index).collect();
                rng.shuffle(&mut possible_connections);

                for (connection, index) in self.connections.iter_mut().zip(possible_connections) {
                    connection.index = index;
                }
            }
            &DeltaBias1 => {
                self.bias += rng.gen_range::<f64>(-LIMIT1, LIMIT1);
            }
            &DeltaBias2 => {
                self.bias += rng.gen_range::<f64>(-LIMIT2, LIMIT2);
            }
            &DeltaBias3 => {
                self.bias += rng.gen_range::<f64>(-LIMIT3, LIMIT3);
            }
            &DeltaBias4 => {
                self.bias += rng.gen_range::<f64>(-LIMIT4, LIMIT4);
            }
            &RandomBias => {
                self.bias = rng.gen_range::<f64>(-10.0, 10.0);
            }
            &DeltaWeightOne1 => {
                let index = rng.gen_range::<usize>(0, num_of_connections);
                self.connections[index].weight += rng.gen_range::<f64>(-LIMIT1, LIMIT1);
            }
            &DeltaWeightOne2 => {
                let index = rng.gen_range::<usize>(0, num_of_connections);
                self.connections[index].weight += rng.gen_range::<f64>(-LIMIT2, LIMIT2);
            }
            &DeltaWeightOne3 => {
                let index = rng.gen_range::<usize>(0, num_of_connections);
                self.connections[index].weight += rng.gen_range::<f64>(-LIMIT3, LIMIT3);
            }
            &DeltaWeightOne4 => {
                let index = rng.gen_range::<usize>(0, num_of_connections);
                self.connections[index].weight += rng.gen_range::<f64>(-LIMIT4, LIMIT4);
            }
            &RandomWeightOne => {
                let index = rng.gen_range::<usize>(0, num_of_connections);
                self.connections[index].weight = rng.gen_range::<f64>(-10.0, 10.0);
            }
            &DeltaWeightAll1 => {
                for connection in &mut self.connections {
                    connection.weight += rng.gen_range::<f64>(-LIMIT1, LIMIT1);
                }
            }
            &DeltaWeightAll2 => {
                for connection in &mut self.connections {
                    connection.weight += rng.gen_range::<f64>(-LIMIT2, LIMIT2);
                }
            }
            &DeltaWeightAll3 => {
                for connection in &mut self.connections {
                    connection.weight += rng.gen_range::<f64>(-LIMIT3, LIMIT3);
                }
            }
            &DeltaWeightAll4 => {
                for connection in &mut self.connections {
                    connection.weight += rng.gen_range::<f64>(-LIMIT4, LIMIT4);
                }
            }
            &RandomWeightAll => {
                for connection in &mut self.connections {
                    connection.weight = rng.gen_range::<f64>(-10.0, 10.0);
                }
            }
        }
    }

    pub fn fix(&mut self, max_connection_index: usize) {
        let highest_connection = self.connections.iter().max_by(|c1, c2| c1.index.cmp(&c2.index)).unwrap().index;

        if highest_connection >= max_connection_index {
            let diff = highest_connection - max_connection_index + 1;

            for connection in &mut self.connections {
                if connection.index >= diff {
                    connection.index -= diff;
                } else {
                    connection.index = 0;
                }
            }
        }
    }

    pub fn add_used_nodes(&self, node_index: usize, set_of_used_nodes: &mut FnvHashSet<usize>) {
        for connection in &self.connections {
            if connection.index == node_index {
                // If node connects to it self it doesn't count as used
                continue
            }
            set_of_used_nodes.insert(connection.index);
        }
    }
}



#[cfg(test)]
mod test {
    // use super::*;
}
