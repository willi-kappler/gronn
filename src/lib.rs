#[macro_use] extern crate serde_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate failure;

extern crate rand;
extern crate serde;
extern crate toml;
extern crate fnv;
extern crate rayon;

mod node;
mod property;
mod network;
mod network_configurations;
pub mod driver;

/*
TODO:

- pass TrainingData and index around instead of provided_input and expected_output
- add more examples
- Change from enum to u8 for mutation operation
- Update to random 0.5: rand::seq::sample_indices for batches

*/
