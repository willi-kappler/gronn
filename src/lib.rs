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

- add more examples
- Change from enum to u8 for mutation operation

- Maybe different data structure:
    - https://github.com/fitzgen/generational-arena
    - https://github.com/orlp/slotmap
    - https://docs.rs/cranelift-entity/0.76.0/cranelift_entity/struct.PrimaryMap.html
    - https://github.com/petgraph/petgraph
    - https://github.com/tokio-rs/slab
    - https://github.com/SimonSapin/rust-typed-arena

*/
