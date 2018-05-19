#[macro_use] extern crate serde_derive;
#[macro_use] extern crate rand_derive;
#[macro_use] extern crate log;
#[macro_use] extern crate failure;

#[cfg(test)]
#[macro_use] extern crate assert_approx_eq;

#[cfg(test)]
extern crate simplelog;

extern crate rand;
extern crate serde;
extern crate serde_json;
extern crate fnv;
extern crate rayon;

mod node;
mod property;
mod network;
mod network_configurations;
pub mod driver;

/*
TODO:

- use TOML instead of JSON ?
- pass TrainingData and index around instead of provided_input and expected_output
- fix time stamp utc -> local time
- add more examples
- Change from enum to u8 for mutation operation

*/
