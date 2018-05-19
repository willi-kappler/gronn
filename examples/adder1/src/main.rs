#[macro_use] extern crate log;
extern crate simplelog;
extern crate gronn;

use std::fs::OpenOptions;

use simplelog::{WriteLogger, LevelFilter};

use gronn::driver::{Driver, TrainingData};

fn main() {
    WriteLogger::init(
        // LevelFilter::Debug,
        LevelFilter::Debug,
        simplelog::Config{time_format: Some("%Y-%m-%d %H:%M:%S"), .. simplelog::Config::default()},
        OpenOptions::new().append(true).create(true).open("adder.log").unwrap());

    let mut driver = Driver::new_from_file("config.json").unwrap();
    let training_data = TrainingData {
        provided_input: vec![
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ],
        expected_output: vec![
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 0.0, 0.0],
            vec![1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ],
    };

    driver.train(&training_data);

    info!("Result: {:?} -> 0 0 1", driver.predict(&[0.0, 0.0, 0.0]));
    info!("Result: {:?} -> 0 1 1", driver.predict(&[0.0, 1.0, 0.0]));
    info!("Result: {:?} -> 1 0 1", driver.predict(&[1.0, 0.0, 0.0]));
    info!("Result: {:?} -> 0 0 0", driver.predict(&[1.0, 1.0, 1.0]));

    driver.save_network("optimal_configuration.json");
}
