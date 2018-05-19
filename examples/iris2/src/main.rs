#[macro_use] extern crate log;
extern crate simplelog;
extern crate gronn;

use std::fs::OpenOptions;

use simplelog::{WriteLogger, LevelFilter};

use gronn::driver::{Driver};

fn main() {
    WriteLogger::init(
        // LevelFilter::Debug,
        LevelFilter::Debug,
        simplelog::Config{time_format: Some("%Y-%m-%d %H:%M:%S"), .. simplelog::Config::default()},
        OpenOptions::new().append(true).create(true).open("iris.log").unwrap());

    let mut driver = Driver::new_from_file("config.json").unwrap();

    driver.train_from_file("iris_data.json");

    info!("Result: {:?} -> 1", driver.predict(&[5.2,4.1,1.5,0.1]));
    info!("Result: {:?} -> 1", driver.predict(&[5.5,4.2,1.4,0.2]));
    info!("Result: {:?} -> 2", driver.predict(&[5.0,2.0,3.5,1.0]));
    info!("Result: {:?} -> 2", driver.predict(&[5.9,3.0,4.2,1.5]));
    info!("Result: {:?} -> 3", driver.predict(&[6.9,3.1,5.1,2.3]));
    info!("Result: {:?} -> 3", driver.predict(&[5.8,2.7,5.1,1.9]));

    driver.save_network("optimal_configuration.json");
}
