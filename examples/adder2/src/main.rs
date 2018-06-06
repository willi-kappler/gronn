#[macro_use] extern crate log;
extern crate log4rs;
extern crate gronn;

use gronn::driver::{Driver, TrainingData};

fn main() {
    let file_logger = log4rs::append::file::FileAppender::builder()
        .encoder(Box::new(log4rs::encode::pattern::PatternEncoder::new("{d} {l} - {m}{n}")))
        .build("adder.log").unwrap();

    let config = log4rs::config::Config::builder()
        .appender(log4rs::config::Appender::builder().build("file_logger", Box::new(file_logger)))
        .build(log4rs::config::Root::builder().appender("file_logger").build(log::LevelFilter::Info))
        .unwrap();

    let _log_handle = log4rs::init_config(config).unwrap();

    let mut driver = Driver::new_from_file("config.toml").unwrap();

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

    driver.train(&training_data).unwrap();

    info!("Result: {:?} -> 0 0 1", driver.predict(&[0.0, 0.0, 0.0]));
    info!("Result: {:?} -> 0 1 1", driver.predict(&[0.0, 1.0, 0.0]));
    info!("Result: {:?} -> 1 0 1", driver.predict(&[1.0, 0.0, 0.0]));
    info!("Result: {:?} -> 0 0 0", driver.predict(&[1.0, 1.0, 1.0]));

    driver.save_network("optimal_configuration.toml").unwrap();
}
