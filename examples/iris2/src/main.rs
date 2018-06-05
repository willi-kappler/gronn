#[macro_use] extern crate log;
extern crate log4rs;
extern crate gronn;

use gronn::driver::{Driver};

fn main() {
    let file_logger = log4rs::append::file::FileAppender::builder()
        .encoder(Box::new(log4rs::encode::pattern::PatternEncoder::new("{d} {l} - {m}{n}")))
        .build("iris.log").unwrap();

    let config = log4rs::config::Config::builder()
        .appender(log4rs::config::Appender::builder().build("file_logger", Box::new(file_logger)))
        .build(log4rs::config::Root::builder().appender("file_logger").build(log::LevelFilter::Info))
        .unwrap();

    let _log_handle = log4rs::init_config(config).unwrap();

    let mut driver = Driver::new_from_file("config.toml").unwrap();

    driver.train_from_file("iris_data.toml");

    info!("Result: {:?} -> 1", driver.predict(&[5.2,4.1,1.5,0.1]));
    info!("Result: {:?} -> 1", driver.predict(&[5.5,4.2,1.4,0.2]));
    info!("Result: {:?} -> 2", driver.predict(&[5.0,2.0,3.5,1.0]));
    info!("Result: {:?} -> 2", driver.predict(&[5.9,3.0,4.2,1.5]));
    info!("Result: {:?} -> 3", driver.predict(&[6.9,3.1,5.1,2.3]));
    info!("Result: {:?} -> 3", driver.predict(&[5.8,2.7,5.1,1.9]));

    driver.save_network("optimal_configuration.toml").unwrap();
}
