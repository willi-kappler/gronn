use toml;

use network::{Network};
use driver::{DriverConfiguration};

pub fn xor02(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/xor02_1.toml");
    let mut network = Network::new_with_property(configuration, toml::from_str(property_json).unwrap(), "xor02");
    network.fix();
    network
}

pub fn xor05(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/xor05_01.toml");
    let mut network = Network::new_with_property(configuration, toml::from_str(property_json).unwrap(), "xor05");
    network.fix();
    network
}
pub fn iris03(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/iris03.toml");
    let mut network = Network::new_with_property(configuration, toml::from_str(property_json).unwrap(), "iris03");
    network.fix();
    network
}

pub fn adder04(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/adder04.toml");
    let mut network = Network::new_with_property(configuration, toml::from_str(property_json).unwrap(), "adder04");
    network.fix();
    network
}

pub fn adder06(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/adder06.toml");
    let mut network = Network::new_with_property(configuration, toml::from_str(property_json).unwrap(), "adder06");
    network.fix();
    network
}
