use serde_json;

use network::{Network};
use driver::{DriverConfiguration};

pub fn xor02(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/xor02.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "xor02");
    network.fix();
    network
}

pub fn xor05(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/xor05_01.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "xor05");
    network.fix();
    network
}
pub fn iris03(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/iris03.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "iris03");
    network.fix();
    network
}

pub fn adder04(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/adder04.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "adder04");
    network.fix();
    network
}

pub fn adder06(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/adder06_1.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "adder06");
    network.fix();
    network
}
