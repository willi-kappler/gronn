use serde_json;

use network::{Network};
use driver::{DriverConfiguration};

pub fn xor01(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/xor01.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "xor01");
    network.fix();
    network
}

pub fn xor02(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/xor02.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "xor02");
    network.fix();
    network
}

pub fn xor03(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/xor03.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "xor03");
    network.fix();
    network
}

pub fn iris01(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/iris01.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "iris01");
    network.fix();
    network
}

pub fn iris02(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/iris02.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "iris02");
    network.fix();
    network
}

pub fn iris03(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/iris03.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "iris03");
    network.fix();
    network
}

pub fn adder01(configuration: DriverConfiguration) -> Network {
    let property_json = include_str!("trained_networks/adder01.json");
    let mut network = Network::new_with_property(configuration, serde_json::from_str(property_json).unwrap(), "adder01");
    network.fix();
    network
}
