from enum import Enum


class Bus:
    num_buses = 0

    def __init__(self, voltage_base, bus_name):
        self.id = self.num_buses
        self.voltage_base = voltage_base
        self.bus_name = bus_name
        self.voltage = 1.0
        self.angle = 0.0
        self.power = 0.0
        self.reactive_power = 0.0
        self.type = "SLACK"
        Bus.num_buses += 1

if __name__ == '__main__':
    bus_1 = Bus(voltage_base=20, bus_name="1")
    bus_2 = Bus(voltage_base=230, bus_name="2")