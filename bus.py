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

    def __str__(self):
        return (f"Bus # {self.bus_name} \nVOLTAGE PU {self.voltage}\n"
                f"VOLTAGE ANGLE {self.angle}\nREAL POWER PU {self.power}\n"
                f"REACTIVE POWER PU {self.reactive_power}\n")