import numpy as np

from transformer import Transformer
from transmission_line import TransmissionLine
from bus import Bus


class PowerSystem():
    def __init__(self):
        self.transformers = []
        self.transmission_lines = []
        self.buses = []
        self.y_bus = np.zeros((1, 1))

    def solve(self):
        return "Power System Solved"

    def add_bus(self, bus: Bus):
        self.buses.append(bus)

    def add_transmission_line(self, line: TransmissionLine):
        self.transmission_lines.append(line)

    def add_transformer(self, transformer: Transformer):
        self.transformers.append(transformer)

    def calculate_y_bus(self):
        self.y_bus = np.zeros(shape=(len(self.buses), len(self.buses)), dtype=np.complex_)
        # This is currently in a "dangerous" state that assumes all element types have attribute named
        # "bus_a" and "bus_b".
        # TODO: refactor to use inheritance for guaranteed consistency
        elements = list(self.transformers + self.transmission_lines)
        for element in elements:
            a = element.bus_a
            b = element.bus_b
            sub_bus = element.sub_bus
            self.y_bus[a.id, a.id] += sub_bus[0, 0]
            self.y_bus[a.id, b.id] += sub_bus[0, 1]
            self.y_bus[b.id, a.id] += sub_bus[1, 0]
            self.y_bus[b.id, b.id] += sub_bus[1, 1]
        return self.y_bus
