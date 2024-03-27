from bus import Bus
from config import config


class Load:
    def __init__(self, pu_power, pu_reactive_power, connected_bus: Bus):
        self.power = pu_power
        self.reactive_power = pu_reactive_power
        self.bus = connected_bus

        if connected_bus.type != "PV":
            connected_bus.type = "PQ"

        connected_bus.power -= self.power
        connected_bus.reactive_power -= self.reactive_power
