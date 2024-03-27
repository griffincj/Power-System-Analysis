from bus import Bus


class Generator:
    def __init__(self, pu_power, pu_voltage, connected_bus: Bus):
        self.power = pu_power
        self.voltage = pu_voltage
        self.bus = connected_bus
        connected_bus.type = "PV"

        connected_bus.power = self.power
        connected_bus.voltage = self.voltage
