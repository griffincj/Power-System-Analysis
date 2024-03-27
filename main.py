import cmath

from transformer import Transformer
from transmission_line import TransmissionLine
from transmission_line import Conductor, ConductorBundle
from bus import Bus
from load import Load
from generator import Generator
from config import config
from power_system import PowerSystem
import numpy as np


def print_example():
    ####################
    # Homework example #
    ####################
    hw_busA = Bus(voltage_base=20, bus_name="A")
    hw_busB = Bus(voltage_base=500, bus_name="B")
    finch_conductor = Conductor(gmr=0.0435, r_per_mile=0.0969, out_diameter=1.293)
    finch_bundle = ConductorBundle(conductor=finch_conductor, num_conductors=3, conductor_distance=1)
    hw_xfmr = Transformer(rating=500, bus_a=hw_busA, bus_b=hw_busB, z_pu=0.075, x_r_ratio=13)
    hw_tl = TransmissionLine(conductor_bundle=finch_bundle, distances=[12.5, 12.5, 20],
                             length=100, bus_a=hw_busB, bus_b=hw_busB)
    print("###########\nSAMPLE OUTPUT (from Homework)\n###########\n")
    print("HW XFMR Impedance PU: " + str(hw_xfmr.z_pu_sys))
    print("HW XFMR Admittance PU: " + str(hw_xfmr.yt_pu) + "\n")

    print("HW TL Impedance PU: " + str(hw_tl.z))
    print("HW TL Susceptance PU: " + str(hw_tl.b))
    print("HW TL Admittance PU: " + str(hw_tl.ytl_pu))


def print_y_bus(power_system: PowerSystem, decimals=2):
    rounded_y_bus = np.around(power_system.y_magnitude, decimals=decimals)
    length_check = np.vectorize(len)
    max_length = np.amax(length_check(rounded_y_bus.astype(dtype=str))) + 1
    for row in range(len(rounded_y_bus)):
        cur_row = ""
        for col in range(len(power_system.y_magnitude)):
            cur_row += (" " * (max_length - len(str(rounded_y_bus[row, col])))) + (str(rounded_y_bus[row, col]))
        print(cur_row)


if __name__ == '__main__':
    # Initialize Power System instance
    ps = PowerSystem()

    # Initialize buses
    # Bus 1 is the SLACK bus
    bus_1 = Bus(voltage_base=20, bus_name="1")
    ps.add_bus(bus_1)
    bus_2 = Bus(voltage_base=230, bus_name="2")
    ps.add_bus(bus_2)
    bus_3 = Bus(voltage_base=230, bus_name="3")
    ps.add_bus(bus_3)
    bus_4 = Bus(voltage_base=230, bus_name="4")
    ps.add_bus(bus_4)
    bus_5 = Bus(voltage_base=230, bus_name="5")
    ps.add_bus(bus_5)
    bus_6 = Bus(voltage_base=230, bus_name="6")
    ps.add_bus(bus_6)
    bus_7 = Bus(voltage_base=18, bus_name="7")
    ps.add_bus(bus_7)

    bus_1.voltage = 1.0
    bus_1.angle = 0
    bus_1.type = "SLACK"
    load_2 = Load(pu_power=(0 / config.power_base), pu_reactive_power=(0 / config.power_base), connected_bus=bus_2)
    load_3 = Load(pu_power=(110 / config.power_base), pu_reactive_power=(50 / config.power_base), connected_bus=bus_3)
    load_4 = Load(pu_power=(100 / config.power_base), pu_reactive_power=(70 / config.power_base), connected_bus=bus_4)
    load_5 = Load(pu_power=(100 / config.power_base), pu_reactive_power=(65 / config.power_base), connected_bus=bus_5)
    load_6 = Load(pu_power=(0 / config.power_base), pu_reactive_power=(0 / config.power_base), connected_bus=bus_6)
    gen_7 = Generator(pu_power=(200 / config.power_base), pu_voltage=1.0, connected_bus=bus_7)

    # Transformer low-side, high-side is based on the buses connected on either side of it
    t1 = Transformer(rating=125, bus_a=bus_1, bus_b=bus_2, z_pu=0.085, x_r_ratio=10)
    ps.add_transformer(t1)
    t2 = Transformer(rating=200, bus_a=bus_6, bus_b=bus_7, z_pu=0.105, x_r_ratio=12)
    ps.add_transformer(t2)

    # Initialize base conductor and bundle, used by each transmission line
    partridge_conductor = Conductor(gmr=0.0217, r_per_mile=0.385, out_diameter=0.642)

    # Initialize bundle, composed of given conductor type
    bundle = ConductorBundle(conductor=partridge_conductor, num_conductors=2, conductor_distance=1.5)

    # Calculated distances
    distances = [19.5, 19.5, 39]

    # Add transmission lines
    tl1 = TransmissionLine(conductor_bundle=bundle, distances=distances, length=10, bus_a=bus_2, bus_b=bus_4)
    tl2 = TransmissionLine(conductor_bundle=bundle, distances=distances, length=25, bus_a=bus_2, bus_b=bus_3)
    tl3 = TransmissionLine(conductor_bundle=bundle, distances=distances, length=20, bus_a=bus_3, bus_b=bus_5)
    tl4 = TransmissionLine(conductor_bundle=bundle, distances=distances, length=20, bus_a=bus_4, bus_b=bus_6)
    tl5 = TransmissionLine(conductor_bundle=bundle, distances=distances, length=10, bus_a=bus_5, bus_b=bus_6)
    tl6 = TransmissionLine(conductor_bundle=bundle, distances=distances, length=35, bus_a=bus_4, bus_b=bus_5)
    ps.add_transmission_line(tl1)
    ps.add_transmission_line(tl2)
    ps.add_transmission_line(tl3)
    ps.add_transmission_line(tl4)
    ps.add_transmission_line(tl5)
    ps.add_transmission_line(tl6)

    ps.calculate_y_bus()
    ps.run_newton_raphson(1)