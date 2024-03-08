import numpy as np

from config import config
from bus import Bus


class Conductor:
    """
    Specification for a conductor
    """
    def __init__(self, gmr, r_per_mile, out_diameter):
        self.gmr = gmr
        self.r_per_mile = r_per_mile
        self.out_radius = (out_diameter / 2) * 1 / 12


class ConductorBundle:
    """
    Bundle of specified conductor type; to be used in a transmission line
    """
    def __init__(self, conductor, num_conductors, conductor_distance):
        self.num_conductors = num_conductors
        self.conductor = conductor
        self.distance = conductor_distance
        self.dsl = self.calc_dsl()
        self.dsc = self.calc_dsc()

    def calc_dsl(self):
        if self.num_conductors == 2:
            return (self.distance * self.conductor.gmr) ** (1 / 2)
        elif self.num_conductors == 3:
            return ((self.distance ** 2) * self.conductor.gmr) ** (1 / 3)
        elif self.num_conductors == 4:
            return 1.041 * ((self.distance ** 3) * self.conductor.gmr) ** (1 / 4)

    def calc_dsc(self):
        if self.num_conductors == 2:
            return (self.distance * self.conductor.out_radius) ** (1 / 2)
        elif self.num_conductors == 3:
            return ((self.distance ** 2) * self.conductor.out_radius) ** (1 / 3)
        elif self.num_conductors == 4:
            return 1.041 * ((self.distance ** 3) * self.conductor.out_radius) ** (1 / 4)


class TransmissionLine:
    """
    Transmission line composed of one or more conductor bundles for specified length
    """
    PERMITTIVITY = 8.854 * 10 ** (-12)

    def __init__(self, conductor_bundle, length, distances, bus_a: Bus, bus_b: Bus, num_bundles=3):
        if len(distances) != num_bundles:
            raise ValueError("Number of distances does not match number of bundles")
        self.conductor_bundle = conductor_bundle
        self.num_bundles = num_bundles
        self.length = length
        self.distances = distances
        self.bus_a = bus_a
        self.bus_b = bus_b
        self.voltage = bus_a.voltage_base
        self.z_base = (self.voltage ** 2) / config.power_base

        self.deq = self.calc_deq()
        self.x = self.calc_reactance()
        self.b = self.calc_susceptance() * 1j
        self.r = self.calc_resistance()
        self.z = self.calc_impedance()
        self.ytl_pu = 1 / self.z
        self.sub_bus = self.calc_bus_admittance_matrix()

    def calc_deq(self):
        return np.prod(self.distances) ** (1 / self.num_bundles)

    def calc_impedance(self):
        return self.r + self.x * 1j

    def calc_resistance(self):
        per_phase_r = self.conductor_bundle.conductor.r_per_mile / self.num_bundles
        total_r = per_phase_r * self.length
        return total_r / self.z_base

    def calc_reactance(self):
        log_term = np.log(self.deq / self.conductor_bundle.dsl)
        per_mile_x = 377.1609 * (2 * 10 ** -7) * log_term * 1609
        total_x = per_mile_x * self.length
        return total_x / self.z_base

    def calc_susceptance(self):
        numerator = (2 * np.pi * self.PERMITTIVITY)
        denominator = np.log(self.deq / self.conductor_bundle.dsc)
        per_mile_b = 377 * (numerator / denominator) * 1609
        total_b = per_mile_b * self.length
        return total_b / (1 / self.z_base)

    def calc_bus_admittance_matrix(self):
        # Calculate "sub-bus" of size 2 x 2, with indexes referring to bus A and bus B
        # DType must be set to "np.complex_", otherwise the matrix will not allow complex numbers
        sub_bus = np.zeros(shape=(2, 2), dtype=np.complex_)
        iterator = np.nditer(sub_bus, flags=['multi_index'])

        while not iterator.finished:
            if iterator.multi_index[0] == iterator.multi_index[1]:
                # Diagonal elements -> sum of shunt susceptance (complex) and admittance
                # Shunt susceptance must be split in half, representing each side of the line
                sub_bus[iterator.multi_index] = self.ytl_pu + (self.b / 2)
            else:
                # Off-diagonal elements -> negative, don't include shunt susceptance
                sub_bus[iterator.multi_index] = -1 * self.ytl_pu
            iterator.iternext()
        return sub_bus
