import cmath
import numpy as np
from config import config
from bus import Bus


class Transformer:
    def __init__(self, rating, bus_a: Bus, bus_b: Bus, z_pu, x_r_ratio, grounding):
        self.rating = rating
        self.bus_a = bus_a
        self.bus_b = bus_b
        self.z_pu = z_pu
        self.z_pu_sys = 0
        self.x_r_ratio = x_r_ratio
        self.theta = np.arctan(self.x_r_ratio)
        self.x_pu = 0
        self.r_pu = 0
        self.calc_impedance_attrs()
        self.yt_pu = 1 / self.z_pu_sys
        self.sub_bus = self.calc_bus_admittance_matrix()
        self.grounding = grounding

    def calc_impedance_attrs(self):
        self.z_pu_sys = self.z_pu * (config.power_base / self.rating)
        self.z_pu_sys = cmath.rect(self.z_pu_sys, self.theta)

        # These are not needed, but can be used for testing
        self.x_pu = np.imag(self.z_pu_sys)
        self.r_pu = np.real(self.z_pu_sys)
        return

    def calc_bus_admittance_matrix(self):
        # Calculate "sub-bus" of size 2 x 2, with indexes referring to bus A and bus B
        # DType must be set to "np.complex_", otherwise the matrix will not allow complex numbers
        sub_bus = np.zeros(shape=(2,2), dtype=np.complex_)
        iterator = np.nditer(sub_bus, flags=['multi_index'])
        while not iterator.finished:
            # Make ytl_pu (1) or (-1) depending on whether it is diagonal or not
            if iterator.multi_index[0] == iterator.multi_index[1]:
                sub_bus[iterator.multi_index] = self.yt_pu
            else:
                sub_bus[iterator.multi_index] = -1 * self.yt_pu
            iterator.iternext()
        return sub_bus

if __name__ == '__main__':
    bus_1 = Bus(voltage_base=20, bus_name="1")
    bus_2 = Bus(voltage_base=230, bus_name="2")
    t1 = Transformer(rating=125, bus_a=bus_1, bus_b=bus_2, z_pu=0.085, x_r_ratio=10)
    t1.calc_bus_admittance_matrix()
    print(t1.sub_bus)