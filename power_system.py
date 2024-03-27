import numpy as np
import pandas as pd
import cmath as cm

from transformer import Transformer
from transmission_line import TransmissionLine
from bus import Bus


class PowerSystem():
    def __init__(self):
        self.transformers = []
        self.transmission_lines = []
        self.buses = []
        self.y_bus = np.zeros((1, 1))
        self.y_magnitude = pd.DataFrame()
        self.y_theta = pd.DataFrame()
        self.j1 = pd.DataFrame((1, 1))
        self.j2 = pd.DataFrame((1, 1))
        self.j3 = pd.DataFrame((1, 1))
        self.j4 = pd.DataFrame((1, 1))

    def solve(self):
        return "Power System Solved"

    def add_bus(self, bus: Bus):
        self.buses.append(bus)

    def add_transmission_line(self, line: TransmissionLine):
        self.transmission_lines.append(line)

    def add_transformer(self, transformer: Transformer):
        self.transformers.append(transformer)

    def calculate_y_bus(self):
        num_buses = len(self.buses)
        self.y_bus = np.zeros(shape=(num_buses, num_buses), dtype=np.complex_)
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

        # Declare and initialize the polar form of the y_bus
        # self.y_magnitude = np.zeros([num_buses, num_buses])
        # self.y_theta = np.zeros([num_buses, num_buses])

        self.y_magnitude = pd.DataFrame(data=np.zeros(shape=(len(self.buses), len(self.buses))),
                                        index=[bus.bus_name for bus in self.buses],
                                        columns=[bus.bus_name for bus in self.buses])
        self.y_theta = pd.DataFrame(data=np.zeros(shape=(len(self.buses), len(self.buses))),
                                    index=[bus.bus_name for bus in self.buses],
                                    columns=[bus.bus_name for bus in self.buses])
        for row in range(self.y_bus.shape[0]):
            for col in range(self.y_bus.shape[1]):
                polar = cm.polar(self.y_bus[row][col])
                self.y_magnitude.loc[str(row + 1), str(col + 1)] = polar[0]
                self.y_theta.loc[str(row + 1), str(col + 1)] = polar[1]
        return self.y_bus

    def init_jacobian(self):
        """
        Function to initialize Jacobian matrix using buses provided
        to the PowerSystem
        """
        # Remove slack bus from buses in Jacobian
        jacobian_buses = [bus for bus in self.buses.copy() if bus.type != "SLACK"]
        non_pv_buses = [bus for bus in jacobian_buses.copy() if bus.type != "PV"]

        num_jac_buses = len(jacobian_buses)
        num_PV_buses = len([bus for bus in self.buses if bus.type == "PV"])

        # Initialize each quadrant of Jacobian to be appropriate size, filled with 0s
        self.j1 = pd.DataFrame(data=np.zeros(shape=(num_jac_buses, num_jac_buses)),
                               index=[bus.bus_name for bus in jacobian_buses],
                               columns=[bus.bus_name for bus in jacobian_buses])

        self.j2 = pd.DataFrame(data=np.zeros(shape=(num_jac_buses, num_jac_buses - num_PV_buses)),
                               index=[bus.bus_name for bus in jacobian_buses],
                               columns=[bus.bus_name for bus in non_pv_buses])

        self.j3 = pd.DataFrame(data=np.zeros(shape=(num_jac_buses - num_PV_buses, num_jac_buses)),
                               index=[bus.bus_name for bus in non_pv_buses],
                               columns=[bus.bus_name for bus in jacobian_buses])

        self.j4 = pd.DataFrame(data=np.zeros(shape=(num_jac_buses - num_PV_buses, num_jac_buses - num_PV_buses)),
                               index=[bus.bus_name for bus in non_pv_buses],
                               columns=[bus.bus_name for bus in non_pv_buses])

    def calc_jacobian_quad_1(self):
        """
        Function to calculate 1st Quadrant Jacobian: Power / Angle
        """
        rows = self.j1.index
        cols = self.j1.columns
        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus

        # K and N are both strings, starting at 2
        for k in rows:
            for n in cols:
                if k == n:
                    # Diagonals
                    cur_sum = 0
                    # bus_dict.keys starts at 1, goes through all buses (including slack and PV)
                    for i in bus_dict.keys():
                        if i != k:
                            cur_sum += (self.y_magnitude.loc[k, i] * bus_dict[i].voltage *
                                        np.sin(bus_dict[k].angle - bus_dict[i].angle - self.y_theta.loc[k, i]))
                    self.j1.loc[k, n] = (-bus_dict[k].voltage * cur_sum)
                else:
                    # Off-diagonals
                    self.j1.loc[k, n] = (bus_dict[k].voltage * self.y_magnitude.loc[k, n] * bus_dict[n].voltage *
                                         np.sin(bus_dict[k].angle - bus_dict[n].angle - self.y_theta.loc[k, n]))

    def calc_jacobian_quad_2(self):
        """
        Function to calculate 2nd Quadrant Jacobian: Power / Voltage Mag
        """
        rows = self.j2.index
        cols = self.j2.columns
        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus

        # K and N are both strings, starting at 2
        for k in rows:
            for n in cols:
                if k == n:
                    # Diagonals
                    cur_sum = 0
                    # bus_dict.keys starts at 1, goes through all buses (including slack and PV)
                    for i in bus_dict.keys():
                        cur_sum += (self.y_magnitude.loc[k, i] * bus_dict[i].voltage *
                                    np.cos(bus_dict[k].angle - bus_dict[i].angle - self.y_theta.loc[k, i]))
                    self.j2.loc[k, n] = (bus_dict[k].voltage * self.y_magnitude.loc[k, k] *
                                         np.cos(self.y_theta.loc[k, k]) + cur_sum)
                else:
                    # Off-diagonals
                    self.j2.loc[k, n] = (bus_dict[k].voltage * self.y_magnitude.loc[k, n] *
                                         np.cos(bus_dict[k].angle - bus_dict[n].angle - self.y_theta.loc[k, n]))

    def calc_jacobian_quad_3(self):
        """
        Function to calculate 3rd Quadrant Jacobian: Reactive Power / Angle
        """
        rows = self.j3.index
        cols = self.j3.columns
        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus

        # K and N are both strings, starting at 2
        for k in rows:
            for n in cols:
                if k == n:
                    # Diagonals
                    cur_sum = 0
                    # bus_dict.keys starts at 1, goes through all buses (including slack and PV)
                    for i in bus_dict.keys():
                        if i != k:
                            cur_sum += (self.y_magnitude.loc[k, i] * bus_dict[i].voltage *
                                        np.cos(bus_dict[k].angle - bus_dict[i].angle - self.y_theta.loc[k, i]))
                    self.j3.loc[k, n] = (bus_dict[k].voltage * cur_sum)
                else:
                    # Off-diagonals
                    self.j3.loc[k, n] = (-bus_dict[k].voltage * self.y_magnitude.loc[k, n] * bus_dict[n].voltage *
                                         np.cos(bus_dict[k].angle - bus_dict[n].angle - self.y_theta.loc[k, n]))

    def calc_jacobian_quad_4(self):
        """
        Function to calculate 4th Quadrant Jacobian: Reactive Power / Voltage Mag
        """
        rows = self.j4.index
        cols = self.j4.columns
        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus

        # K and N are both strings, starting at 2
        for k in rows:
            for n in cols:
                if k == n:
                    # Diagonals
                    cur_sum = 0
                    # bus_dict.keys starts at 1, goes through all buses (including slack and PV)
                    for i in bus_dict.keys():
                        cur_sum += (self.y_magnitude.loc[k, i] * bus_dict[i].voltage *
                                    np.sin(bus_dict[k].angle - bus_dict[i].angle - self.y_theta.loc[k, i]))
                    self.j4.loc[k, n] = (-bus_dict[k].voltage * self.y_magnitude.loc[k, k]
                                         * np.sin(self.y_theta.loc[k, k]) + cur_sum)
                else:
                    # Off-diagonals
                    self.j4.loc[k, n] = (bus_dict[k].voltage * self.y_magnitude.loc[k, n] *
                                         np.sin(bus_dict[k].angle - bus_dict[n].angle - self.y_theta.loc[k, n]))

    def run_newton_raphson(self, iterations):
        print('Running Newton-Raphson')
        self.init_jacobian()

        # Create bus_dict to refer to buses by name
        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus
        N = len(bus_dict.keys())

        # Initialize a vector with the voltage angles+real power and magnitudes+reactive powers
        # every bus. This will be needed for the matrix multiplication with the Jacobian
        num_p_ang = (len(self.j1))
        num_q_mag = (len(self.j3))

        # Create delta y vector: the power mismatches
        delta_y = np.zeros(((num_p_ang + num_q_mag), 1))

        # Create voltage angle and magnitude vector
        volt_angle_mag = np.zeros((1, (num_p_ang + num_q_mag)))

        # Create a delta_v vector with all buses, including slack and PV
        update_v_attr = np.zeros((len(bus_dict.keys())*2, 1))
        print(len(update_v_attr))

        # Perform iterations of algorithm
        for i in range(0, iterations):
            self.calc_jacobian_quad_1()
            self.calc_jacobian_quad_2()
            self.calc_jacobian_quad_3()
            self.calc_jacobian_quad_4()

            # Concat partial Jacobians to form full, 4-quadrant Jacobian
            j1_j2 = pd.concat([self.j1, self.j2], axis=1, ignore_index=True)
            j3_j4 = pd.concat([self.j3, self.j4], axis=1, ignore_index=True)
            full_jacobian = pd.concat([j1_j2, j3_j4], axis=0, ignore_index=True)

            # Calculate delta_y
            # Both mismatches have to be of length 11
            # Fill real power and voltage angle
            for p in range(0, num_p_ang):
                volt_angle_mag[0, p] = self.buses[p].angle
                power_sum = 0
                for n in bus_dict.keys():
                    power_sum += (self.y_magnitude.loc[str(p + 1), n] * bus_dict[n].voltage
                                  * np.cos(
                                bus_dict[str(p + 1)].angle - bus_dict[n].angle - self.y_theta.loc[str(p + 1), n]))
                delta_y[p][0] = bus_dict[str(p+1)].power - power_sum
                update_v_attr[p+1] = 1

            # Fill reactive power and voltage magnitude
            for q in range(0, num_q_mag):
                if bus_dict[str(q+1)].type != 'PQ':
                    continue
                volt_angle_mag[0, q + num_p_ang] = self.buses[q].voltage
                reactive_sum = 0
                for n in bus_dict.keys():
                    reactive_sum += (self.y_magnitude.loc[str(q + 1), n] * bus_dict[n].voltage
                                     * np.sin(
                                bus_dict[str(q + 1)].angle - bus_dict[n].angle - self.y_theta.loc[str(q + 1), n]))
                delta_y[q + num_p_ang][0] = bus_dict[str(q + 1)].reactive_power - reactive_sum

            # Check convergence criteria
            inverse_jacobian = np.linalg.inv(pd.DataFrame.to_numpy(full_jacobian))
            delta_v = np.matmul(inverse_jacobian, delta_y)

            #TODO: Update voltage magnitude and angle
            print(str('Voltages Mismatches' + str(delta_v)))
