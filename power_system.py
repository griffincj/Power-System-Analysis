import cmath
import copy
import math

import numpy as np
import pandas as pd
import cmath as cm
from config import config

from transformer import Transformer
from transmission_line import TransmissionLine
from bus import Bus
from generator import Generator
import warnings

warnings.filterwarnings("ignore")

class PowerSystem():
    def __init__(self):
        self.generators = []
        self.transformers = []
        self.transmission_lines = []
        self.buses = []
        self.y_bus = np.zeros((1, 1))
        self.z_bus = np.zeros((1,1))

        self.z_bus_pos = np.zeros((1, 1))
        self.z_bus_neg = np.zeros((1, 1))
        self.z_bus_zero = np.zeros((1, 1))

        self.y_magnitude = pd.DataFrame()
        self.y_theta = pd.DataFrame()
        self.j1 = pd.DataFrame((1, 1))
        self.j2 = pd.DataFrame((1, 1))
        self.j3 = pd.DataFrame((1, 1))
        self.j4 = pd.DataFrame((1, 1))

    def add_bus(self, bus: Bus):
        self.buses.append(bus)

    def add_generator(self, generator: Generator):
        self.generators.append(generator)

    def add_transmission_line(self, line: TransmissionLine):
        self.transmission_lines.append(line)

    def add_transformer(self, transformer: Transformer):
        self.transformers.append(transformer)

    def calc_real_power(self, cur_bus: Bus):
        real_power = 0

        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus

        for n in self.buses:
            real_power += (self.y_magnitude.loc[cur_bus.bus_name, n.bus_name] * bus_dict[n.bus_name].voltage
                           * np.cos(
                        bus_dict[cur_bus.bus_name].angle - bus_dict[n.bus_name].angle - self.y_theta.loc[
                            cur_bus.bus_name, n.bus_name]))

        real_power *= cur_bus.voltage
        cur_bus.power = real_power
        return

    def calc_reactive_power(self, cur_bus: Bus):
        reactive_power = 0

        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus

        for n in self.buses:
            reactive_power += (self.y_magnitude.loc[cur_bus.bus_name, n.bus_name] * bus_dict[n.bus_name].voltage
                               * np.sin(
                        bus_dict[cur_bus.bus_name].angle - bus_dict[n.bus_name].angle - self.y_theta.loc[
                            cur_bus.bus_name, n.bus_name]))

        reactive_power *= cur_bus.voltage
        cur_bus.reactive_power = reactive_power
        return

    def calc_power_loss(self):
        power_loss = 0
        for i_bus in self.buses:
            power_loss += i_bus.power
        return power_loss

    def calc_reactive_loss(self):
        reactive_loss = 0
        for i_bus in self.buses:
            reactive_loss += i_bus.reactive_power
        return reactive_loss

    def calc_ampacity_exceptions(self):
        # I believe something might be wrong here. In class, we discussed
        # current calculation to be |(va-vb)/z_series| where v1 and v2 are the rectangular voltages with mag and ang
        # I might be making a mistake in the conversion back to nominal, or maybe I'm doing
        # the above calculation incorrectly in this function
        print("\nCalculating ampacity exceptions...")
        for t_line in self.transmission_lines:
            # Convert angle magnitude and angle to rectangular/complex form
            va = cmath.rect(t_line.bus_a.voltage, t_line.bus_a.angle)
            vb = cmath.rect(t_line.bus_b.voltage, t_line.bus_b.angle)
            print("Bus " + str(t_line.bus_a.bus_name) + " to Bus " + str(t_line.bus_b.bus_name))
            pu_current = abs((va - vb) * t_line.sub_bus[1,0])
            unit_factor = (config.power_base/(math.sqrt(3) * t_line.bus_a.voltage_base)) * 1000
            current_amps = pu_current * unit_factor

            print("CURRENT IN AMPS: " + str(current_amps))
            if current_amps > t_line.conductor_bundle.conductor.ampacity:
                print("AMPACITY EXCEEDED\n")
            else:
                print("Ampacity not exceeded\n")


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

    def calc_z_bus(self):
        cur_y_bus = self.calculate_y_bus()
        self.z_bus = np.linalg.inv(cur_y_bus)
        pos_seq_y_bus = self.y_bus.copy()
        neg_seq_y_bus = self.y_bus.copy()
        zero_seq_y_bus = self.y_bus.copy()

        # Add the POSITIVE sequence reactance for each generator
        for gen in self.generators:
            bus_id = gen.bus.id
            pos_seq_y_bus[bus_id, bus_id] += (1/gen.pos_seq_x * 1j)
        # Z_bus is the inverse of the updated Y_bus
        self.z_bus_pos = np.linalg.inv(pos_seq_y_bus)

        # Add the NEGATIVE sequence reactance for each generator
        for gen in self.generators:
            bus_id = gen.bus.id
            neg_seq_y_bus[bus_id, bus_id] += (1/gen.neg_seq_x * 1j)
        # Z_bus is the inverse of the updated Y_bus
        self.z_bus_neg = np.linalg.inv(neg_seq_y_bus)

        for gen in self.generators:
            bus_id = gen.bus.id
            # Check if element is grounded. If not, z-bus elem = 1/inf = 0
            if(gen.zero_seq_x >= 0):
                zero_seq_y_bus[bus_id, bus_id] += (1/((gen.zero_seq_x * 1j) + (3 * gen.grounding)))
            else:
                zero_seq_y_bus[bus_id, bus_id] = 0
        # Z_bus is the inverse of the updated Y_bus
        self.z_bus_zero = np.linalg.inv(zero_seq_y_bus)

    def calc_balanced_fault(self, selected_bus, pre_fault_v):
        # Convert selected bus from bus name to bus ID
        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus
        cur_bus = bus_dict[selected_bus]

        # We want to use the voltage phasor for calculations
        prefaut_v_complex = cmath.rect(pre_fault_v, 0)

        # Current vector should be all zeros except for the faulted bus,
        # which should be -pre_fault_v / Z_bus[k,k]
        i_vector = np.zeros(shape=(len(self.buses), 1), dtype=complex)
        i_fault = prefaut_v_complex/(self.z_bus_pos[cur_bus.id, cur_bus.id])
        i_vector[cur_bus.id] = -i_fault

        fault_v_vector = np.matmul(self.z_bus_pos, i_vector)
        for i in range(fault_v_vector.shape[0]):
            fault_v_vector[i] = pre_fault_v - cmath.polar(fault_v_vector[i])[0]
        # Returned values are round to config.decimal_precision, since the 0 value was returning an exponential e^-16
        # Thus, it likely makes more sense to just display this as 0
        return np.round(np.real(i_fault), config.decimal_precision), np.round(fault_v_vector.astype(float),
                                                                              config.decimal_precision)

    def calc_unbalanced_fault(self, selected_bus, pre_fault_v, fault_type):
        # FAULT TYPES:
        # 1 = Single Line to Ground
        # 2 = LINE TO LINE
        # 3 = DOUBLE LINE TO GROUND
        # Convert selected bus from bus name to bus ID
        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus
        cur_bus = bus_dict[selected_bus]

        # We want to use the voltage phasor for calculations
        prefaut_v_complex = cmath.rect(pre_fault_v, 0)

        # Current vector should be all zeros except for the faulted bus,
        # which should be -pre_fault_v / Z_bus[k,k]
        i_vector = np.zeros(shape=(len(self.buses), 1), dtype=complex)

        # SINGLE LINE TO GROUND
        if fault_type == 1:
            # Get bus impedance from "regular" z-bus, i.e. non-sequence from inverting the y-bus used for power flows
            bus_impedance = self.z_bus[cur_bus.id, cur_bus.id]
            sum = (self.z_bus_zero[cur_bus.id, cur_bus.id] + self.z_bus_neg[cur_bus.id, cur_bus.id]
                   + self.z_bus_pos[cur_bus.id, cur_bus.id] + 3 * bus_impedance)
            i_fault = prefaut_v_complex / sum
            return i_fault
        # LINE TO LINE
        elif fault_type == 2:
            bus_impedance = self.z_bus[cur_bus.id, cur_bus.id]
            sum = self.z_bus_pos[cur_bus.id, cur_bus.id] + self.z_bus_neg[cur_bus.id, cur_bus.id] + bus_impedance
            pos_i_fault = prefaut_v_complex / sum
            return pos_i_fault
        elif fault_type == 3:
            return
        i_vector[cur_bus.id] = -i_fault

        fault_v_vector = np.matmul(self.z_bus_neg, i_vector)
        for i in range(fault_v_vector.shape[0]):
            fault_v_vector[i] = pre_fault_v - cmath.polar(fault_v_vector[i])[0]
        # Returned values are round to config.decimal_precision, since the 0 value was returning an exponential e^-16
        # Thus, it likely makes more sense to just display this as 0
        return np.round(np.real(i_fault), config.decimal_precision), np.round(fault_v_vector.astype(float),
                                                                              config.decimal_precision)

    def init_jacobian(self):
        """
        Function to initialize Jacobian matrix using buses provided
        to the PowerSystem
        """
        # Remove slack bus from buses in Jacobian
        jacobian_buses = [bus for bus in self.buses.copy() if bus.type != "SLACK"]
        # Determine number of PV and non-PV buses, as well as the number of buses in the Jacobian (i.e. minus slack)
        # Num Jacobian buses does include PV buses, since they do appear in some of the quadrants
        non_pv_buses = [bus for bus in jacobian_buses.copy() if bus.type != "PV"]
        num_PV_buses = len([bus for bus in self.buses if bus.type == "PV"])
        num_jac_buses = len(jacobian_buses)

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

    def print_jacobians(self):
        print('\nQUADRANT 1 JACOBIAN')
        print(self.j1)
        print('\nQUADRANT 2 JACOBIAN')
        print(self.j2)
        print('\nQUADRANT 3 JACOBIAN')
        print(self.j3)
        print('\nQUADRANT 4 JACOBIAN')
        print(self.j4)

    def run_newton_raphson(self, iterations: int, tolerance: float):
        print('Running Newton-Raphson...')
        self.init_jacobian()

        # Create bus_dict to refer to buses by name
        bus_dict = {}
        for bus in self.buses:
            bus_dict[bus.bus_name] = bus

        # Initialize a vector with the voltage angles+real power and magnitudes+reactive powers
        # every bus. This will be needed for the matrix multiplication with the Jacobian
        num_p_ang = (len(self.j1))
        num_q_mag = (len(self.j3))

        # Create delta y vector: the power mismatches
        delta_y = np.zeros(((num_p_ang + num_q_mag), 1))

        # Create a vector for all x's (14 x 1)
        x_full = np.zeros((len(bus_dict.keys()) * 2, 1))
        i_b = 0
        for b in self.buses:
            x_full[i_b, 0] = b.angle
            x_full[i_b + len(self.buses), 0] = b.voltage
            i_b += 1

        # Perform iterations of algorithm
        for iteration in range(0, iterations):
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
            y_ang_loc = 0
            y_mag_loc = 0 + num_p_ang
            for k in bus_dict.values():
                power_sum = 0
                reactive_i = 0
                if k.type == "SLACK":
                    x_full[int(k.bus_name) - 1][0] = float('-inf')
                    x_full[int(k.bus_name) - 1 + len(self.buses)][0] = float('-inf')
                    continue
                for n in self.buses:
                    reactive_i += (self.y_magnitude.loc[k.bus_name, n.bus_name] * bus_dict[n.bus_name].voltage
                                   * np.sin(
                                bus_dict[k.bus_name].angle - bus_dict[n.bus_name].angle - self.y_theta.loc[
                                    k.bus_name, n.bus_name]))
                    power_sum += (self.y_magnitude.loc[k.bus_name, n.bus_name] * bus_dict[n.bus_name].voltage
                                  * np.cos(
                                bus_dict[k.bus_name].angle - bus_dict[n.bus_name].angle - self.y_theta.loc[
                                    k.bus_name, n.bus_name]))
                reactive_i *= k.voltage
                power_sum *= k.voltage
                if k.type == "PQ":
                    delta_y[y_ang_loc][0] = bus_dict[k.bus_name].power - power_sum
                    delta_y[y_mag_loc][0] = bus_dict[k.bus_name].reactive_power - reactive_i
                    y_mag_loc += 1
                    y_ang_loc += 1
                elif k.type == "PV":
                    delta_y[y_ang_loc][0] = bus_dict[k.bus_name].power - power_sum
                    x_full[int(k.bus_name) - 1 + len(self.buses)][0] = float('-inf')
                    y_ang_loc += 1

            # Calculate inverse to find delta_x vector, used to update angles and voltages
            inverse_jacobian = np.linalg.inv(pd.DataFrame.to_numpy(full_jacobian))
            delta_x = np.matmul(inverse_jacobian, delta_y)

            # Fill the full x vector using the x_i vector calculated this iteration
            # Use d_x_i (delta_x_iterator) counter to separately iterate the smaller vector
            d_x_i = 0
            for i in range(x_full.shape[0]):
                if x_full[i][0] == float('-inf'):
                    continue
                else:
                    x_full[i][0] = delta_x[d_x_i][0]
                    d_x_i += 1

            # Update angles and voltages on each bus if not SLACK or PV
            for i in range(int(x_full.shape[0] / 2)):
                angle_update = x_full[i]
                mag_update = x_full[i + len(self.buses)]
                if angle_update != float('-inf'):
                    self.buses[i].angle += angle_update
                if mag_update != float('-inf'):
                    self.buses[i].voltage += mag_update

            if (np.abs(delta_y) <= tolerance).all():
                print("CONVERGED IN " + str(iteration+1) + " ITERATIONS\n")
                break
            elif iteration == (iterations - 1):
                print("DID NOT CONVERGE IN " + str(iterations) + " ITERATIONS\n")
                return

        for bus in self.buses:
            if bus.type == 'SLACK':
                self.calc_reactive_power(bus)
                self.calc_real_power(bus)
            if bus.type == 'PV':
                self.calc_reactive_power(bus)
            print(bus)

