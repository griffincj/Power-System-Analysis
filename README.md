# Power-System-Analysis
This is the project for the ECE2774 project

For milestone 1, ``main.py`` contains all initializations of each instance of the 
classes required for the project model. By running ``main.py``, 7 buses, 6 transmission lines,
and 2 transformers are created.

For milestone 2, ``main.py`` also creates an instance of the PowerSystem class, and calls
its method to calculate the y-bus matrix. See the writeup in the writeups folder
for more information.

For milestone 3, ``main.py`` calls the methods needed to run the Newton-Raphson algorithm.
For the first part of milestone 3, the vectors for delta_y (power mismatch) and delta_x
(voltage angle and magnitude) as well as the four quadrants of the Jacobian are printed to the console.

For milestone 4, ``main.py`` the user is now provided a choice to run either a power flow study or a 
fault study. The power flow study is the same as milestone 3. The fault study allows the user to select a bus
to add the fault to and the pre-fault voltage. The faulted bus's current and the vector of bus voltages is output
from the program.
This includes the creation of the positive, negative, and zero sequence networks and the algorithm for calculating
phase voltage and current for any type of fault (line to line, single line to ground, double line to ground, and symmetric)
