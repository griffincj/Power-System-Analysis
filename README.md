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