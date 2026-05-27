import numpy as np
from scipy.optimize import fsolve

# Your parameters
t0 = 5.0
A = 10.0
B = 3.0
tau_1 = 5.0
tau_2 = 1.0
tau_3 = 15.0

# Define the equation as a function that equals 0 at the root
def equation(t):
    dt = t - t0
    return -(A + B) * np.exp(-dt / tau_1) + A * np.exp(-dt / tau_2) + B * np.exp(-dt / tau_3)

# Use fsolve with an initial guess (e.g., t = 10.0)
root = fsolve(equation, 10.0)

print(f"The signal crosses the baseline at t = {root[0]:.4f}")