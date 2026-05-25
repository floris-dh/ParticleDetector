# import time
# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt

# i = 1
# data = pd.read_csv(f"RawData/acq{i:04d}.csv", skiprows=10, header=None)

# for i in range(1, 100):
#     filepath = f"RawData/acq{i:04d}.csv"
#     if os.path.exists(filepath):
#         try:
#             data = pd.read_csv(filepath, skiprows=10, header=None)
#             print(f"Loaded: {filepath}")
#             data.columns = ['times', 'voltage']  # Add a new column for cleaned voltage data
#             min_voltage = data['voltage'].min()

#             base_voltage = data['voltage'][:500].mean()
#             base_voltage_std = data['voltage'][:500].std()
#             # Calculate moving average directly with pandas

#             pulse_window = np.where(data['voltage'].rolling(window=15).mean() < (base_voltage - base_voltage_std), data['voltage'].rolling(window=15).mean(), np.nan)
#             pulse_times = np.where(data['voltage'].rolling(window=15).mean() < (base_voltage - base_voltage_std), data['times'], np.nan)

#             fig, ax = plt.subplots()
#             ax.plot(pulse_times, pulse_window, markersize=3, label='Detected Pulses')
#             ax.axhline(base_voltage, color='g', linestyle='--', label=f'Base Voltage: {base_voltage:.3e} V')
#             ax.axhline(min_voltage, color='r', linestyle='--', label=f'Min Voltage: {min_voltage:.3e} V')
#             ax.fill_between(pulse_times, base_voltage, pulse_window, interpolate=True, color='orange', alpha=0.5, label='Detected Pulses')
#             plt.show(block=True)

#             integral = np.trapezoid(pulse_times[~np.isnan(pulse_times)], pulse_window[~np.isnan(pulse_window)] - base_voltage)

#             print(f"Integral of detected pulses: {integral:.3e} V*s")
#         except Exception as e:
#             print(f"Error loading {filepath}: {e}")
#             time.sleep(0.1)  # Wait briefly before trying again
#     else:
#         print(f"File not found: {filepath}")
#         time.sleep(0.1)  # Wait briefly before checking for the next file
        
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# 1. Define the Pulse Shape Model (Option A)
def pulse_model(t, t0, A, tau_decay, tau_rise, A_undershoot, tau_undershoot, baseline):
    dt = t - t0
    
    with np.errstate(divide='ignore', invalid='ignore'):
        # 1. Standard fast physical pulse shape
        pulse = A * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise))
        
        # 2. Add the hardware undershoot term (slowly pulls the signal positive)
        undershoot = A_undershoot * (np.exp(-dt / tau_undershoot) - np.exp(-dt / tau_decay))
        
    # Combine the pulse and undershoot shapes
    total_shape = pulse - undershoot
    
    # Apply Heaviside condition (exactly 0 before t0)
    total_shape = np.where(dt >= 0, total_shape, 0.0)
    
    # Invert and apply baseline
    return -total_shape + baseline

# 2. Load and Prepare Data
data = pd.read_csv("RawData/acq0001.csv", skiprows=10, header=None)
data.columns = ['times', 'voltage']

t = data['times'].values
v = data['voltage'].values
baseline = v[:500].mean()  # Estimate baseline from the first 500 samples

# 3. Provide Initial Guesses for the Fit [t0, Amplitude, tau_decay, tau_rise, baseline]
# Looking at your graph: t0 ~ 0.0000, Amp ~ 0.04, tau_decay ~ 0.0003, tau_rise ~ 0.0001, baseline ~ 0.0
# Initial guess order: [t0, A, tau_decay, tau_rise, A_undershoot, tau_undershoot, baseline]
initial_guesses = [-0.00005, 0.035, 0.0004, 0.00005, 0.015, 0.002, 0.0]

try:
    # 4. Perform the Fit
    popt, pcov = curve_fit(pulse_model, t, v, p0=initial_guesses, maxfev=5000)
    t0_fit, A_fit, tau_decay_fit, tau_rise_fit, baseline_fit = popt
    
    # Generate the fitted curve
    v_fit = pulse_model(t, *popt)
    
    # 5. Calculate Residuals (Error) to determine if it's a "good" pulse
    residuals = v - v_fit
    chi_squared = np.sum(residuals**2) / (len(v) - len(popt))
    
    print(f"Fit Successful!")
    print(f"Chi-Squared Error: {chi_squared:.3e}")
    print(f"Calculated Decay Constant (tau_decay): {tau_decay_fit:.6f} s")
    print(f"Calculated Rise Constant (tau_rise): {tau_rise_fit:.6f} s")

    # 6. Plot the Results
    plt.figure(figsize=(10, 6))
    plt.plot(t, v, label='Raw Data', alpha=0.6)
    plt.plot(t, v_fit, label='Mathematical Fit', color='red', linewidth=2)
    plt.title(f"Pulse Fit (Reduced $\chi^2$: {chi_squared:.2e})")
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.show()

except RuntimeError:
    print("Fit failed! The shape is too distorted or initial guesses were off.")
    