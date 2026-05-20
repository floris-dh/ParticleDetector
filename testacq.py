import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

i = 1 
filepath = f"RawData/acq{i:04d}.csv"

# Keep the dictionary structure
pulses = {
    "voltage": [],
    "time": [],
    "max": [],
    "energy": []
}

def process_single_file(file_path):
    data = pd.read_csv(file_path, skiprows=9)
    times = data.iloc[:, 0]
    voltage = data.iloc[:, 1]
    return times, voltage

while os.path.exists(filepath):    
    time, voltage = process_single_file(filepath)
    max_voltage = max(voltage)
    
    # Check threshold
    if max_voltage > 0.4:
        energy = np.trapezoid(voltage, time)  # Calculate energy using trapezoidal rule
        # Use .append() instead of index assignment
        pulses["voltage"].append(voltage)
        pulses["time"].append(time)
        pulses["max"].append(max_voltage)
        pulses["energy"].append(energy)
        print(f"Max voltage {max_voltage} toegevoegd aan pulses.")
        
        # plt.plot(time, voltage)
        # plt.title(f"Voltage vs Time for {filepath}")
        # plt.xlabel("Time (s)")
        # plt.ylabel("Voltage (V)")
        # plt.show()
    else:
        print(f"discarded {filepath} with max voltage {max_voltage}")
        
    i += 1
    filepath = f"RawData/acq{i:04d}.csv"
    
print(f"Total pulses collected: {len(pulses['energy'])}")

counts, bins = np.histogram(pulses["energy"], bins=len(pulses["energy"]), density=True)
plt.stairs(counts, bins)
plt.title("Energy Distribution of Pulses")
plt.xlabel("Energy (J)")
plt.ylabel("Density")
plt.show()