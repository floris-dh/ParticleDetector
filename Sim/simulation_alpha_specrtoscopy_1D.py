import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import time
E_start = float(input("Enter the starting energy (in MeV): "))
distance = float(input("Distance from the source to the detector (in mm): "))
n_particles = int(1e5)
step_size = 0.05 #in mm
start_time = time.time()

base_dir = os.path.dirname(__file__)
file_path = os.path.join(base_dir, 'Data', 'NIST_data.csv') # of .txt

df = pd.read_csv(file_path, header=None, sep=r'\s+', names=['Energy', 'MassStpPow'])
df_filtered = df[(df['Energy'] >= 2) & (df['Energy'] <= 6.0)].copy()
x_data = df_filtered['Energy'].values
y_data = df_filtered['MassStpPow'].values

#omrekenen MeV/mm
density_air = 0.001225 # g/cm^3
y_mm = y_data * density_air * 0.1 # g/cm^3 * MeV
def power_law(E, a, b):
    return a * E**b
#_ is de prullenbak variabele
popt, _ = curve_fit(power_law, df_filtered['Energy'], y_mm)
a, b = popt

print(f"Constant a = {a:.6f}")
print(f"Exponent b = {b:.6f}")

def simulation(E_start, distance, n_particles, step_size= 0.01):
    E = E_start
    x = 0
    stopping_power = a * (E**b)
    while E > 0 and x < distance:
        dE = stopping_power * step_size + np.random.normal(0, 0.01 * stopping_power) # adding noise
        E -= dE
        x += step_size
    return max(E, 0) #Cool way to make sure we don't return negative energies

energies_at_detector = []
for i in range(n_particles):
    # fire each particle sequentially
    result_E = simulation(E_start, distance, n_particles, step_size)
    energies_at_detector.append(result_E)


plt.hist(energies_at_detector, bins=int(np.sqrt(len(energies_at_detector))), color='blue', edgecolor='black')
plt.title(f"Recorded Spectrum ({distance} mm air)")
plt.xlabel("Energy at detector [MeV]")
plt.ylabel(f"n_particles {n_particles}")


end_time = time.time()
print(f"{end_time - start_time:.2f} seconds.")
plt.show()


