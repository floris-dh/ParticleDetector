import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

MEASUREMENT_ID = "Ra226-290526-1"

INTEGRAL_CSV = f"Data/{MEASUREMENT_ID}/integrals.csv"

data = pl.read_csv(INTEGRAL_CSV)

arbitrary_energy = data["integral_Vs"].to_numpy()

binsize = np.sqrt(len(arbitrary_energy))

counts, bins = np.histogram(arbitrary_energy, bins=300)
x = (bins[:-1] + bins[1:]) / 2

peaks, _ = find_peaks(counts, prominence=100)

def func(x, a, b):
    return a * x + b

x_array = np.linspace(2e-5, 4e-5, len(counts))

popt, pcov = curve_fit(func, x[peaks], [4.87, 5.59, 6.11, 7.89])

print("Calibration parameters:", popt)

# 1. Transform your actual histogram bin centers into real MeV units
calibrated_x = func(x, *popt)

# 2. Plot the spectrum with energy on the X-axis and counts on the Y-axis
plt.plot(calibrated_x, counts, drawstyle="steps-mid", color='blue', label='Energy Spectrum')

# 3. Plot your 4 calibration points using the corrected 'best_peaks' array
calibrated_peaks = func(x[peaks], *popt)

# 4. Dynamic annotations that use your calibrated peak positions
energies = [4.87, 5.59, 6.11, 7.89]
for idx, energy in enumerate(energies):
    plt.annotate(
        f'{energy} MeV', 
        (calibrated_peaks[idx], counts[peaks[idx]]), 
        textcoords="offset points", 
        xytext=(30, 10), 
        ha='center',
        weight='bold'
    )
    plt.axvline(energy, color='red', linestyle='--', alpha=0.7)

# 5. Graph formatting
plt.title(f'Energy Spectrum {MEASUREMENT_ID}')
plt.xlabel('Energy (MeV)')
plt.ylabel('Counts')
plt.xlim(3, 9)
plt.ylim(0, max(counts) * 1.1)
plt.legend()
plt.grid(True, alpha=0.3)

# 6. Save and render
plt.savefig(f"Figures/E_Spectrum_Calibrated_{MEASUREMENT_ID}.pdf")
plt.show()
