import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

INTEGRAL_CSV = r"Data\Am241-050626_output\pulse_integrals.csv"
data = pl.read_csv(INTEGRAL_CSV)

Am_energies = [5.63782]

arbitrary_energy = -1 * data["pulse_integral"].to_numpy()

binsize = int(np.sqrt(len(arbitrary_energy)))
counts, bins = np.histogram(arbitrary_energy, bins=binsize, range=(0, 4e-5))
plt.stairs(counts, bins, fill=True)
x = (bins[:-1] + bins[1:]) / 2
plt.plot(x, counts, label='Data', drawstyle='steps-mid')
plt.xlabel('Pulse Integral (Arbitrary Units)')
plt.ylabel('Counts')
plt.show()
def gaussian(x, A1, mu1, sigma1):
    Gaussian1 = A1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2))
    return Gaussian1

init_guess = [1000, 2.2e-5, 1e-6] 
bounds = (
    [0, 2e-5, 0],       # Lower bounds
    [2000, 2.5e-5, 0.01]  # Upper bounds
)
popt, pcov = curve_fit(gaussian, x, counts, p0=init_guess, bounds=bounds)
plt.stairs(counts, bins, fill=True, label='Data')
x_fit = np.linspace(bins[0], bins[-1], 1000)
plt.plot(x_fit, gaussian(x_fit, *popt), color='red', label='Gaussian Fit')
plt.xlabel('Pulse Integral (Arbitrary Units)')
plt.ylabel('Counts')
plt.title('Energy Spectrum with Gaussian Fit')
plt.legend()
plt.grid(True)
plt.show()