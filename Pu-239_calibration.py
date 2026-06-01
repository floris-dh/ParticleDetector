import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

INTEGRAL_CSV = f"Data\Pu-239_010626_output\pulse_integrals.csv"

Pu_energies = [4.67826, 5.24451]

data = pl.read_csv(INTEGRAL_CSV)

arbitrary_energy = -1 * data["pulse_integral"].to_numpy()

binsize = int(np.sqrt(len(arbitrary_energy)))
counts, bins = np.histogram(arbitrary_energy, bins=binsize, range=(2e-5, 4e-5))
plt.stairs(counts, bins, fill=True)



x = (bins[:-1] + bins[1:]) / 2

def double_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2):
    Gaussian1 = A1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2))
    Gaussian2 = A2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
    return Gaussian1 + Gaussian2

init_guess = [
    1050, 2.7e-5, 5e-6,  # Peak 1 Guess
    1400, 3.4e-5, 1e-6   # Background baseline noise guess
]

bounds = (
    [0, 2e-5, 0, 0, 3e-5, 0],       # Lower bounds
    [2000, 3e-5, 0.01, 2000, 4e-5, 0.01]  # Upper bounds
)

popt, pcov = curve_fit(double_gaussian, x, counts, p0=init_guess, bounds=bounds)

plt.stairs(counts, bins, fill=True, label='Data')
x_fit = np.linspace(bins[0], bins[-1], 1000)
plt.plot(x_fit, double_gaussian(x_fit, *popt), color='red', label='Double Gaussian Fit')
plt.xlabel('Pulse Integral (Arbitrary Units)')
plt.ylabel('Counts')
plt.title('Energy Spectrum with Double Gaussian Fit')
plt.legend()
plt.show()
    

def func(x, a, b):
    return a * x + b

peaks = np.array([popt[1], popt[4]])  # Extract the means of the two fitted Gaussians as peak positions
print("Identified peak indices:", peaks)

popt2, pcov = curve_fit(func, peaks, Pu_energies, p0=[1e5, 0], maxfev=10000)
print("Calibration parameters:", popt2)

plt.figure()
calibrated_x = func(x, *popt2)
plt.plot(calibrated_x, counts, drawstyle="steps-mid", color='blue', label='Energy Spectrum')

calibrated_peaks = func(peaks, *popt2)
peak_heights = [popt[0], popt[3]]  # Amplitudes of the two fitted Gaussians

# annotations that use calibrated peak positions
energies = Pu_energies
for idx, energy in enumerate(energies):
    plt.annotate(
        f'{energy} MeV', 
        (calibrated_peaks[idx], peak_heights[idx]), 
        textcoords="offset points", 
        xytext=(30, 10), 
        ha='center',
        weight='bold'
    )
    plt.axvline(energy, color='red', linestyle='--', alpha=0.7)

a_slope = popt2[0]
popt_mev = popt.copy()

# scale mu1 and mu2
popt_mev[1] = func(popt[1], *popt2) # mu1
popt_mev[4] = func(popt[4], *popt2) # mu2

# scale sigma1 and sigma2 
popt_mev[2] = popt[2] * a_slope     # sigma1
popt_mev[5] = popt[5] * a_slope     # sigma2

# 5. Plot the Gaussian using the newly converted MeV parameters
x_fit_mev = np.linspace(calibrated_x[0], calibrated_x[-1], 1000)
plt.plot(x_fit_mev, double_gaussian(x_fit_mev, *popt_mev), color='red', linewidth=2, label='Double Gaussian Fit (MeV)', alpha=0.4)

plt.xlabel('Energy (MeV)')
plt.ylabel('Counts')
plt.xlim(3, 9)
plt.ylim(0, max(counts) * 1.1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(4,6)

FWHM_1 = 2.355 * popt_mev[2]  # FWHM for the first peak in MeV
FWHM_2 = 2.355 * popt_mev[5]  # FWHM for the second peak in MeV
print(f"Fitted Peak 1 (MeV): Amplitude={popt_mev[0]:.2f}, Mean={popt_mev[1]:.2f} MeV, Sigma={popt_mev[2]:.2f} MeV, FWHM={FWHM_1:.2f} MeV")
print(f"Fitted Peak 2 (MeV): Amplitude={popt_mev[3]:.2f}, Mean={popt_mev[4]:.2f} MeV, Sigma={popt_mev[5]:.2f} MeV, FWHM={FWHM_2:.2f} MeV")

# 6. Save and render
plt.savefig(f"Figures/E_Spectrum_Calibrated.pdf")
plt.show()


