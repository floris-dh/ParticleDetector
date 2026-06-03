import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

INTEGRAL_CSV = r"Data\Ra226-290526-2\integrals.csv"

Pu_energies = [4.87062, 5.59031, 6.11468, 7.83346]

data = pl.read_csv(INTEGRAL_CSV)
data.head(5)

arbitrary_energy = data["integral_Vs"].to_numpy()

binsize = int(np.sqrt(len(arbitrary_energy)))
counts, bins = np.histogram(arbitrary_energy, bins=binsize)#, range=(2e-5, 4e-5))
plt.stairs(counts, bins, fill=True)

x = (bins[:-1] + bins[1:]) / 2

def quadruple_gaussian(x, A1, mu1, sigma1, A2, mu2, sigma2, A3, mu3, sigma3, A4, mu4, sigma4):
    Gaussian1 = A1 * np.exp(-(x - mu1) ** 2 / (2 * sigma1 ** 2))
    Gaussian2 = A2 * np.exp(-(x - mu2) ** 2 / (2 * sigma2 ** 2))
    Gaussian3 = A3 * np.exp(-(x - mu3) ** 2 / (2 * sigma3 ** 2))
    Gaussian4 = A4 * np.exp(-(x - mu4) ** 2 / (2 * sigma4 ** 2))
    return Gaussian1 + Gaussian2 + Gaussian3 + Gaussian4

init_guess = [
    300, 2e-5, 1e-6,  # Peak 1 Guess
    500, 2.4e-5, 1e-6,   # Background baseline noise guess
    320, 3.5e-5, 1e-6, # Peak 3 Guess
    400, 5e-5, 1e-6    # Peak 4 Guess
]

bounds = (
    [0, 2.1e-5, 0, 0, 2e-5, 0, 0, 3.5e-5, 0, 0, 4.5e-5, 0],       # Lower bounds
    [2000, 2.e-5, 0.01, 2000, 3e-5, 0.01, 2000, 4.5e-5, 0.01, 2000, 5.5e-5, 0.01]  # Upper bounds
)

popt, pcov = curve_fit(quadruple_gaussian, x, counts, p0=init_guess, bounds=bounds)

plt.stairs(counts, bins, fill=True, label='Data')
x_fit = np.linspace(bins[0], bins[-1], 1000)
plt.plot(x_fit, quadruple_gaussian(x_fit, *popt), color='red', label='Quadruple Gaussian Fit')
plt.xlabel('Pulse Integral (Arbitrary Units)')
plt.ylabel('Counts')
plt.title('Energy Spectrum with Quadruple Gaussian Fit')
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

for i in [1, 4, 7, 10]:  # Indices of mu1, mu2, mu3, mu4
    popt_mev[i] = func(popt[i], *popt2)  # Scale the mean values to MeV

for i in [2, 5, 8, 11]:  # Indices of sigma1, sigma2, sigma3, sigma4
    popt_mev[i] = popt[i] * a_slope  # Scale the sigma values to MeV using the slope from the linear fit

# 5. Plot the Gaussian using the newly converted MeV parameters
x_fit_mev = np.linspace(calibrated_x[0], calibrated_x[-1], 1000)
plt.plot(x_fit_mev, quadruple_gaussian(x_fit_mev, *popt_mev), color='red', linewidth=2, label='Quadruple Gaussian Fit (MeV)', alpha=0.4)

plt.xlabel('Energy (MeV)')
plt.ylabel('Counts')
plt.xlim(3, 9)
plt.ylim(0, max(counts) * 1.1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim(4,6)


FWHM_1 = 2.355 * popt_mev[2]  # FWHM for the first peak in MeV
FWHM_2 = 2.355 * popt_mev[5]  # FWHM for the second peak in MeV
FWHM_3 = 2.355 * popt_mev[8]  # FWHM for the third peak in MeV
FWHM_4 = 2.355 * popt_mev[11] # FWHM for the fourth peak in MeV

print(f"Fitted Peak 1 (MeV): Amplitude={popt_mev[0]:.2f}, Mean={popt_mev[1]:.2f} MeV, Sigma={popt_mev[2]:.2f} MeV, FWHM={FWHM_1:.2f} MeV")
print(f"Fitted Peak 2 (MeV): Amplitude={popt_mev[3]:.2f}, Mean={popt_mev[4]:.2f} MeV, Sigma={popt_mev[5]:.2f} MeV, FWHM={FWHM_2:.2f} MeV")
print(f"Fitted Peak 3 (MeV): Amplitude={popt_mev[6]:.2f}, Mean={popt_mev[7]:.2f} MeV, Sigma={popt_mev[8]:.2f} MeV, FWHM={FWHM_3:.2f} MeV")
print(f"Fitted Peak 4 (MeV): Amplitude={popt_mev[9]:.2f}, Mean={popt_mev[10]:.2f} MeV, Sigma={popt_mev[11]:.2f} MeV, FWHM={FWHM_4:.2f} MeV")

# 6. Save and render
plt.savefig(f"Figures/E_Spectrum_Calibrated.pdf")
plt.show()


