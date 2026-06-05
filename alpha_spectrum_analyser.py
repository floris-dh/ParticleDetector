import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

class AlphaSpectrumAnalyser:
    def __init__(self, csv_path, isotope_name):
        """
        Init analyser for alpha spectrum data.
        """
        
        isotope_data = {"Am241": ([5.63782], (1e-5, 5.5e-5)),
                    "Pu239" : ([4.67826, 5.24451], (2.5e-5, 4.0e-5)),
                    "Ra226": ([4.87062, 5.59031, 6.11468, 7.83346], (1.4e-5, 5.5e-5))}
        
        
        self.csv_path = csv_path
        self.isotope_name = isotope_name
        self.lit_energies = isotope_data[isotope_name][0]
        self.integral_range = isotope_data[isotope_name][1]
        self.num_peaks = len(self.lit_energies)
        
        # Data storage structures
        self.arbitrary_energy = None
        self.counts = None
        self.sigma_counts = None
        self.bins = None
        self.x = None
        self.threshold = 0.0
        
        # Fit results structures
        self.popt = None
        self.pcov = None
        self.chi2red = None
        self.calib_popt = None  # [a, b] voor a*x + b
        self.popt_mev = None

    def load_and_bin_data(self):
        """Laadt de integralen en maakt het initiële histogram."""
        print(f"[{self.isotope_name}] Data laden en histogram berekenen...")
        df = pl.read_csv(self.csv_path)
        self.arbitrary_energy = df["pulse_integral"].to_numpy()
        
        binsize = int(np.sqrt(len(self.arbitrary_energy)))
        counts, bins = np.histogram(self.arbitrary_energy, bins=binsize, range=self.integral_range)
        
        # compute using moving average
        self.counts, self.sigma_counts = self._moving_average(counts, window_size=4)
        self.bins = bins
        self.x = (bins[:-1] + bins[1:]) / 2
        self.threshold = 0.2 * max(self.counts)

    def _moving_average(self, data, window_size=5):
        """Internal helper for smoothing the histogram data."""
        weights = np.ones(window_size) / window_size
        smoothed_counts = np.convolve(data, weights, mode='full')[:len(data)]
        variance_smoothed = np.convolve(data, weights**2, mode='full')[:len(data)]
        return smoothed_counts, np.sqrt(variance_smoothed)

    def _multi_gaussian_model(self, x, *params):
        """Dynamic Gaussian based on N peaks"""
        y = np.zeros_like(x, dtype=float)
        for i in range(self.num_peaks):
            amp = params[i * 3]
            mu = params[i * 3 + 1]
            sigma = params[i * 3 + 2]
            y += amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        return y

    def fit_spectrum(self):
        """Vindt pieken via find_peaks en voert de multi-Gauss fit uit in Arbitrary Units."""
        print(f"[{self.isotope_name}] Automatische piekdetectie en Gaussische curve-fit...")
        
        # Find initial peaks using find_peaks with prominence
        peaks, _ = find_peaks(self.counts, prominence=25)
        
        if len(peaks) != self.num_peaks:
            raise ValueError(f"Not correct amount of peaks found for {self.isotope_name}. Detected: {len(peaks)}, Required: {self.num_peaks}")

        # Build Dynamic initial guess and bounds based on detected peaks
        init_guess = []
        lower_bounds = []
        upper_bounds = []
        
        for i in range(self.num_peaks):
            p_idx = peaks[i]
            amp_g = self.counts[p_idx]
            mu_g = self.x[p_idx]
            sig_g = 1e-6
            
            init_guess.extend([amp_g, mu_g, sig_g])
            lower_bounds.extend([0, mu_g * 0.8, 0])
            upper_bounds.extend([amp_g * 1.5, mu_g * 1.2, 0.01])

        mask = self.counts > self.threshold

        # Fit to the data using the multi-Gaussian model
        popt, pcov, infodict, mesg, ier = curve_fit(
            self._multi_gaussian_model, self.x[mask], self.counts[mask], 
            p0=init_guess, bounds=(lower_bounds, upper_bounds), full_output=True
        )
        
        self.popt = popt
        self.pcov = pcov
        self.chi2red = np.sum(infodict['fvec']**2) / (len(self.x[mask]) - len(popt))
        print(f"  Gereduceerde Chi2 in AU: {self.chi2red:.2e}")

    def calibrate_and_scale(self):
        """Voert de lineaire energie-kalibratie uit (AU -> MeV)."""
        
        if self.num_peaks < 2:
            print(f"[{self.isotope_name}] Not enough Peaks for Calibration. Skipping...")
            return None
        
        print(f"[{self.isotope_name}] Kalibratie naar MeV uitvoeren...")
        
        # Extract fitted means (mu) for each peak
        fitted_means = np.array([self.popt[i * 3 + 1] for i in range(self.num_peaks)])
        
        # Linear fit to calibrate x-axis from AU to MeV
        def linear_func(x, a, b): return a * x + b
        
        calib_popt, _ = curve_fit(linear_func, fitted_means, self.lit_energies, p0=[1e5, 0])
        self.calib_popt = calib_popt
        a_slope, b_intercept = calib_popt
        
        # Scale the fitted parameters to MeV for plotting and analysis
        self.popt_mev = self.popt.copy()
        for i in range(self.num_peaks):
            idx = i * 3
            # Amplitudes (idx) blijven gelijk
            self.popt_mev[idx + 1] = linear_func(self.popt[idx + 1], a_slope, b_intercept) # mu naar MeV
            self.popt_mev[idx + 2] = self.popt[idx + 2] * a_slope                         # sigma naar MeV
            
            # Print statistics for each peak
            fwhm = 2.355 * self.popt_mev[idx + 2]
            print(f"  Piek {i+1} ({self.lit_energies[i]} MeV Ref) -> Gekalibreerd: {self.popt_mev[idx+1]:.3f} MeV | FWHM: {fwhm:.3f} MeV")

    def generate_plots(self, output_folder="Figures"):
        """Genereert en slaat de definitieve MeV-gekalibreerde grafiek op."""
        os.makedirs(output_folder, exist_ok=True)
        plt.figure(figsize=(8, 5.5))
        
        def linear_func(x, a, b): return a * x + b
        
        if self.calib_popt is not None:
            calibrated_x = linear_func(self.x, *self.calib_popt)
            
            # 1. Plot energiespectrum
            plt.plot(calibrated_x, self.counts, drawstyle="steps-mid", color='black', label='Energy Spectrum Data')
            
            # 2. Plot de totale multi-Gauss fit in MeV
            x_fit_mev = np.linspace(calibrated_x[0], calibrated_x[-1], 1000)
            y_fit_mev = self._multi_gaussian_model(x_fit_mev, *self.popt_mev)
            # Maskeer waarden onder de threshold voor een clean plotbeeld
            y_fit_mev_masked = np.where(y_fit_mev > self.threshold, y_fit_mev, np.nan)
            
            plt.plot(x_fit_mev, y_fit_mev_masked, color='red', linewidth=2, label=f'Multi-Gaussian Fit ($\chi^2_{{red}}$ AU: {self.chi2red:.1e})')
            plt.axhline(self.threshold, color='orange', linestyle='--', alpha=0.4, label=f'Threshold ({self.threshold:.1f} counts)')
            
            # 3. Annotaties en verticale indicatielijnen plaatsen
            for i in range(self.num_peaks):
                amp_mev = self.popt_mev[i * 3]
                mu_mev = self.popt_mev[i * 3 + 1]
                lit_e = self.lit_energies[i]
                
                plt.axvline(lit_e, color='blue', linestyle=':', alpha=0.5)
                plt.axvline(mu_mev, color='red', linestyle='--', alpha=0.7)
                
                plt.annotate(
                    f'{lit_e:.2f} MeV', 
                    (mu_mev, amp_mev), 
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    weight='bold'
                )
            # Grafiekopmaak
            plt.xlabel('Energy (MeV)')
            plt.ylabel('Counts')
            plt.title(f'Calibrated Alpha Energy Spectrum - {self.isotope_name}')
            plt.ylim(0, max(self.counts) * 1.2)
            plt.xlim(min(calibrated_x), max(calibrated_x))
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            
        elif self.calib_popt is None:
            print(f"[{self.isotope_name}] Calibration not available. Plotting in Arbitrary Units (AU) instead.")
            plt.plot(self.x, self.counts, drawstyle="steps-mid", color='black', label='Energy Spectrum Data (AU)')
            plt.plot(self.x, self._multi_gaussian_model(self.x, *self.popt), color='red', linewidth=2, label=f'Multi-Gaussian Fit (AU)')
            plt.axhline(self.threshold, color='orange', linestyle='--', alpha=0.4, label=f'Threshold ({self.threshold:.1f} counts)')
            plt.xlabel('Arbitrary Units (AU)')
            plt.ylabel('Counts')
            plt.title(f'Alpha Energy Spectrum - {self.isotope_name} (Uncalibrated)')
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(output_folder, f"E_Spectrum_{self.isotope_name}_Calibrated.pdf")
        plt.savefig(save_path)
        print(f"[{self.isotope_name}] Grafiek succesvol opgeslagen naar '{save_path}'")
        plt.show()

    def run_full_analysis(self):
        """Voert de volledige analysepijplijn achter elkaar uit."""
        self.load_and_bin_data()
        self.fit_spectrum()
        self.calibrate_and_scale()
        self.generate_plots()


# =============================================================================
# HOOFDPROGRAMMA: CONFIGURATIE & UITVOERING
# =============================================================================
if __name__ == "__main__":
    
    
    # VOORBEELD: Draai de analyse voor de gevraagde Ra226 bron
    target_source = "Ra226" 
    cfg = r"Data\Ra226-030626-1_output\pulse_integrals.csv"
    
    analyser = AlphaSpectrumAnalyser(
        csv_path=cfg,
        isotope_name=target_source,
    )
    
    analyser.run_full_analysis()