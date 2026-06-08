import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

class AlphaSpectrumAnalyser:
    """
    Class to analyse alpha energy spectra from pulse integral data.
    
    INPUTS:
    - csv_path: Path to the CSV file containing pulse integrals.
    - isotope_name: Name of the isotope (e.g., "Am241", "Pu239", "Ra226") to determine expected peak energies and integral ranges.
    
    This class performs the following steps:
    1. Loads the pulse integral data and creates a histogram.
    2. Automatically detects peaks and fits a multi-Gaussian model to the spectrum in arbitrary units (AU).
    3. Calibrates the x-axis from AU to MeV using a linear fit based on the known energies of the detected peaks.
    4. Generates a final plot of the calibrated energy spectrum with annotated peaks and saves it as a PDF.
    """
    def __init__(self, csv_path, isotope_name):

        
        isotope_data = {"Am241": ([5.63782], (1e-5, 5.5e-5)),
                    "Pu239" : ([4.67826, 5.24451], (2.5e-5, 4.0e-5)),
                    "Ra226": ([4.87062, 5.59031, 6.11468, 7.83346], (1.4e-5, 6.5e-5))}
        
        
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
        """
        Function to load the pulse integral data from CSV and create a histogram in arbitrary units (AU).
        
        Following Steps:
        1. Load the pulse integral data using Polars.
        2. Create a histogram of the pulse integrals with an appropriate binning strategy (e.g., sqrt(N) bins).
        3. Apply a moving average to smooth the histogram counts and compute the corresponding sigma for error bars.
        4. Store the histogram data and set a threshold for peak detection based on the maximum count.
        """
        print(f"[{self.isotope_name}] Loading and binning data...")
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
        """
        Function to apply a moving average to the histogram counts and compute the corresponding sigma for error bars.
        
        INPUTS:
        - data: The raw histogram counts.
        - window_size: The size of the moving average window (default is 5).
        
        OUTPUTS:
        - smoothed_counts: The counts after applying the moving average.
        - sigma_counts: The corresponding sigma values for the smoothed counts, computed as the square root of the variance of the smoothed data.
        
        METHOD:
        1. Create a uniform kernel of the specified window size for the moving average.
        2. Convolve the raw counts with the kernel to obtain the smoothed counts.
        3. Convolve the raw counts with the squared kernel to compute the variance of the smoothed counts, and take the square root to get sigma.
        """
        weights = np.ones(window_size) / window_size
        smoothed_counts = np.convolve(data, weights, mode='full')[:len(data)]
        variance_smoothed = np.convolve(data, weights**2, mode='full')[:len(data)]
        return smoothed_counts, np.sqrt(variance_smoothed)

    def _multi_gaussian_model(self, x, *params):
        """
        Function to define the multi-Gaussian model for fitting the energy spectrum.
        
        INPUTS:
        - x: The independent variable (energy in AU).
        - params: A variable-length list of parameters for the multi-Gaussian model, where each peak is defined by three parameters (amplitude, mean, sigma).
        
        OUTPUTS:
        - y: The computed values of the multi-Gaussian model at the given x values.
        
        METHOD:
        1. Initialize an array of zeros for the output y values.
        2. Loop over the number of peaks, extracting the amplitude, mean, and sigma for each peak from the params list.
        3. For each peak, compute the Gaussian function and add it to the output y values.
        """
        y = np.zeros_like(x, dtype=float)
        for i in range(self.num_peaks):
            amp = params[i * 3]
            mu = params[i * 3 + 1]
            sigma = params[i * 3 + 2]
            y += amp * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
        return y

    def fit_spectrum(self):
        """
        Function to perform automatic peak detection and fit the energy spectrum with a multi-Gaussian model in arbitrary units (AU).
        
        OUTPUTS:
        - popt: The optimal parameters for the multi-Gaussian fit.
        - pcov: The covariance of the fit parameters.
        - chi2red: The reduced chi-squared value of the fit in arbitrary units (AU).
        
        METHOD:
        1. Use the find_peaks function from scipy.signal to detect initial peak positions in the histogram counts, using a prominence threshold to ensure significant peaks are detected.
        2. If the number of detected peaks does not match the expected number of peaks for the isotope, raise a ValueError.
        3. Build the initial guess and bounds for the multi-Gaussian fit parameters based on the detected peaks, where each peak is defined by its amplitude, mean, and sigma.
        4. Create a mask to select only the data points above the defined threshold for fitting.
        5. Use the curve_fit function from scipy.optimize to fit the multi-Gaussian model to the data, providing the initial guess and bounds for the parameters, and compute the reduced chi-squared value for the fit in arbitrary units (AU).
        6. Compute the reduced chi-squared value using the residuals from the fit and the number of degrees of freedom (number of data points minus number of fit parameters).
        """
        
        print(f"[{self.isotope_name}] Automatic peak detection and Gaussian curve fit...")
        
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
        print(f"  Reduced Chi2 (AU): {self.chi2red:.2e}")

    def calibrate_and_scale(self):
        """
        Function to calibrate the x-axis from arbitrary units (AU) to MeV using a linear fit based on the known energies of the detected peaks.
        
        OUTPUTS:
        - calib_popt: The optimal parameters for the linear calibration (slope and intercept).
        - popt_mev: The fitted parameters for the multi-Gaussian model scaled to MeV for plotting and analysis.
        
        METHOD:
        1. Check if the number of detected peaks is sufficient for calibration; if not, print a warning and skip the calibration step.
        2. Extract the fitted mean (mu) values for each detected peak from the multi-Gaussian fit parameters.
        3. Define a linear function for calibration and use the curve_fit function to fit this linear function to the fitted mean values (in AU) and the known literature energies (in MeV) of the peaks, obtaining the calibration parameters (slope and intercept).
        4. Scale the fitted parameters of the multi-Gaussian model to MeV using the obtained calibration parameters, where the mean values are converted to MeV using the linear function and the sigma values are scaled by the slope of the calibration.
        5. Print the calibration parameters and the calibrated peak positions and FWHM for each detected peak for verification.
        """
        
        if self.num_peaks < 2:
            print(f"[{self.isotope_name}] Not enough Peaks for Calibration. Skipping...")
            self.calib_popt = (1.06e5, 2.527)  # Fallback calibration parameters
        
        if self.num_peaks >= 2:
            print(f"[{self.isotope_name}] Performing calibration to MeV...")
            
            # Extract fitted means (mu) for each peak
            fitted_means = np.array([self.popt[i * 3 + 1] for i in range(self.num_peaks)])
            
            # Linear fit to calibrate x-axis from AU to MeV
            def linear_func(x, a, b): return a * x + b
            
            calib_popt, _ = curve_fit(linear_func, fitted_means, self.lit_energies, p0=[1e5, 0])
            self.calib_popt = calib_popt
            a_slope, b_intercept = calib_popt
            
            print(f"Calibration parameters: a (slope) = {a_slope:.2e} MeV/AU, b (intercept) = {b_intercept:.3f} MeV")
        
        def linear_func(x, a, b): return a * x + b
        
        # Scale the fitted parameters to MeV for plotting and analysis
        self.popt_mev = self.popt.copy()
        for i in range(self.num_peaks):
            idx = i * 3
            # Amplitudes (idx) blijven gelijk
            self.popt_mev[idx + 1] = linear_func(self.popt[idx + 1], *self.calib_popt) # mu naar MeV
            self.popt_mev[idx + 2] = self.popt[idx + 2] * self.calib_popt[0]             # sigma naar MeV
            
            # Print statistics for each peak
            fwhm = 2.355 * self.popt_mev[idx + 2]
            print(f"  Peak {i+1} ({self.lit_energies[i]} MeV ref) -> Calibrated: {self.popt_mev[idx+1]:.3f} MeV | FWHM: {fwhm:.3f} MeV")

    def generate_plots(self, output_folder="Figures"):
        """
        Function to generate a final plot of the calibrated energy spectrum with annotated peaks and save it as a PDF.
        
        OUTPUTS:
        - A PDF file containing the plot of the calibrated energy spectrum with annotated peaks, saved in the specified output folder.
        
        METHOD:
        1. Create the output folder if it does not exist.
        2. Set up the plot with appropriate size and labels.
        3. If calibration parameters are available, plot the energy spectrum in MeV and the multi-Gaussian fit, masking values below the threshold for a cleaner visualization.
        4. Annotate the detected peaks with their calibrated energies and add vertical lines for visual indication.
        5. If calibration is not available, plot the spectrum in arbitrary units (AU) and indicate that the fit is in AU.
        6. Add legends, grid, and title to the plot, and save it as a PDF file in the specified output folder with a filename that includes the isotope name and calibration status.
        """
        os.makedirs(output_folder, exist_ok=True)
        plt.figure(figsize=(8, 5.5))
        
        def linear_func(x, a, b): return a * x + b
            
        calibrated_x = linear_func(self.x, *self.calib_popt)
        
        # 1. Plot energiespectrum
        plt.plot(calibrated_x, self.counts, drawstyle="steps-mid", color='black', label='Energy Spectrum Data')
        
        # 2. Plot de totale multi-Gauss fit in MeV
        x_fit_mev = np.linspace(calibrated_x[0], calibrated_x[-1], 1000)
        y_fit_mev = self._multi_gaussian_model(x_fit_mev, *self.popt_mev)
        # Maskeer waarden onder de threshold voor een clean plotbeeld
        y_fit_mev_masked = np.where(y_fit_mev > self.threshold, y_fit_mev, np.nan)
        
        plt.plot(x_fit_mev, y_fit_mev_masked, color='red', linewidth=2, label=fr'Multi-Gaussian Fit ($\chi^2_{{red}}$ AU: {self.chi2red:.1e})', alpha=0.4)
        plt.axhline(self.threshold, color='orange', linestyle='--', alpha=0.4, label=f'Threshold ({self.threshold:.1f} counts)')
        
        # 3. Annotaties en verticale indicatielijnen plaatsen
        for i in range(self.num_peaks):
            amp_mev = self.popt_mev[i * 3]
            mu_mev = self.popt_mev[i * 3 + 1]
            lit_e = self.lit_energies[i]
            
            plt.annotate(
                f'{lit_e:.2f} MeV', 
                (mu_mev, amp_mev), 
                textcoords="offset points", 
                xytext=(0, 15), 
                ha='center',
                weight='bold'
            )
        # Grafiekopmaak
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Counts')
        plt.title(f'Calibrated Alpha Energy Spectrum - {self.isotope_name}')
        plt.ylim(0, max(self.counts) * 1.2)
        plt.xlim(min(calibrated_x), max(calibrated_x))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        save_path = os.path.join(output_folder, f"E_Spectrum_{self.isotope_name}_Calibrated.pdf")
        plt.savefig(save_path)
        print(f"[{self.isotope_name}] Plot successfully saved to '{save_path}'")
        plt.show()

    def run_full_analysis(self):
        """
        Runs the full analysis pipeline for the alpha energy spectrum, including data loading, peak fitting, calibration, and plotting.
        """
        self.load_and_bin_data()
        self.fit_spectrum()
        self.calibrate_and_scale()
        self.generate_plots()


# =============================================================================
# HOOFDPROGRAMMA: CONFIGURATIE & UITVOERING
# =============================================================================
if __name__ == "__main__":
    
    # target_source = "Ra226"  
    # cfg = r"Data\Ra226-290526-1\integrals.csv"
    
    # analysis = AlphaSpectrumAnalyser(
    #     csv_path=cfg,
    #     isotope_name=target_source
    # )
    
    # analysis.run_full_analysis()

    # target_source = "Pu239"  
    # cfg = r"Data\Pu239-050626-1_output\pulse_integrals.csv"
    
    # analysis = AlphaSpectrumAnalyser(
    #     csv_path=cfg,
    #     isotope_name=target_source
    #     )
    
    # analysis.run_full_analysis()
    
    target_source = "Am241"  
    cfg = r"Data\Am241-050626-1_output\pulse_integrals.csv"
    
    analysis = AlphaSpectrumAnalyser(
        csv_path=cfg,
        isotope_name=target_source
        )
    
    analysis.run_full_analysis()
    
    