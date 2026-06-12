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
                    "Ra226": ([4.87062, 5.59031, 6.11468, 7.83346], (1.4e-5, 10e-5))}
        
        
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
        self.calib_popt = None  # [a, b] for a*x + b
        self.popt_mev = None

    def _require_loaded(self):
        if self.x is None or self.counts is None or self.sigma_counts is None:
            raise RuntimeError("Spectrum data is not loaded yet. Call load_and_bin_data() first.")

    def _require_fit(self):
        if self.popt is None:
            raise RuntimeError("Spectrum has not been fit yet. Call fit_spectrum() first.")

    def _require_calibration(self):
        if self.calib_popt is None or self.popt_mev is None:
            raise RuntimeError("Spectrum has not been calibrated yet. Call calibrate_and_scale() first.")

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
        self.counts, self.sigma_counts = self._moving_average(counts, window_size=5)
        self.bins = bins
        self.x = (bins[:-1] + bins[1:]) / 2
        self.threshold = 0.2 * max(self.counts)

    def _moving_average(self, data, window_size=3):
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
        variance_smoothed = np.convolve(data **2, weights, mode='full')[:len(data)] - smoothed_counts ** 2
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
        self._require_loaded()
        assert self.x is not None and self.counts is not None and self.sigma_counts is not None
        x = self.x
        counts = self.counts
        sigma_counts = self.sigma_counts
        
        # Find initial peaks using find_peaks with prominence
        all_peaks, properties = find_peaks(counts, prominence=5)
        
        if len(all_peaks) != self.num_peaks:
            prominences = properties['prominences']
            top_indices = np.argsort(prominences)[-self.num_peaks:]
            peaks = all_peaks[top_indices]
        else:
            peaks = all_peaks
        
        # 3. CRITICAL: Sort left-to-right so peak index matching works chronologically
        peaks = np.sort(peaks)
        
        print(f"  Detected peaks at indices: {peaks} with energies (AU): {x[peaks]} and counts: {counts[peaks]}")

        # Build Dynamic initial guess and bounds based on detected peaks
        init_guess = []
        lower_bounds = []
        upper_bounds = []
        
        for i in range(self.num_peaks):
            p_idx = peaks[i]
            amp_g = counts[p_idx]
            mu_g = x[p_idx]
            sig_g = 1e-6
            
            init_guess.extend([amp_g, mu_g, sig_g])
            lower_bounds.extend([0, mu_g * 0.8, 0])
            upper_bounds.extend([amp_g * 1.5, mu_g * 1.2, 0.01])

        mask = counts > self.threshold

        # Fit to the data using the multi-Gaussian model
        popt, pcov, infodict, mesg, ier = curve_fit(
            self._multi_gaussian_model, x[mask], counts[mask], 
            p0=init_guess, bounds=(lower_bounds, upper_bounds), full_output=True, sigma=sigma_counts[mask], absolute_sigma=True
        )
        
        self.popt = popt
        self.pcov = pcov
        self.chi2red = np.sum(infodict['fvec']**2) / (len(x[mask]) - len(popt))
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
        
        self._require_fit()
        assert self.popt is not None
        popt = self.popt

        if self.num_peaks < 2:
            print(f"[{self.isotope_name}] Not enough Peaks for Calibration. Skipping...")
            self.calib_popt = (1.06e5, 2.527)  # Fallback calibration parameters
        
        if self.num_peaks >= 2:
            print(f"[{self.isotope_name}] Performing calibration to MeV...")
            
            # Extract fitted means (mu) for each peak
            fitted_means = np.array([popt[i * 3 + 1] for i in range(self.num_peaks)])
            
            # Linear fit to calibrate x-axis from AU to MeV
            def linear_func(x, a, b): return a * x + b
            
            calib_popt, _ = curve_fit(linear_func, fitted_means, self.lit_energies, p0=[1e5, 0])
            self.calib_popt = calib_popt
            a_slope, b_intercept = calib_popt
            
            print(f"Calibration parameters: a (slope) = {a_slope:.2e} MeV/AU, b (intercept) = {b_intercept:.3f} MeV")
        
        def linear_func(x, a, b): return a * x + b
        
        # Scale the fitted parameters to MeV for plotting and analysis
        assert self.calib_popt is not None
        calib_popt = self.calib_popt
        self.popt_mev = popt.copy()
        for i in range(self.num_peaks):
            idx = i * 3
            # Amplitudes (idx) blijven gelijk
            self.popt_mev[idx + 1] = linear_func(popt[idx + 1], *calib_popt) # mu naar MeV
            self.popt_mev[idx + 2] = popt[idx + 2] * calib_popt[0]             # sigma naar MeV
            
            # Print statistics for each peak
            fwhm = 2.355 * self.popt_mev[idx + 2]
            print(f"  Peak {i+1} ({self.lit_energies[i]} MeV ref) -> Calibrated: {self.popt_mev[idx+1]:.3f} MeV | FWHM: {fwhm:.3f} MeV")

    def generate_plots(self, ax, output_folder="Figures"):
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
        # plt.figure(figsize=(8, 5.5), facecolor='#B8B8AA')

        self._require_loaded()
        self._require_calibration()
        assert self.x is not None and self.counts is not None and self.sigma_counts is not None
        assert self.calib_popt is not None and self.popt_mev is not None
        x = self.x
        counts = self.counts
        sigma_counts = self.sigma_counts
        calib_popt = self.calib_popt
        popt_mev = self.popt_mev
        
        def linear_func(x, a, b): return a * x + b
            
        calibrated_x = linear_func(x, *calib_popt)
        
        # 1. Plot energy spectrum
        ax.plot(calibrated_x, counts, drawstyle="steps-mid", color='#9D593D', label='Data')
        
        # 2. Plot the total multi-Gauss fit in MeV
        x_fit_mev = np.linspace(calibrated_x[0], calibrated_x[-1], 1000)
        y_fit_mev = self._multi_gaussian_model(x_fit_mev, *popt_mev)
        
        # Mask values below threshold for a cleaner plot
        y_fit_mev_masked = np.where(y_fit_mev > self.threshold, y_fit_mev, np.nan)
        
        ax.plot(x_fit_mev, y_fit_mev_masked, color='#EFCD88', linewidth=1, linestyle='dotted', 
                label=fr'Gaussian Fit', alpha=0.8)
        
        ax.fill_between(calibrated_x, counts - 2 *sigma_counts, counts + 2 * sigma_counts, 
                        color='#9D593D', alpha=0.5, step='mid', label=r'Error (95% CI)')
        
        # 3. Annotations
        for i in range(self.num_peaks):
            mu_mev = popt_mev[i * 3 + 1]
            lit_e = self.lit_energies[i]
            
            ax.annotate(
                f'{lit_e:.2f} MeV', 
                xy=(mu_mev, 0),               
                xytext=(mu_mev, 0.2 * np.max(counts) + i * 0.05 * np.max(counts)),         
                textcoords="data",
                color='#BBC2C6', fontsize=9,          
                ha='center', va='bottom', weight='bold',
                arrowprops=dict(arrowstyle="->", color='#7F8550', lw=1.5, shrinkA=3, shrinkB=2)
            )

        # Axis formatting applied to the specific subplot axis
        ax.set_xlabel('Energy (MeV)', color='#BBC2C6', weight='bold', fontsize=12)
        ax.set_title(f'{self.isotope_name} Spectrum', color='#F5FDFF', weight='bold')
        ax.set_ylim(0, float(np.max(counts)) * 1.3)


    def run_full_analysis(self, ax):
        """
        Runs the full analysis pipeline for the alpha energy spectrum, including data loading, peak fitting, calibration, and plotting.
        """
        self.load_and_bin_data()
        self.fit_spectrum()
        self.calibrate_and_scale()
        self.generate_plots(ax)


# =============================================================================
# HOOFDPROGRAMMA: CONFIGURATIE & UITVOERING
# =============================================================================
if __name__ == "__main__":
    
    # Define your configuration dataset
    datasets = [
        {"source": "Am241", "path": r"Data\Am241-050626-1_output\pulse_integrals.csv"},
        {"source": "Pu239", "path": r"Data\Pu239-050626-1_output\pulse_integrals.csv"},
        {"source": "Ra226", "path": r"Data\Ra-080626-2_output\pulse_integrals.csv"} 
    ]
    
    num_plots = len(datasets)
    
    # Create a figure with N subplots in 1 row (change to nrows=num_plots for vertical stacking)
    fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(6 * num_plots, 5), sharey=False, facecolor='#182731')
    
    
    # Handle edge case if only 1 dataset is passed (so axes is iterable)
    if num_plots == 1:
        axes = [axes]
        
    # Process each isotope and plot onto its respective axis
    for i, data in enumerate(datasets):
        analysis = AlphaSpectrumAnalyser(
            csv_path=data["path"],
            isotope_name=data["source"]
        )
        # Pass the specific subplot axis to the analysis pipeline
        analysis.run_full_analysis(ax=axes[i])
    
    # Final global adjustments and saving the consolidated figure
    plt.suptitle("Calibrated Alpha Energy Spectra Comparison", fontsize=14, weight='bold', color="#F5FDFF")
    
    for i, ax in enumerate(axes):
        if i == 0:
            ax.set_ylabel('Counts', color='#BBC2C6', weight='bold', fontsize=12)
        ax.grid(alpha=0.3, color='#BBC2C6', linestyle='--')
        ax.set_xlim(3.5, 8.5)
        ax.set_facecolor('#223441')
        ax.tick_params(colors='#BBC2C6', which='both')  # Set tick colors for both major and minor ticks
                
    handles, labels = axes[0].get_legend_handles_labels()
    
    # 2. Add the legend to the overall figure layout
    fig.legend(
        handles, 
        labels, 
        loc='lower center',       # Positions it at the bottom middle
        ncol=3,                   # Forces the items to sit side-by-side horizontally
        bbox_to_anchor=(0.5, -0.05), # Fine-tunes positioning (X=center, Y=slightly below plots)
        facecolor='inherit',
        frameon=False,              # True or False depending on if you want a box border
        labelcolor='#BBC2C6',                     # Changes the text color
        prop={'size': 12, 'weight': 'bold'}       # Changes the font properties
    )
    
    plt.tight_layout(rect=(0, 0.05, 1, 1))

    os.makedirs("Figures", exist_ok=True)
    save_path = os.path.join("Figures", "Combined_Alpha_Spectra.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"\n[SUCCESS] Combined plot successfully saved to '{save_path}'")
    
    plt.show()
    
    