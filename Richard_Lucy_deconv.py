import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import scipy.signal as signal
from pathlib import Path

# =============================================================================
# GLOBAL CONFIGURATION (YOUR DASHBOARD)
# =============================================================================
# Het centrale pad naar jouw Data map
DATA_DIRECTORY      = r"C:\Users\Nats91\Particle_Detector_PC\ParticleDetector\Data"

AM241_CSV_NAME       = "integrals_Am241.csv"       
RA226_CSV_NAME       = "integrals_Ra226.csv" 

DECONV_ITERATIONS    = 3  
# =============================================================================

def extract_empiric_psf(am241_integrals, num_bins):
    """
    Extracts, smooths, centers, and normalizes the empirical PSF 
    from the Am241 calibration data based on your supervisor's feedback.
    """
    # Generate raw Poisson counts histogram
    counts, bin_edges = np.histogram(am241_integrals, bins=num_bins, density=False)
    counts = counts.astype(float)
    
    # -------------------------------------------------------------------------
    # BASELINE / PEDESTAL SUBTRACTION
    # -------------------------------------------------------------------------
    left_bg = counts[:20]
    right_bg = counts[-20:]

    baseline = np.median(np.concatenate([left_bg, right_bg]))
    print(f"PSF baseline level: {baseline:.2f} counts/bin")

    counts = counts - baseline

    # Enforce physical positivity
    counts[counts < 0] = 0
    # -------------------------------------------------------------------------
    
    # Find the main peak of Am241 (5.49 MeV) after baseline subtraction
    peak_idx = np.argmax(counts)
    
    # Extract a wide enough window to capture the low-energy tail
    window_half = num_bins // 5  
    start_idx = max(0, peak_idx - window_half)
    end_idx = min(len(counts), peak_idx + window_half + 1)
    peak_window = counts[start_idx:end_idx].copy()
    
    # Apply Savitzky-Golay smoothing to eliminate statistical noise
    if len(peak_window) > 15:
        peak_window = signal.savgol_filter(peak_window, window_length=15, polyorder=2)
    peak_window[peak_window < 0] = 0  # Enforce physical positivity
    
    # Center the PSF perfectly in the middle of the kernel array
    local_center = np.argmax(peak_window)
    target_center = len(peak_window) // 2
    shift_amount = target_center - local_center
    centered_psf = np.roll(peak_window, shift_amount)
    
    # Normalize so the total sum is 1.0 (particle conservation)
    empiric_psf = centered_psf / np.sum(centered_psf)
    
    print(f"PSF Extracted: Smoothed, centered around index {target_center}, and normalized.")
    return empiric_psf

def richard_lucy_deconvolution(measurement, response_function, iterations=20):
    """Performs non-parametric Richardson-Lucy deconvolution using FFT convolutions."""
    response_function = response_function / np.sum(response_function)
    initial_guess = np.copy(measurement).astype(float)  
    flipped_response = response_function[::-1]
    
    for _ in range(iterations):
        og_prediction = signal.fftconvolve(initial_guess, response_function, mode='same')
        ratio = measurement / (og_prediction + 1e-10)
        correction = signal.fftconvolve(ratio, flipped_response, mode='same')
        initial_guess *= correction  
        
    return initial_guess

if __name__ == "__main__":
    # We halen de data direct uit de centrale map
    am241_path = os.path.join(DATA_DIRECTORY, AM241_CSV_NAME)
    ra226_path = os.path.join(DATA_DIRECTORY, RA226_CSV_NAME)
    
    if not Path(am241_path).exists() or not Path(ra226_path).exists():
        print("ERROR: One or both input CSV files could not be found!")
        print(f"Looking for:\n -> {am241_path}\n -> {ra226_path}")
        exit()
        
    df_am241 = pl.read_csv(am241_path)
    df_ra226 = pl.read_csv(ra226_path)
    
    am241_data = df_am241['integral_Vs'].to_numpy()
    ra226_data = df_ra226['integral_Vs'].to_numpy()
    
    # Set bin count gebaseerd op de statistiek van je nieuwe grote Radium-meting
    num_bins = int(np.sqrt(len(ra226_data)))
    if num_bins < 150: num_bins = 200  # Iets more bins voor betere resolutie bij veel data
    
    # Process PSF met de feedback regels (inclusief baseline subtraction!)
    empiric_psf = extract_empiric_psf(am241_data, num_bins)
    
    # Process Target Radium Spectrum (Raw Counts)
    ra226_counts, bin_edges = np.histogram(ra226_data, bins=num_bins, density=False)
    ra226_counts = ra226_counts.astype(float)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # TARGET SPECTRUM BASELINE BACKGROUND SUBTRACTION
    left_bg = ra226_counts[:20]
    right_bg = ra226_counts[-20:]
    baseline_target = np.median(np.concatenate([left_bg, right_bg]))
    print(f"Target (Radium) baseline level: {baseline_target:.2f} counts/bin")
    
    ra226_counts = ra226_counts - baseline_target
    ra226_counts[ra226_counts < 0] = 0
    
    print("Executing fast FFT Richardson-Lucy deconvolution...")
    deconvolved_spectrum = richard_lucy_deconvolution(ra226_counts, empiric_psf, iterations=DECONV_ITERATIONS)
    
    # =============================================================================
    # VISUALIZATION
    # =============================================================================
    print("Generating calibrated energy spectrum plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [1, 3]})
    
    # Left subplot: Clean, centered PSF
    ax1.plot(empiric_psf, color='darkorange', linewidth=2, label='Centered & Smoothed PSF')
    ax1.fill_between(range(len(empiric_psf)), empiric_psf, color='darkorange', alpha=0.15)
    ax1.axvline(x=len(empiric_psf)//2, color='black', linestyle=':', alpha=0.7, label='Center')
    ax1.set_title("Detector PSF Profile\n(Net Background $^{241}$Am)", fontsize=11, fontweight='bold')
    ax1.set_xlabel("Relative Bin Index")
    ax1.set_ylabel("Relative Probability")
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Right subplot: True Radium spectrum
    ax2.stairs(
        ra226_counts, bin_edges, 
        edgecolor="black", fill=True, alpha=0.2, 
        color='steelblue', label='Raw Measured $^{226}$Ra Spectrum'
    )
    ax2.step(
        bin_centers, deconvolved_spectrum, 
        where='mid', color='crimson', linewidth=2, 
        label=f'Deconvolved Spectrum ({DECONV_ITERATIONS} iterations via FFT)'
    )
    
    ax2.set_title("Alpha Particle Energy Spectrum — Radium-226 (Optimized)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("Pulse Integral (V*s) ~ Energy Proxy", fontsize=11)
    ax2.set_ylabel("Counts (Absolute Pulse Frequency)", fontsize=11)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper right', fontsize=10)
    
    output_pdf = os.path.join(DATA_DIRECTORY, "final_calibrated_spectrum_Radium226.pdf")
    plt.savefig(output_pdf, bbox_inches='tight')
    print(f"Analysis complete! Plot exported to: {output_pdf}")
    
    plt.show()