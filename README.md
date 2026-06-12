# Alpha Particle Detector with Digilent AD2

## Overview
This project involves reading and processing alpha particle signals using a Digilent Analog Discovery 2 (AD2) as a data acquisition system. It provides an end-to-end pipeline: from raw oscilloscope measurements to calibrated energy spectra capable of identifying specific radioactive isotopes (such as Am-241, Pu-239, and Ra-226).

## Background
When alpha particles interact with a solid-state detector (like a PIN photodiode), they deposit energy and create electron-hole pairs along their track. This ionization generates a microscopic charge pulse. A charge-sensitive preamplifier and a shaping amplifier convert this minute charge into a measurable macroscopic voltage pulse. 

The Digilent AD2 acts as an oscilloscope, triggering on these transient events and capturing the high-speed voltage waveforms. The captured traces are then exported and analyzed by our software stack to reconstruct the original energy of the incident alpha particles.

## Mathematical Models

### 1. Pulse Shape Fitting
The raw voltage pulses recorded by the AD2 are fitted to a custom analytical multi-exponential model. This accounts for the physics of charge collection and the RC characteristics of the shaping electronics.

The pulse voltage $V(t)$ is modeled as:

$$ V(t) = C - (A + B) e^{-\frac{t - t_0}{\tau_1}} + A e^{-\frac{t - t_0}{\tau_2}} + B e^{-\frac{t - t_0}{\tau_3}} $$

for $t \ge t_0$, where:
*   **$t_0$**: Start time of the pulse.
*   **$C$**: Baseline voltage offset.
*   **$\tau_1$**: Rise time constant.
*   **$\tau_2, \tau_3$**: Fall/recovery time constants.
*   **$A, B$**: Amplitudes associated with the respective decay components.

The kinetic energy of the alpha particle is proportional to the total integrated charge. We analytically compute the bounded integral of the negative excursion of the fitted pulse:

$$ \text{Integral} = \int_{0}^{\Delta t} \left( -(A+B)e^{-\frac{t}{\tau_1}} + Ae^{-\frac{t}{\tau_2}} + Be^{-\frac{t}{\tau_3}} \right) dt $$

### 2. Energy Spectrum Analysis
The computed pulse integrals represent the energy of the particles in arbitrary units (AU). The project groups these measurements into a histogram to form an energy spectrum.

To automatically identify the energy peaks, the software fits a multi-Gaussian distribution to the spectrum:

$$ H(x) = \sum_{i=1}^{N} A_i e^{-\frac{(x - \mu_i)^2}{2\sigma_i^2}} $$

Where:
*   **$N$**: Number of expected peaks (e.g., 4 peaks for Ra-226).
*   **$A_i$**: Amplitude (counts) of the $i$-th peak.
*   **$\mu_i$**: Centroid (mean energy) of the $i$-th peak.
*   **$\sigma_i$**: Standard deviation, representing the detector's energy resolution.

### 3. Energy Calibration
To convert the arbitrary units (AU) into physical units (MeV), a linear calibration is applied based on established literature values of alpha decay energies:

$$ E(\text{MeV}) = a \cdot \mu_{\text{AU}} + b $$

Once calibrated, we evaluate the detector's performance by measuring the Full Width at Half Maximum (FWHM) of the detected peaks:

$$ \text{FWHM} = 2.355 \cdot \sigma_{\text{MeV}} $$

*(Note: $2.355 \approx 2\sqrt{2\ln 2}$ is the conversion factor from standard deviation to FWHM for a Gaussian distribution).*

## Project Architecture
*   **`read_and_process.py`**: Takes raw CSV outputs from the AD2. It chunks the data, cleans it, and uses parallel processing (`ProcessPoolExecutor`) to fit the multi-exponential model to thousands of pulses simultaneously.
*   **`alpha_spectrum_analyser.py`**: Ingests the pulse integrals, constructs the histograms, and executes the multi-Gaussian peak fitting. Finally, it aligns the centroids with expected literature energies to produce calibrated plots.
*   **`informative_pulses.py`**: Extracts representative, high-quality pulses from the dataset for visualization and model verification.
*   **`Schematic_design_electrical_circuit.txt`**: The raw netlist / schematic specification for the analogue frontend hardware driving the AD2.

## Dependencies
*   Python 3.x
*   `numpy` & `scipy` (Signal processing, non-linear least squares fitting)
*   `polars` (Fast out-of-core data manipulation for large oscilloscope CSVs)
*   `matplotlib` (Data visualization)
