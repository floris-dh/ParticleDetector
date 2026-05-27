import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit

# 1. Define the model (Removed np.errstate overhead; bounds protect from divide-by-zero)
@njit(fastmath=True, cache=True)
def pulse_model_with_undershoot(t, t0, A, tau_1, tau_2, B, tau_3, C):
    dt = t - t0
    E1 = np.exp(-dt / tau_1)
    E2 = np.exp(-dt / tau_2)
    E3 = np.exp(-dt / tau_3)
    model = C - (A + B) * E1 + A * E2 + B * E3
    return np.where(dt >= 0, model, C)

# 2. Define the worker function for a single file
def process_single_file(filepath):
    try:
        # Fast CSV reading using the C engine
        data = pd.read_csv(filepath, skiprows=10, header=None, engine='c')
        t = data[0].values
        v = data[1].values
        
        # Setup bounds and guesses
        v_min_idx = np.argmin(v)
        t_at_min = t[v_min_idx]          
        v_min = v[v_min_idx]               
        est_baseline = np.mean(v[:200])    
        est_A = np.abs(v_min - est_baseline)
        total_time_span = t[-1] - t[0]

        p0 = [
            t_at_min - (total_time_span * 0.05),
            est_A,                                
            total_time_span * 0.15,               
            total_time_span * 0.01,               
            est_A * 0.3,                          
            total_time_span * 0.5,                
            est_baseline                          
        ]

        lower_bounds = [
            t[0], 0.001, 1e-6, 1e-7, 0.0, 1e-5, est_baseline - 0.01
        ]
        upper_bounds = [
            t_at_min, 0.100, total_time_span, total_time_span * 0.2, 
            est_A * 2.0, total_time_span * 5.0, est_baseline + 0.01
        ]
        bounds = (lower_bounds, upper_bounds)

        # Fit with a reduced maxfev to avoid hanging on garbage data
        popt, pcov = curve_fit(
            pulse_model_with_undershoot, t, v, 
            p0=p0, bounds=bounds, maxfev=5000
        )
        
        # Calculate Goodness of fit
        C_fit = popt[6]
        v_fit = pulse_model_with_undershoot(t, *popt)
        chi_squared_red = np.sum((v - v_fit) ** 2) / (len(v) - len(popt))
        
        if chi_squared_red > 1e-6:
            return None  # Discard bad fits
            
        # Fast Integration
        centered_fit = v_fit - C_fit
        negative_only_curve = np.where(centered_fit < 0, centered_fit, 0.0)
        negative_integral = np.trapezoid(negative_only_curve, t)
        
        return negative_integral

    except Exception:
        # Silently catch optimizer failures or read errors to keep the pool moving
        return None

# 3. Main execution block
if __name__ == '__main__':
    # Grab all files instantly
    file_pattern = "Data/RawData/acq*.csv"
    all_files = glob.glob(file_pattern)
    
    if not all_files:
        print("No files found matching the pattern.")
    else:
        print(f"Found {len(all_files)} files. Starting parallel processing...")
        
        integral_list = []
        processed_count = 0
        discarded_count = 0

        # Execute using all available CPU cores
        with ProcessPoolExecutor() as executor:
            # Submit all tasks
            futures = {executor.submit(process_single_file, fp): fp for fp in all_files}
            
            # Gather results as they finish (order doesn't matter for a histogram)
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    integral_list.append(result)
                else:
                    discarded_count += 1
                
                # Simple progress tracker
                processed_count += 1
                if processed_count % 10 == 0:
                    print(f"Processed {processed_count}/{len(all_files)} files. "
                          f"(Discarded: {discarded_count})", end="\r")
        
        print(f"\nProcessing complete! Successfully extracted {len(integral_list)} integrals.")

        # 4. Plot Histogram
        if integral_list:
            counts, bins = np.histogram(integral_list, bins=50)
            plt.figure(figsize=(10, 6))
            plt.stairs(counts, bins, fill=True, color='steelblue', alpha=0.7)
            plt.xlabel('Negative Integral (V*s)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Negative Integrals from Fitted Pulses')
            plt.grid(True, alpha=0.3)
            plt.show()