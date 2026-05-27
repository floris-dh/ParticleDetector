import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, root
import time
import os

# 1. Define the complete model function
def pulse_model_with_undershoot(t, t0, A, tau_1, tau_2, B, tau_3, C):
    dt = t - t0
    with np.errstate(divide='ignore', invalid='ignore'):
        # Correctly expanded one-liner
        E1 = np.exp(-dt / tau_1)
        E2 = np.exp(-dt / tau_2)
        E3 = np.exp(-dt / tau_3)
        model = C - (A + B) * E1 + A * E2 + B * E3
    
    return np.where(dt >= 0, model, C)

plt.figure(figsize=(10, 6))

integral_list = []

i = 1
while True:
    filepath = f"Data/RawData/acq{i:04d}.csv"
    if os.path.exists(filepath):
        try:
            data = pd.read_csv(f"Data/RawData/acq{i:04d}.csv", skiprows=10, header=None)
            data.columns = ['times', 'voltage']

            t = data['times'].values
            v = data['voltage'].values
            # --- FIX: Dynamically scaled Guesses and Bounds ---
            v_min_idx = np.argmin(v)
            t_at_min = t[v_min_idx]           
            v_min = v[v_min_idx]               
            est_baseline = np.mean(v[:200])    
            est_A = np.abs(v_min - est_baseline)

            # Calculate the total time width of your data file to scale bounds properly
            total_time_span = t.max() - t.min()

            # 3. Dynamic Initial Guesses
            p0 = [
                t_at_min - (total_time_span * 0.05),  # t0: Guess 5% of the window width before the peak
                est_A,                                # A
                total_time_span * 0.15,               # tau_decay: assume tail takes up ~15% of window
                total_time_span * 0.01,               # tau_rise: assume very fast 1% rise
                est_A * 0.3,                          # A_undershoot: assume undershoot is ~30% of peak height
                total_time_span * 0.5,                # tau_undershoot: slow tail across half the window
                est_baseline                          # baseline
            ]

            # 4. Safe, Multi-Order-of-Magnitude Bounds
            # Format: [t0, A, tau_decay, tau_rise, A_undershoot, tau_undershoot, baseline]
            lower_bounds = [
                t.min(),              # t0 must be within the file time
                0.001,                # Min amplitude (1 mV)
                1e-6,                 # Min decay time constant
                1e-7,                 # Min rise time constant (keeps it positive)
                0.0,                  # Undershoot can be zero if pulse is perfect
                1e-5,                 # Min undershoot decay time
                est_baseline - 0.01   # Baseline variance lower limit
            ]

            upper_bounds = [
                t_at_min,             # t0 MUST happen before the minimum peak value is reached
                0.100,                # Max amplitude (100 mV)
                total_time_span,      # Decay cannot be longer than the whole file window
                total_time_span * 0.2,# Rise cannot take up more than 20% of the file
                est_A * 2.0,          # Undershoot amplitude shouldn't exceed twice the pulse height
                total_time_span * 5.0,# Undershoot can be extremely slow
                est_baseline + 0.01   # Baseline variance upper limit
            ]

            bounds = (lower_bounds, upper_bounds)

            # 5. Execute with robust error handling
            try:
                popt, pcov = curve_fit(
                    pulse_model_with_undershoot, t, v, 
                    p0=p0, 
                    bounds=bounds, 
                    maxfev=20000  # Give it plenty of iterations to converge
                )
                
                # Extract optimized parameters
                t0, A, tau_1, tau_2, B, tau_3, C = popt
                v_fit = pulse_model_with_undershoot(t, *popt)
                chi_squared_red = np.sum((v - v_fit) ** 2) / (len(v) - len(popt))
                if chi_squared_red > 1e-6:  # Arbitrary threshold for "goodness" of fit
                    discard = True
                else:
                    discard = False
                
                # root_result = root(lambda x: pulse_model_with_undershoot(x, *popt) - C, 0.0015)
                # t_root = root_result.x[0]
                
                # 1. Center the fit on zero
                centered_fit = v_fit - C
                negative_only_curve = np.where(centered_fit < 0, centered_fit, 0.0)
                negative_integral = np.trapezoid(negative_only_curve, t)
                
                # print("\n--- FIT SUCCESSFUL ---")
                # print(f"Calculated True Peak Amplitude: {A:.4f} V")
                # print(f"Pulse Start Time (t0): {t0:.6f} s")
                # print(f"Preamplifier Decay Constant: {tau_1:.6f} s")
                # print(f"Negative Integral: {negative_integral:.4e} V*s")
                # plt.cla()
                # plt.plot(t, v, label='Raw Data', alpha=0.5, color='steelblue')
                # plt.plot(t, v_fit, label=f'Fit (χ²: {chi_squared_red:.2e})', color='red', linewidth=2.5)
                # plt.axhline(C, color='black', linestyle='--', label='Baseline')
                # plt.axvline(t0, color='purple', linestyle=':', label='Detected Onset (t0)')
                # plt.axvline(t_root, color='orange', linestyle='-.', label='Baseline Crossing Time (t_base)')
                # if discard:
                #     plt.title("DISCARDED PULSE - Poor Fit")
                # else:
                #     plt.title("CORRECT PULSE - Good Fit")
                # plt.xlabel("Time (s)")
                # plt.ylabel("Voltage (V)")
                # plt.legend()
                # plt.grid(True, alpha=0.3)
                # plt.show(block=True)
                if discard:
                    print("Discard" + " "*100, end="\r")
                else:
                    print(f"{i/1000:.2f}% | t0: {t0:.6f} s | A: {A:.4f} V | tau_decay: {tau_1:.6f} s | tau_rise: {tau_2:.6f} s | A_undershoot: {B:.4f} V | tau_undershoot: {tau_3:.6f} s | Baseline: {C:.4f} V | χ²_red: {chi_squared_red:.2e}", end="\r")

                
            except ValueError as e:
                print(f"Configuration Error: Check your bounds setup. Info: {e}", end="\r")
            except RuntimeError:
                print("Fit still failed. The optimizer couldn't converge within limits.", end="\r")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            time.sleep(0.1)  # Wait briefly before trying again
        i += 1
    else:
        print(f"File not found: {filepath}")
        break
        
counts, bins = np.histogram(integral_list, bins=50)
plt.figure(figsize=(10, 6))
plt.stairs(counts, bins)
plt.xlabel('Negative Integral (V*s)')
plt.ylabel('Frequency')
plt.title('Distribution of Negative Integrals from Fitted Pulses')
plt.grid(True, alpha=0.3)
plt.show()