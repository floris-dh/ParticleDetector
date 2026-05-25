import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import time
import os

# 1. Define the complete model function
def pulse_model_with_undershoot(t, t0, A, tau_decay, tau_rise, A_undershoot, tau_undershoot, baseline):
    dt = t - t0
    with np.errstate(divide='ignore', invalid='ignore'):
        pulse = A * (np.exp(-dt / tau_decay) - np.exp(-dt / tau_rise))
        undershoot = A_undershoot * (np.exp(-dt / tau_undershoot) - np.exp(-dt / tau_decay))
    
    total_shape = pulse - undershoot
    total_shape = np.where(dt >= 0, total_shape, 0.0)
    return -total_shape + baseline

# 2. Load and isolate data


plt.figure(figsize=(10, 6))

i = 1
while True:
    filepath = f"RawData/acq{i:04d}.csv"
    if os.path.exists(filepath):
        try:
            data = pd.read_csv(f"RawData/acq{i:04d}.csv", skiprows=10, header=None)
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
                print("Attempting fit with bounded optimization...")
                popt, pcov = curve_fit(
                    pulse_model_with_undershoot, t, v, 
                    p0=p0, 
                    bounds=bounds, 
                    maxfev=20000  # Give it plenty of iterations to converge
                )
                
                # Extract optimized parameters
                t0_f, A_f, td_f, tr_f, Au_f, tu_f, b_f = popt
                v_fit = pulse_model_with_undershoot(t, *popt)
                chi_squared_red = np.sum((v - v_fit) ** 2) / (len(v) - len(popt))
                if chi_squared_red > 1e-6:  # Arbitrary threshold for "goodness" of fit
                    discard = True
                    print(f"Fit converged but has high chi-squared: {chi_squared_red:.3e}. Discarding this pulse.")
                else:
                    discard = False
                
                print("\n--- FIT SUCCESSFUL ---")
                print(f"Calculated True Peak Amplitude: {A_f:.4f} V")
                print(f"Pulse Start Time (t0): {t0_f:.6f} s")
                print(f"Preamplifier Decay Constant: {td_f:.6f} s")
                
                plt.cla()
                plt.plot(t, v, label='Raw Data', alpha=0.5, color='steelblue')
                plt.plot(t, v_fit, label=f'Fit (χ²: {chi_squared_red:.2e})', color='red', linewidth=2.5)
                plt.axvline(t0_f, color='purple', linestyle=':', label='Detected Onset (t0)')
                if discard:
                    plt.title("DISCARDED PULSE - Poor Fit")
                else:
                    plt.title("CORRECT PULSE - Good Fit")
                plt.xlabel("Time (s)")
                plt.ylabel("Voltage (V)")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.show(block=False)
                plt.pause(0.1)  # Brief pause
                


            except ValueError as e:
                print(f"\nConfiguration Error: Check your bounds setup. Info: {e}")
            except RuntimeError:
                print("\nFit still failed. The optimizer couldn't converge within limits.")
                print("Fallback strategy: Try narrowing your input time window array (t and v) around the pulse.")
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            time.sleep(0.1)  # Wait briefly before trying again
        i += 1
    else:
        print(f"File not found: {filepath}")
        break
        
