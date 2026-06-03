import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
CSV_FILE_PATH = r"Data\Pu-239_010626.csv"
OUTPUT_FOLDER = r"Data\Pu-239_010626_output"
MAX_PLOTS = 10  

def pulse_model(t, t0, a, tau_1, tau_2, b, tau_3, c) -> np.ndarray:
    with np.errstate(over='ignore'):
        dt = t - t0
        result = (c - (a + b) * np.exp(-dt / tau_1)
                  + a * np.exp(-dt / tau_2)
                  + b * np.exp(-dt / tau_3))
    return np.where(dt >= 0, result, c)

def fit_single_file(packet):
    """
    Receives a tuple of (chunk_id, time_array, voltage_array)
    """
    try:
        chunk_id, t, v = packet
        
        v_min_idx = np.argmin(v)
        t_at_min = t[v_min_idx]
        v_min = v[v_min_idx]
        est_baseline = np.mean(v[:200])
        est_a = abs(v_min - est_baseline)
        span = t[-1] - t[0]

        # Pre-filter low quality pulses
        snr = est_a / (np.std(v[:200]) + 1e-12)
        if snr < 3.0:
            return None

        stride = max(1, len(t) // 100)
        t_fit, v_fit = t[::stride], v[::stride]

        p0 = [t_at_min - span * 0.05, est_a, span * 0.15,
              span * 0.01, est_a * 0.3, span * 0.5, est_baseline]
        bounds = (
            [-0.005, 0.001, 1e-6, 1e-7, 0.0, 1e-5, est_baseline - 0.01],
            [0.005, 0.5, span, span * 0.2, est_a * 2.0, span * 5.0, est_baseline + 0.01]
        )

        popt, _ = curve_fit(
            pulse_model, t_fit, v_fit, p0=p0, bounds=bounds, method='trf',
            maxfev=2000, ftol=1e-4, xtol=1e-4, gtol=1e-4
        )
        
        chi2_red = np.sum((v_fit - pulse_model(t_fit, *popt))**2) / (len(t_fit) - len(popt))
        if chi2_red > 1e-6:
            return None
        
        # Keep arrays for the first few items to handle plotting without re-filtering
        return {
            'chunk_id': chunk_id,
            't0': popt[0], 'A': popt[1], 'tau_1': popt[2],
            'tau_2': popt[3], 'B': popt[4], 'tau_3': popt[5], 'C': popt[6],
            't_start': float(t[0]), 't_end': float(t[-1]), 'n_points': len(t),
            '_raw_t': t if chunk_id < MAX_PLOTS * 2 else None, # Cache raw data only for early chunks
            '_raw_v': v if chunk_id < MAX_PLOTS * 2 else None
        }
    except Exception:
        return None
    
def compute_integrals(results_df):
    print(f"Computing integrals for {len(results_df)} pulses...")

    # Vectorized expressions entirely native to Polars (Much cleaner than NumPy unpacking)
    dt = results_df["t_end"] - results_df["t0"]
    dt = pl.when(dt > 0).then(dt).otherwise(0.0)

    integrals_df = results_df.with_columns(
        pulse_integral = (
            (pl.col("A") + pl.col("B")) * pl.col("tau_1") * ((-dt / pl.col("tau_1")).exp() - 1.0)
            - pl.col("A") * pl.col("tau_2") * ((-dt / pl.col("tau_2")).exp() - 1.0)
            - pl.col("B") * pl.col("tau_3") * ((-dt / pl.col("tau_3")).exp() - 1.0)
        )
    ).select(["chunk_id", "pulse_integral"]).sort("chunk_id")

    output_path = os.path.join(OUTPUT_FOLDER, "pulse_integrals.csv")
    integrals_df.write_csv(output_path)
    
    print("\n--- Processing Complete ---")
    print(integrals_df.head(5))
    print(f"\nSuccessfully saved integrals to: '{output_path}'")

def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("Loading and cleaning data...")

    # Unified Ingestion: Skips junk metadata rows, infers types, ignores comment/corrupt data rows.
    df = pl.read_csv(
            CSV_FILE_PATH,
            skip_rows=9,
            has_header=False,
            new_columns=["Time (s)", "Channel 1 (V)"],
            schema_overrides={
                "Time (s)": pl.Float64,
                "Channel 1 (V)": pl.Float64
            },
            comment_prefix="Time",  # Drops any repeated header rows safely
            null_values=["NA", "null", "NaN"],
            ignore_errors=True,     # Forces Polars to skip rows with corrupt text artifacts instead of crashing
            infer_schema_length=0   # Let schema_overrides do 100% of the type work
        ).drop_nulls()

    print(f"Data loaded successfully. Total rows: {len(df)}")
    print("Detecting new acquisition windows via time resets...")

    # Optimize processing using Polars expressions 
    df_with_chunks = df.with_columns(
        chunk_id = pl.col("Time (s)").diff().fill_null(0.0).lt(0).cum_sum()
    )

    df_with_chunks.write_parquet(os.path.join(OUTPUT_FOLDER, "processed_data.parquet"))

    print("Packing pulse signals into memory pools...")
    # Generate list of tuples (chunk_id, time_array, voltage_array) without forcing heavy dict structures
    grouped = df_with_chunks.group_by("chunk_id").agg([pl.col("Time (s)"), pl.col("Channel 1 (V)")]).sort("chunk_id")
    
    packets = list(zip(
        grouped["chunk_id"].to_list(),
        [a.to_numpy() for a in grouped["Time (s)"]],
        [a.to_numpy() for a in grouped["Channel 1 (V)"]]
    ))
    
    total_chunks = len(packets)
    del df, grouped # Clear overhead memory

    print(f"Spawning worker processes. Safely iterating down traces...")
    results_list = []
    
    # Utilizing 4 workers as configured in your executor
    with ProcessPoolExecutor(max_workers=4) as executor:
        raw_results = executor.map(fit_single_file, packets, chunksize=10)
        for i, res in enumerate(raw_results):
            if res is not None:
                results_list.append(res)
            if i % 100 == 0:
                print(f"Processed trace {i} / {total_chunks}...", end="\r")

    print(f"\nCompleted processing. Successfully fit {len(results_list)} / {total_chunks} windows.")

    if not results_list:
        print("[Warning] No alpha pulses successfully passed fit validation metrics.")
        return

    # Convert results into Polars Frame
    results_df = pl.DataFrame(results_list)

    # 5. Optimized Plotting (Uses pre-cached raw data arrays inside the dictionary)
    print(f"Plotting up to the first {MAX_PLOTS} verified successful fits...")
    plots_made = 0

    for res in results_list:
        if plots_made >= MAX_PLOTS:
            break
        if res['_raw_t'] is None: # Edge case handler
            continue
            
        c_id = res['chunk_id']
        t_arr, v_arr = res['_raw_t'], res['_raw_v']
        
        plt.figure(figsize=(9, 4))
        plt.plot(t_arr, v_arr, label=f"Data (Chunk {c_id})", alpha=0.6)
        
        fit_v = pulse_model(
            t_arr, res['t0'], res['A'], res['tau_1'], res['tau_2'], res['B'], res['tau_3'], res['C']
        )
        
        plt.plot(t_arr, fit_v, label="Model Fit", color="red", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(f"Plutonium-239 Pulse Fit Analysis - Chunk {c_id}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"fit_plot_{c_id:04d}.png"), dpi=120)
        plt.close()
        plots_made += 1

    # Drop hidden numpy caching arrays before saving results structural table
    clean_results_df = results_df.drop(["_raw_t", "_raw_v"]).sort("chunk_id")
    clean_results_df.write_csv(os.path.join(OUTPUT_FOLDER, "fitted_params.csv"))
    print(f"Parameters saved successfully to '{OUTPUT_FOLDER}/fitted_params.csv'.")
        
    compute_integrals(clean_results_df)
    print(f"\nProcessing complete! All outputs securely isolated in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    main()