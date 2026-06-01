import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
CSV_FILE_PATH = r"Data\Pu-239_010626.csv"
OUTPUT_FOLDER = r"Data\Pu-239_010626_output"
MAX_PLOTS = 10  # Safety limit for rendering images

def pulse_model(t, t0, a, tau_1, tau_2, b, tau_3, c) -> np.ndarray:
    with np.errstate(over='ignore'):
        dt = t - t0
        result = (c - (a + b) * np.exp(-dt / tau_1)
              + a * np.exp(-dt / tau_2)
              + b * np.exp(-dt / tau_3))
    return np.where(dt >= 0, result, c)

def fit_single_file(sub_df):
    """
    Receives a preloaded slice of data for a specific file, avoiding disk I/O.
    Accepts a dictionary packet containing pre-extracted numpy arrays.
    """
    try:
        # Pull vectors from the passed dictionary packet
        t = np.array(sub_df['Time (s)'])
        v = np.array(sub_df['Channel 1 (V)'])
        chunk_id = sub_df['chunk_id']

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

        # Downsample for fitting to save CPU cycles
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
        
        return {
            'chunk_id': chunk_id,
            't0': popt[0], 'A': popt[1], 'tau_1': popt[2],
            'tau_2': popt[3], 'B': popt[4], 'tau_3': popt[5],
            'C': popt[6],
            't_start': float(t[0]), 't_end': float(t[-1]), 'n_points': len(t)
        }
    except Exception:
        return None
    
def compute_integrals(results_df):

    print(f"Computing integrals for {len(results_df)} pulses...")

    # Extract all param columns as raw, flat NumPy arrays for unified vectorization
    chunk_ids = results_df["chunk_id"].to_numpy()
    t0 = results_df["t0"].to_numpy()
    t_end = results_df["t_end"].to_numpy()
    a = results_df["A"].to_numpy()
    b = results_df["B"].to_numpy()
    tau_1 = results_df["tau_1"].to_numpy()
    tau_2 = results_df["tau_2"].to_numpy()
    tau_3 = results_df["tau_3"].to_numpy()

    # Calculate the upper boundary integration limit relative to t0
    dt = t_end - t0

    # Ensure we don't calculate backward times if window tracking artifacted
    dt = np.where(dt > 0, dt, 0.0)

    # --- Analytical Integral
    pulse_integrals = (
        (a + b) * tau_1 * (np.exp(-dt / tau_1) - 1.0)
        - a * tau_2 * (np.exp(-dt / tau_2) - 1.0)
        - b * tau_3 * (np.exp(-dt / tau_3) - 1.0)
    )

    # Compile vectors back cleanly into a structural Polars Frame
    integrals_df = pl.DataFrame({
        "chunk_id": chunk_ids,
        "pulse_integral": pulse_integrals
    }).sort("chunk_id")

    # Output results
    output_path = os.path.join(OUTPUT_FOLDER, "pulse_integrals.csv")
    integrals_df.write_csv(output_path)
    
    print("\n--- Processing Complete ---")
    print(integrals_df.head(5))
    print(f"\nSuccessfully saved integrals to: '{output_path}'")

def main():
    # Ensure the output directory exists
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading and cleaning data...")
    # 1. Load the CSV bypassing WaveForms metadata and ragged lines
    df = pl.read_csv(
        CSV_FILE_PATH, 
        skip_rows=9,                      
        truncate_ragged_lines=True,       
        infer_schema_length=10000,        
        ignore_errors=True                
    )

    # 2. Clean data
    df = df.filter(pl.col("Time (s)") != "Time (s)")

    # 3. Explicitly enforce numeric data types
    df = df.with_columns([
        pl.col("Time (s)").cast(pl.Float64, strict=False),
        pl.col("Channel 1 (V)").cast(pl.Float64, strict=False)
    ]).drop_nulls()

    print(f"Data loaded successfully. Total rows: {len(df)}")
    print("Detecting new acquisition windows via time resets...")

    # 4. Dynamic Slicing Engine
    df_with_chunks = df.with_columns(
        pl.col("Time (s)").diff().fill_null(0.0).alias("time_diff")
    ).with_columns(
        (pl.col("time_diff") < 0).cum_sum().alias("chunk_id")
    ).drop("time_diff")

    # Save out the master cleaned Parquet file for archiving
    df_with_chunks.write_parquet(os.path.join(OUTPUT_FOLDER, "processed_data.parquet"))
    print(f"Cleaned tracking data saved to '{OUTPUT_FOLDER}/processed_data.parquet'.")

    # --- INSTANT PACKETING EXTRACTION ---
    print("Packing pulse signals into memory via fast aggregations...")
    
    # We aggregate columns into Lists grouped by chunk_id natively in Polars
    aggregated_df = (
        df_with_chunks.group_by("chunk_id")
        .agg([
            pl.col("Time (s)"),
            pl.col("Channel 1 (V)")
        ])
        .sort("chunk_id")
    )
    
    total_chunks = len(aggregated_df)
    print(f"Successfully identified {total_chunks} distinct acquisition windows.")

    # Convert the aggregated rows directly into Python dictionaries instantly
    chunk_dicts = aggregated_df.to_dicts()
    
    # Free up memory from raw frames we no longer need
    del df
    del aggregated_df
    
    # --- LAPTOP-SAFE PARALLEL EXECUTION ---
    print(f"Spawning 2 worker processes. Safely iterating down traces...")
    results_list = []
    
    with ProcessPoolExecutor(max_workers=4) as executor:
        # chunksize=10 streams the data arrays incrementally instead of flooding memory core bridges
        raw_results = executor.map(fit_single_file, chunk_dicts, chunksize=10)
        
        for i, res in enumerate(raw_results):
            if res is not None:
                results_list.append(res)
            if i % 100 == 0:
                print(f"Processed trace {i} / {total_chunks}...", end="\r")

    print(f"\nCompleted processing. Successfully fit {len(results_list)} / {total_chunks} windows.")

    # Save calculated parameter structures out to CSV
    if results_list:
        results_df = pl.DataFrame(results_list).sort("chunk_id")
        results_df.write_csv(os.path.join(OUTPUT_FOLDER, "fitted_params.csv"))
        print(f"Parameters saved successfully to '{OUTPUT_FOLDER}/fitted_params.csv'.")
    else:
        print("[Warning] No alpha pulses successfully passed fit validation metrics.")
        return

    # 5. Plotting Step (Runs sequentially on local thread to avoid UI canvas crashes)
    print(f"Plotting up to the first {MAX_PLOTS} verified successful fits...")
    plots_made = 0

    for res in results_list:
        if plots_made >= MAX_PLOTS:
            break
            
        c_id = res['chunk_id']
        chunk_df = df_with_chunks.filter(pl.col("chunk_id") == c_id)
        
        plt.figure(figsize=(9, 4))
        plt.plot(chunk_df["Time (s)"], chunk_df["Channel 1 (V)"], label=f"Data (Chunk {c_id})", alpha=0.6)
        
        # Build calculated fit trace profile curves
        fit_v = pulse_model(
            chunk_df["Time (s)"].to_numpy(), 
            res['t0'], res['A'], res['tau_1'], res['tau_2'], res['B'], res['tau_3'], res['C']
        )
        
        plt.plot(chunk_df["Time (s)"], fit_v, label="Model Fit", color="red", linestyle="--")
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage (V)")
        plt.title(f"Plutonium-239 Pulse Fit Analysis - Chunk {c_id}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save plots directly to storage to prevent RAM leak cleanups
        plt.savefig(os.path.join(OUTPUT_FOLDER, f"fit_plot_{c_id:04d}.png"), dpi=120)
        plt.close()
        plots_made += 1
        
    compute_integrals(results_df)

    print(f"\nProcessing complete! All outputs securely isolated in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    main()