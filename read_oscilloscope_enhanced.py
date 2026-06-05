import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor

# --- Configuration ---
CSV_FILE_PATH = r"Data\Am241_050626-1.csv"
OUTPUT_FOLDER = r"Data\Am241-050626-1_output"
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
    print(f"Computing integrals until signal returns to baseline (C)...")

    # 1. Haal de parameters naar NumPy
    chunk_ids = results_df["chunk_id"].to_numpy()
    A = results_df["A"].to_numpy()
    B = results_df["B"].to_numpy()
    tau_1 = results_df["tau_1"].to_numpy()
    tau_2 = results_df["tau_2"].to_numpy()
    tau_3 = results_df["tau_3"].to_numpy()
    
    num_pulses = len(results_df)
    dt_zero_arr = np.zeros(num_pulses)

    # We scannen per puls heel snel een tijd-vector om het exacte nulpunt te vinden
    # Dit is extreem robuust en voorkomt dat een solver verdwaalt in de asymptotische staart
    for i in range(num_pulses):
        # Maak een fijn tijdsraster vanaf 0 tot een ruime schatting van de pulslengte
        max_t = 5.0 * max(tau_2[i], tau_3[i])
        t_scan = np.linspace(0.0, max_t, 1000)
        
        # Bereken het model (zonder baseline C, dus we zoeken waar dit 0 wordt)
        with np.errstate(over='ignore'):
            f_scan = -(A[i] + B[i]) * np.exp(-t_scan / tau_1[i]) + A[i] * np.exp(-t_scan / tau_2[i]) + B[i] * np.exp(-t_scan / tau_3[i])
        
        # De puls start negatief. We zoeken het EERSTE punt NA de piek waar het signaal >= 0 wordt
        # Zoek eerst de index van de minimale waarde (de piek van de puls)
        peak_idx = np.argmin(f_scan)
        
        # Zoek vanaf de piek naar het moment dat hij de nullijn kruist
        zero_crossings = np.where(f_scan[peak_idx:] >= 0)[0]
        
        if len(zero_crossings) > 0:
            # Gevonden! Neem de exacte tijd van de kruising
            dt_zero_arr[i] = t_scan[peak_idx + zero_crossings[0]]
        else:
            # Mocht hij de 0 net niet aantikken in de scan, neem dan het einde van de scan
            dt_zero_arr[i] = max_t

    # 2. Voeg de stabiele dt_zero toe aan Polars
    results_with_dt = results_df.with_columns(
        dt_zero = pl.Series(dt_zero_arr)
    )

    # 3. Bereken de integraal (C - V(t)), wat de pure positieve oppervlakte geeft van de dip
    integrals_df = results_with_dt.with_columns(
        pulse_integral = (
            - (pl.col("A") + pl.col("B")) * pl.col("tau_1") * ((-pl.col("dt_zero") / pl.col("tau_1")).exp() - 1.0)
            + pl.col("A") * pl.col("tau_2") * ((-pl.col("dt_zero") / pl.col("tau_2")).exp() - 1.0)
            + pl.col("B") * pl.col("tau_3") * ((-pl.col("dt_zero") / pl.col("tau_3")).exp() - 1.0)
        )
    )
    
    # Vervang eventuele zeldzame NaN/negatieve uitschieters door 0.0
    integrals_df = integrals_df.with_columns(
        pulse_integral = pl.when(pl.col("pulse_integral") > 0)
                           .then(pl.col("pulse_integral"))
                           .otherwise(0.0)
    ).select(["chunk_id", "pulse_integral"]).sort("chunk_id")

    output_path = os.path.join(OUTPUT_FOLDER, "pulse_integrals.csv")
    integrals_df.write_csv(output_path)
    
    print("\n--- Processing Complete ---")
    print(integrals_df.head(20)) # Toon er direct 20 om het resultaat te keuren
    print(f"\nSuccessfully saved bounded integrals to: '{output_path}'")

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