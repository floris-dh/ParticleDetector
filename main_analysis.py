from pathlib import Path
import glob
import os
import zipfile
import numpy as np
import polars as pl
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt

# ── Model ─────────────────────────────────────────────────────────────────────
def pulse_model(t, t0, a, tau_1, tau_2, b, tau_3, c) -> np.ndarray:
    dt = t - t0
    result = (c - (a + b) * np.exp(-dt / tau_1)
              + a * np.exp(-dt / tau_2)
              + b * np.exp(-dt / tau_3))
    return np.where(dt >= 0, result, c)


# ── Stage 1 worker: Polars reads the CSV, scipy fits it ───────────────────────
def fit_single_file(MEASUREMENT_ID, target_file):
    """
    Receives a preloaded slice of data for a specific file, avoiding disk I/O.
    """
    try:
        sub_df = (
            pl.scan_parquet(f"Data/{MEASUREMENT_ID}/raw_pulses_combined.parquet")
            .filter(pl.col("file") == target_file)
            .collect()
        )

        t = sub_df['time'].to_numpy()
        v = sub_df['voltage'].to_numpy()

        v_min_idx = np.argmin(v)
        t_at_min = t[v_min_idx]
        v_min = v[v_min_idx]
        est_baseline = np.mean(v[:200])
        est_a = abs(v_min - est_baseline)
        span = t[-1] - t[0]

        # Pre-filter
        snr = est_a / (np.std(v[:200]) + 1e-12)
        if snr < 3.0:
            return None

        # Downsample for fitting
        stride = max(1, len(t) // 100)
        t_fit, v_fit = t[::stride], v[::stride]

        p0 = [t_at_min - span * 0.05, est_a, span * 0.15,
              span * 0.01, est_a * 0.3, span * 0.5, est_baseline]
        bounds = (
            [-0.005, 0.001, 1e-6, 1e-7, 0.0, 1e-5, est_baseline - 0.01],
            [0.005, 0.5, span, span * 0.2, est_a * 2.0, span * 5.0, est_baseline + 0.01]
        )

        popt, _ = curve_fit(
            pulse_model, t_fit, v_fit, p0=p0, bounds=bounds,method='trf',
            maxfev=2000, ftol=1e-4, xtol=1e-4, gtol=1e-4
        )
        
        chi2_red = np.sum((v_fit - pulse_model(t_fit, *popt))**2) / (len(t_fit) - len(popt))
        
        if chi2_red > 1e-6:
            return None
        
        return {
            'file': target_file,
            't0': popt[0], 'A': popt[1], 'tau_1': popt[2],
            'tau_2': popt[3], 'B': popt[4], 'tau_3': popt[5],
            'C': popt[6],
            't_start': float(t[0]), 't_end': float(t[-1]), 'n_points': len(t)
        }
    except Exception:
        return None


# ── Stage 2: vectorized integrals, fully in Polars ────────────────────────────
def compute_all_integrals(params: pl.DataFrame, n_points: int = 1000):
    """
    Reconstruct each fitted pulse on an n_points grid and integrate.
    All heavy math stays in numpy (Polars doesn't do broadcasting),
    but Polars handles all I/O, filtering, and the final DataFrame.
    """
    # Pull columns as numpy arrays — zero-copy
    t0 = params['t0'].to_numpy()
    a = params['A'].to_numpy()
    tau_1 = params['tau_1'].to_numpy()
    tau_2 = params['tau_2'].to_numpy()
    b = params['B'].to_numpy()
    tau_3 = params['tau_3'].to_numpy()
    c = params['C'].to_numpy()
    t_start = params['t_start'].to_numpy()
    t_end = params['t_end'].to_numpy()

    # (n × n_points) time grid
    fracs = np.linspace(0, 1, n_points)
    t_grid = t_start[:, None] + fracs[None, :] * (t_end - t_start)[:, None]

    # Vectorized model evaluation
    dt = t_grid - t0[:, None]
    fitted = (c[:, None]
              - (a + b)[:, None] * np.exp(-dt / tau_1[:, None])
              + a[:, None] * np.exp(-dt / tau_2[:, None])
              + b[:, None] * np.exp(-dt / tau_3[:, None]))
    fitted = np.where(dt >= 0, fitted, c[:, None])

    # Integrate negative portion only
    centered = fitted - c[:, None]
    negative_only = np.where(centered < 0, centered, 0.0)
    avg_height = 0.5 * (negative_only[:, :-1] + negative_only[:, 1:])
    integrals = -1.0 * np.sum(avg_height * np.diff(t_grid, axis=1), axis=1)

    # Hand back a Polars DataFrame
    return params.select('file').with_columns(
        pl.Series('integral_Vs', integrals)
    )
    
def compress_to_parquet(MEASUREMENT_ID) -> None:
    file_pattern = f"Data/{MEASUREMENT_ID}/acq*.csv"
    COMPRESSED_PARQUET = f"Data/{MEASUREMENT_ID}/raw_pulses_combined.parquet"
    all_files = glob.glob(file_pattern)

    if not Path(COMPRESSED_PARQUET).exists():
        if not all_files:
            print("No loose acq*.csv files found to process.")
        else:
            print(f"Stage 1: Compressing {len(all_files)} CSVs into a single Parquet file...")
            dfs = []
            for fp in all_files:
                # Quickly read files using Polars
                df = pl.read_csv(
                    fp, skip_rows=10, has_header=False, columns=[0, 1],
                    schema_overrides={'column_1': pl.Float64, 'column_2': pl.Float64}
                ).rename({'column_1': 'time', 'column_2': 'voltage'})
                
                # Tag rows with their originating filename so we can separate them later
                df = df.with_columns(pl.lit(fp).alias('file'))
                dfs.append(df)
                
                print(f"  Processed {len(dfs)}/{len(all_files)} files...", end="\r")
            
            # Merge and save to disk
            combined_df = pl.concat(dfs)
            combined_df.write_parquet(COMPRESSED_PARQUET)
            print(f"Done! Compressed data saved to: {COMPRESSED_PARQUET}")
            
            # ── NEW: ZIP BACKUP AND CLEANUP STEP ──────────────────────────────────
            DATA_DIR = Path(COMPRESSED_PARQUET).parent
            ZIP_FILENAME = DATA_DIR / "raw_csv_backup.zip"
            
            print(f"\nCreating ZIP backup at {ZIP_FILENAME}...")
            with zipfile.ZipFile(ZIP_FILENAME, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
                for idx, file_path in enumerate(all_files, 1):
                    p = Path(file_path)
                    zipf.write(p, arcname=p.name)
                    if idx % 500 == 0:
                        print(f"  Zipped {idx}/{len(all_files)} files...", end="\r")
                        
            print(f"🎉 Backup successful!")
            
            # Double check both files exist and are populated before deleting anything
            if Path(COMPRESSED_PARQUET).stat().st_size > 0 and ZIP_FILENAME.stat().st_size > 0:
                print("\nSafely removing original loose CSV files from workspace...")
                for file_path in all_files:
                    os.remove(file_path)
                print("Cleanup complete! Your folder layout is fully optimized.")
            else:
                print("Warning: File validation failed. Original CSVs were NOT deleted.")
    else:
        print(f"Stage 1 Skipped: {COMPRESSED_PARQUET} already exists (Loose CSVs already archived).")
    
    
def fit_all_files(MEASUREMENT_ID) -> None:
    PARAMS_CSV = f"Data/{MEASUREMENT_ID}/fitted_params.csv"
    COMPRESSED_PARQUET = f"Data/{MEASUREMENT_ID}/raw_pulses_combined.parquet"
    # Load and print the first 5 rows
    df = pl.read_parquet(COMPRESSED_PARQUET)
    print(df.head(5))

    # 1. Get the unique filenames from the parquet file without loading the data
    unique_files = (
        pl.scan_parquet(COMPRESSED_PARQUET)
        .select("file")
        .unique()
        .collect(engine='streaming')
        ["file"]
        .to_list()
    )

    rows = []
    discarded = 0

    with ProcessPoolExecutor() as executor:
        # Pass just the FILEPATH and the TARGET FILENAME (simple strings!) to the workers
        futures = {
            executor.submit(fit_single_file, MEASUREMENT_ID, fp): fp 
            for fp in unique_files
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result is not None:
                rows.append(result)
            else:
                discarded += 1
            if i % 10 == 0:
                print(f"  {i}/{len(unique_files)} — kept {len(rows)}, discarded {discarded}", end='\r')

    params_df = pl.DataFrame(rows)
    params_df.write_csv(PARAMS_CSV)
    print(f"\nStage 2 done — {len(params_df)} fits saved to {PARAMS_CSV}")
    
def compute_integrals(MEASUREMENT_ID) -> None:
    PARAMS_CSV   = f"Data/{MEASUREMENT_ID}/fitted_params.csv"
    INTEGRAL_CSV = f"Data/{MEASUREMENT_ID}/integrals.csv"
    params_df = pl.read_csv(PARAMS_CSV)
    integrals_df = compute_all_integrals(params_df)
    integrals_df.write_csv(INTEGRAL_CSV)
    print(f"Stage 3 done — integrals saved to {INTEGRAL_CSV}")
    
def plot_hist(MEASUREMENT_ID) -> None:
    INTEGRAL_CSV = f"Data/{MEASUREMENT_ID}/integrals.csv"
    import matplotlib.pyplot as plt
    integrals_df = pl.read_csv(INTEGRAL_CSV)
    binsize = np.sqrt(len(integrals_df))
    plt.hist(integrals_df['integral_Vs'], bins=int(binsize), color='blue', alpha=0.7)
    plt.title(f'Energy Spectrum {MEASUREMENT_ID}')
    plt.xlabel('Integral (Vs)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(f"Figures/E_Spectrum_{MEASUREMENT_ID}.pdf")
    plt.show()

    
if __name__ == "__main__":
    
    MEASUREMENT_ID = "Ra226-290526-1"
    
    compress_to_parquet(MEASUREMENT_ID) 
    fit_all_files(MEASUREMENT_ID)
    compute_integrals(MEASUREMENT_ID)
    plot_hist(MEASUREMENT_ID)