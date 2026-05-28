import numpy as np
import polars as pl
from scipy.optimize import curve_fit
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Model ─────────────────────────────────────────────────────────────────────
def pulse_model(t, t0, A, tau_1, tau_2, B, tau_3, C):
    dt = t - t0
    result = (C - (A + B) * np.exp(-dt / tau_1)
                + A       * np.exp(-dt / tau_2)
                + B       * np.exp(-dt / tau_3))
    return np.where(dt >= 0, result, C)

# ── Stage 1 worker: Polars reads the CSV, scipy fits it ───────────────────────
def fit_single_file(parquet_path, target_file):
    """
    Receives a pre-loaded slice of data for a specific file, avoiding disk I/O.
    """
    try:
        sub_df = (
            pl.scan_parquet(parquet_path)
            .filter(pl.col("file") == target_file)
            .collect()
        )
        
        t = sub_df['time'].to_numpy()
        v = sub_df['voltage'].to_numpy()

        v_min_idx    = np.argmin(v)
        t_at_min     = t[v_min_idx]
        v_min        = v[v_min_idx]
        est_baseline = np.mean(v[:200])
        est_A        = abs(v_min - est_baseline)
        span         = t[-1] - t[0]

        # Pre-filter
        snr = est_A / (np.std(v[:200]) + 1e-12)
        if snr < 3.0:
            return None

        # Downsample for fitting
        stride       = max(1, len(t) // 300)
        t_fit, v_fit = t[::stride], v[::stride]

        p0 = [t_at_min - span * 0.05, est_A, span * 0.15,
                span * 0.01, est_A * 0.3, span * 0.5, est_baseline]
        bounds = (
            [t[0],     0.001, 1e-6, 1e-7, 0.0,         1e-5,          est_baseline - 0.01],
            [t_at_min, 0.1,   span, span * 0.2, est_A * 2.0, span * 5.0, est_baseline + 0.01]
        )

        popt, _ = curve_fit(
            pulse_model, t_fit, v_fit,
            p0=p0, bounds=bounds, method='trf',
            maxfev=2000, ftol=1e-4, xtol=1e-4, gtol=1e-4
        )

        residuals = v_fit - pulse_model(t_fit, *popt)
        chi2_red  = np.sum(residuals ** 2) / (len(v_fit) - len(popt))
        if chi2_red > 1e-6:
            return None

        return {
            'file':    target_file,
            't0':      popt[0], 'A':     popt[1], 'tau_1': popt[2],
            'tau_2':   popt[3], 'B':     popt[4], 'tau_3': popt[5],
            'C':       popt[6],
            't_start': float(t[0]), 't_end': float(t[-1]), 'n_points': len(t)
        }
    except Exception:
        return None

# ── Stage 2: vectorized integrals, fully in Polars ────────────────────────────
def compute_all_integrals(params: pl.DataFrame, n_points: int = 1000) -> pl.DataFrame:
    """
    Reconstruct each fitted pulse on an n_points grid and integrate.
    All heavy math stays in numpy (Polars doesn't do broadcasting),
    but Polars handles all I/O, filtering, and the final DataFrame.
    """
    # Pull columns as numpy arrays — zero-copy
    t0    = params['t0'].to_numpy()
    A     = params['A'].to_numpy()
    tau_1 = params['tau_1'].to_numpy()
    tau_2 = params['tau_2'].to_numpy()
    B     = params['B'].to_numpy()
    tau_3 = params['tau_3'].to_numpy()
    C     = params['C'].to_numpy()
    t_start = params['t_start'].to_numpy()
    t_end   = params['t_end'].to_numpy()

    n = len(params)

    # (n × n_points) time grid
    fracs  = np.linspace(0, 1, n_points)
    t_grid = t_start[:, None] + fracs[None, :] * (t_end - t_start)[:, None]

    # Vectorized model evaluation
    dt      = t_grid - t0[:, None]
    fitted  = (C[:, None]
               - (A + B)[:, None] * np.exp(-dt / tau_1[:, None])
               + A[:, None]       * np.exp(-dt / tau_2[:, None])
               + B[:, None]       * np.exp(-dt / tau_3[:, None]))
    fitted  = np.where(dt >= 0, fitted, C[:, None])

    # Integrate negative portion only
    centered      = fitted - C[:, None]
    negative_only = np.where(centered < 0, centered, 0.0)
    avg_height    = 0.5 * (negative_only[:, :-1] + negative_only[:, 1:])
    integrals     = -1.0 * np.sum(avg_height * np.diff(t_grid, axis=1), axis=1)

    # Hand back a Polars DataFrame
    return params.select('file').with_columns(
        pl.Series('integral_Vs', integrals)
    )

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    PARAMS_CSV   = "Data/Radium2705/fitted_params.csv"
    INTEGRAL_CSV = "Data/Radium2705/integrals.csv"
    file_pattern = "Data/Radium2705/acq*.csv"

    all_files = glob.glob(file_pattern)
    if not all_files:
        print("No files found.")
    else:
        # ── STAGE 1: parallel fitting ──────────────────────────────────────────
        print(f"Stage 1: fitting {len(all_files)} files...")
        rows      = []
        discarded = 0

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(fit_single_file, fp): fp for fp in all_files}
            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result is not None:
                    rows.append(result)
                else:
                    discarded += 1
                if i % 10 == 0:
                    print(f"  {i}/{len(all_files)} — kept {len(rows)}, "
                          f"discarded {discarded}", end='\r')

        # Polars builds the DataFrame from dicts in one shot — no concat loop
        params_df = pl.DataFrame(rows)
        params_df.write_csv(PARAMS_CSV)
        print(f"\nStage 1 done — {len(params_df)} fits saved to {PARAMS_CSV}")

        # ── STAGE 2: vectorized integrals ──────────────────────────────────────
        print("Stage 2: computing integrals...")
        params_df    = pl.read_csv(PARAMS_CSV)          # can be run standalone
        integrals_df = compute_all_integrals(params_df)
        integrals_df.write_csv(INTEGRAL_CSV)
        print(f"Stage 2 done — {len(integrals_df)} integrals saved to {INTEGRAL_CSV}")

        # ── Plot ───────────────────────────────────────────────────────────────
        integrals = integrals_df['integral_Vs'].to_numpy()
        counts, bins = np.histogram(integrals, bins=150)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.stairs(counts, bins, fill=True, color='steelblue', alpha=0.7)
        plt.xlabel('Integral (V·s)')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Pulse Integrals (n={len(integrals)})')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()