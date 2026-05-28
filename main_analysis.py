import numpy as np
import polars as pl
from scipy.optimize import curve_fit


# ── Model ─────────────────────────────────────────────────────────────────────
def pulse_model(t, t0, a, tau_1, tau_2, b, tau_3, c):
    dt = t - t0
    result = (c - (a + b) * np.exp(-dt / tau_1)
              + a * np.exp(-dt / tau_2)
              + b * np.exp(-dt / tau_3))
    return np.where(dt >= 0, result, c)


# ── Stage 1 worker: Polars reads the CSV, scipy fits it ───────────────────────
def fit_single_file(parquet_path, target_file):
    """
    Receives a preloaded slice of data for a specific file, avoiding disk I/O.
    """
    # noinspection PyBroadException
    try:
        sub_df = (
            pl.scan_parquet(parquet_path)
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
        stride = max(1, len(t) // 300)
        t_fit, v_fit = t[::stride], v[::stride]

        p0 = [t_at_min - span * 0.05, est_a, span * 0.15,
              span * 0.01, est_a * 0.3, span * 0.5, est_baseline]
        bounds = (
            [t[0], 0.001, 1e-6, 1e-7, 0.0, 1e-5, est_baseline - 0.01],
            [t_at_min, 0.1, span, span * 0.2, est_a * 2.0, span * 5.0, est_baseline + 0.01]
        )

        popt, _ = curve_fit(
            pulse_model, t_fit, v_fit, p0=p0, bounds=bounds,method='trf',
            maxfev=2000, ftol=1e-4, xtol=1e-4, gtol=1e-4
        )

        residuals = v_fit - pulse_model(t_fit, *popt)
        chi2_red = np.sum(residuals ** 2) / (len(v_fit) - len(popt))
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
def compute_all_integrals(params: pl.DataFrame, n_points: int = 1000) -> pl.DataFrame:
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