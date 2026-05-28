import numpy as np
import polars as pl
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor, as_completed

# ── Model ─────────────────────────────────────────────────────────────────────
def pulse_model(t, t0, A, tau_1, tau_2, B, tau_3, C):
    dt = t - t0
    result = (C - (A + B) * np.exp(-dt / tau_1)
                + A       * np.exp(-dt / tau_2)
                + B       * np.exp(-dt / tau_3))
    return np.where(dt >= 0, result, C)

# ── Stage 1 worker: Polars reads the CSV, scipy fits it ───────────────────────
def fit_single_file(filepath):
    try:
        # Polars C engine is faster than pandas for plain numeric CSVs
        data = pl.read_csv(
            filepath,
            skip_rows=10,
            has_header=False,
            columns=[0, 1],                    # only read the two columns we need
            schema_overrides={'column_1': pl.Float64, 'column_2': pl.Float64}
        )
        t = data['column_1'].to_numpy()        # zero-copy into numpy for scipy
        v = data['column_2'].to_numpy()

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
        stride      = max(1, len(t) // 300)
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
            'file':    filepath,
            't0':      popt[0], 'A':     popt[1], 'tau_1': popt[2],
            'tau_2':   popt[3], 'B':     popt[4], 'tau_3': popt[5],
            'C':       popt[6],
            't_start': float(t[0]), 't_end': float(t[-1]), 'n_points': len(t)
        }

    except Exception:
        return None