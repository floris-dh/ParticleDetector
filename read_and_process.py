import os
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from concurrent.futures import ProcessPoolExecutor


# --- Global Helpers for Multiprocessing ---
def _pulse_model(t, t0, a, tau_1, tau_2, b, tau_3, c) -> np.ndarray:
    with np.errstate(over="ignore"):
        dt = t - t0
        result = (
            c
            - (a + b) * np.exp(-dt / tau_1)
            + a * np.exp(-dt / tau_2)
            + b * np.exp(-dt / tau_3)
        )
    return np.where(dt >= 0, result, c)


def _fit_single_packet(packet):
    """
    Process a single chunk of oscilloscope data to fit the pulse model.
    Expects a tuple of (chunk_id, time_array, voltage_array).
    """
    try:
        chunk_id, t, v = packet

        v_min_idx = np.argmin(v)
        t_at_min = t[v_min_idx]
        v_min = v[v_min_idx]
        est_baseline = np.mean(v[:200])
        est_a = abs(v_min - est_baseline)
        span = t[-1] - t[0]

        # Pre-filter kwalitatief slechte pulsen (SNR check)
        snr = est_a / (np.std(v[:200]) + 1e-12)
        if snr < 3.0:
            return None

        stride = max(1, len(t) // 100)
        t_fit, v_fit = t[::stride], v[::stride]

        p0 = [
            t_at_min - span * 0.05,
            est_a,
            span * 0.15,
            span * 0.01,
            est_a * 0.3,
            span * 0.5,
            est_baseline,
        ]
        bounds = (
            [-0.05, 0, 1e-7, 1e-7, 0.0, 1e-7, est_baseline - 0.1],
            [
                0.05,
                1,
                span * 1.2,
                span * 0.5,
                est_a * 5.0,
                span * 5.0,
                est_baseline + 0.1,
            ],
        )

        popt, _ = curve_fit(
            _pulse_model,
            t_fit,
            v_fit,
            p0=p0,
            bounds=bounds,
            method="trf",
            maxfev=2000,
            ftol=1e-4,
            xtol=1e-4,
            gtol=1e-4,
        )

        chi2_red = np.sum((v_fit - _pulse_model(t_fit, *popt)) ** 2) / (
            len(t_fit) - len(popt)
        )
        if chi2_red > 1e-6:
            return None

        return {
            "chunk_id": chunk_id,
            "t0": popt[0],
            "A": popt[1],
            "tau_1": popt[2],
            "tau_2": popt[3],
            "B": popt[4],
            "tau_3": popt[5],
            "C": popt[6],
            "t_start": float(t[0]),
            "t_end": float(t[-1]),
            "n_points": len(t),
            # Sla ruwe data alleen op voor de eerste paar chunks (i.v.m. geheugenbesparing)
            "_raw_t": t if chunk_id < 100 else None,
            "_raw_v": v if chunk_id < 100 else None,
        }
    except Exception:
        return None


# --- Hoofdklasse voor Pulsverwerking ---
class AlphaPulseProcessor:
    def __init__(
        self,
        csv_file_path,
        output_folder,
        isotope_name="Unknown",
        max_plots=10,
        max_workers=4,
    ):
        """
        Init processor to extract and analyse alpha pulse data from oscilloscope CSV file.
        """
        self.csv_file_path = csv_file_path
        self.output_folder = output_folder
        self.isotope_name = isotope_name
        self.max_plots = max_plots
        self.max_workers = max_workers

        # Intern op te bouwen dataframes
        self.clean_results_df = None
        self.integrals_df = None

    def _require_clean_results(self):
        if self.clean_results_df is None:
            raise RuntimeError("No fitted pulse results are available. Run process_and_fit_pulses() first.")

    def load_and_chunk_data(self):
        """Laadt het ruwe oscilloscoopbestand en identificeert de afzonderlijke pulstraces."""
        print(f"[{self.isotope_name}] Loading and cleaning raw CSV data...")
        os.makedirs(self.output_folder, exist_ok=True)

        df = pl.read_csv(
            self.csv_file_path,
            skip_rows=9,
            has_header=False,
            new_columns=["Time (s)", "Channel 1 (V)"],
            schema_overrides={"Time (s)": pl.Float64, "Channel 1 (V)": pl.Float64},
            comment_prefix="Time",
            null_values=["NA", "null", "NaN"],
            ignore_errors=True,
            infer_schema_length=0,
        ).drop_nulls()

        print(f"  Data successfully loaded. Total rows: {len(df)}")
        print("  Detecting acquisition windows via time jumps...")

        # Chunk-ID bepalen op basis van tijdstappen die terug naar 0 springen
        df_with_chunks = df.with_columns(
            chunk_id=pl.col("Time (s)").diff().fill_null(0.0).lt(0).cum_sum()
        )

        # Sla geoptimaliseerde tussentijdse data op
        df_with_chunks.write_parquet(
            os.path.join(self.output_folder, "processed_data.parquet")
        )

        # Pak data in voor de pool executor
        print("  Grouping signals into memory pools")
        grouped = (
            df_with_chunks.group_by("chunk_id")
            .agg([pl.col("Time (s)"), pl.col("Channel 1 (V)")])
            .sort("chunk_id")
        )

        packets = list(
            zip(
                grouped["chunk_id"].to_list(),
                [a.to_numpy() for a in grouped["Time (s)"]],
                [a.to_numpy() for a in grouped["Channel 1 (V)"]],
            )
        )
        return packets

    def process_and_fit_pulses(self, packets):
        """Start the parallel processes to fit the pulse shapes."""
        total_chunks = len(packets)
        print(
            f"[{self.isotope_name}] Parallel workers starting ({self.max_workers} cores) for {total_chunks} traces..."
        )

        discards = 0

        results_list = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            raw_results = executor.map(_fit_single_packet, packets, chunksize=10)
            for i, res in enumerate(raw_results):
                if res is not None:
                    results_list.append(res)
                elif res is None:
                    discards += 1
                if i % 100 == 0:
                    print(
                        f"  Processed: trace {i} / {total_chunks}, discarded {discards}",
                        end="\r",
                    )

        print(
            f"\n  Processing completed. Successful fits: {len(results_list)} / {total_chunks} windows."
        )

        if not results_list:
            raise RuntimeError(
                "No valid pulse fits were found. Check data quality and fitting parameters."
            )

        results_df = pl.DataFrame(results_list)

        # Genereer controle-plots voor de eerste 'max_plots' succesvolle fits
        self._generate_diagnostic_plots(results_list)

        # Caching arrays weggooien en opslaan als CSV
        self.clean_results_df = results_df.drop(["_raw_t", "_raw_v"]).sort("chunk_id")
        param_path = os.path.join(self.output_folder, "fitted_params.csv")
        self.clean_results_df.write_csv(param_path)
        print(f"  Fit parameters saved to: '{param_path}'")

    def _generate_diagnostic_plots(self, results_list):
        """Internal helper to generate diagnostic plots for the pulse fits."""
        print(
            f"  Generating diagnostic plots for the first {self.max_plots} verified fits..."
        )
        plots_made = 0

        for res in results_list:
            if plots_made >= self.max_plots:
                break
            if res["_raw_t"] is None:
                continue

            c_id = res["chunk_id"]
            t_arr, v_arr = res["_raw_t"], res["_raw_v"]

            plt.figure(figsize=(8, 4))
            plt.plot(
                t_arr,
                v_arr,
                label=f"Scope Data (Chunk {c_id})",
                alpha=0.6,
                color="black",
            )

            fit_v = _pulse_model(
                t_arr,
                res["t0"],
                res["A"],
                res["tau_1"],
                res["tau_2"],
                res["B"],
                res["tau_3"],
                res["C"],
            )

            plt.plot(t_arr, fit_v, label="Model Fit", color="red", linestyle="--")
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage (V)")
            plt.title(f"{self.isotope_name} Diagnostic Pulse Fit - Chunk {c_id}")
            plt.grid(True, alpha=0.3)
            plt.legend(loc="upper right")

            plt.savefig(
                os.path.join(self.output_folder, f"fit_plot_{c_id:04d}.png"),
                dpi=100,
                bbox_inches="tight",
            )
            plt.close()
            plots_made += 1

    def compute_bounded_integrals(self):
        """Compute the analytical integral for the negative pulse."""
        print(f"[{self.isotope_name}] Computing negative pulse integrals...")
        self._require_clean_results()
        assert self.clean_results_df is not None
        clean_results_df = self.clean_results_df

        # Parameters overhevelen naar NumPy arrays voor snelle iteratie
        A = clean_results_df["A"].to_numpy()
        B = clean_results_df["B"].to_numpy()
        tau_1 = clean_results_df["tau_1"].to_numpy()
        tau_2 = clean_results_df["tau_2"].to_numpy()
        tau_3 = clean_results_df["tau_3"].to_numpy()

        num_pulses = len(clean_results_df)
        dt_zero_arr = np.zeros(num_pulses)

        for i in range(num_pulses):
            max_t = 5.0 * max(tau_2[i], tau_3[i])
            t_scan = np.linspace(0.0, max_t, 1000)

            # Evalueer de modelvorm (zonder baseline C) om het nulpunt te scannen
            with np.errstate(over="ignore"):
                f_scan = (
                    -(A[i] + B[i]) * np.exp(-t_scan / tau_1[i])
                    + A[i] * np.exp(-t_scan / tau_2[i])
                    + B[i] * np.exp(-t_scan / tau_3[i])
                )

            # Zoek het eerste punt na de dip waar het signaal weer terugkeert bij de nullijn (>= 0)
            peak_idx = np.argmin(f_scan)
            zero_crossings = np.where(f_scan[peak_idx:] >= 0)[0]

            if len(zero_crossings) > 0:
                dt_zero_arr[i] = t_scan[peak_idx + zero_crossings[0]]
            else:
                dt_zero_arr[i] = max_t

        # dt_zero toevoegen aan Polars en de wiskundige integraal uitrekenen
        results_with_dt = clean_results_df.with_columns(
            pl.Series("dt_zero", dt_zero_arr)
        )

        self.integrals_df = results_with_dt.with_columns(
            pulse_integral=(
                -(pl.col("A") + pl.col("B"))
                * pl.col("tau_1")
                * ((-pl.col("dt_zero") / pl.col("tau_1")).exp() - 1.0)
                + pl.col("A")
                * pl.col("tau_2")
                * ((-pl.col("dt_zero") / pl.col("tau_2")).exp() - 1.0)
                + pl.col("B")
                * pl.col("tau_3")
                * ((-pl.col("dt_zero") / pl.col("tau_3")).exp() - 1.0)
            )
        )

        # Filter negatieve uitschieters / ruis wegschrijven naar 0.0
        self.integrals_df = (
            self.integrals_df.with_columns(
                pulse_integral=pl.when(pl.col("pulse_integral") > 0)
                .then(pl.col("pulse_integral"))
                .otherwise(0.0)
            )
            .select(["chunk_id", "pulse_integral"])
            .sort("chunk_id")
        )

        output_path = os.path.join(self.output_folder, "pulse_integrals.csv")
        self.integrals_df.write_csv(output_path)
        print(f"  Integrals successfully computed and saved to: '{output_path}'")

    def run_pipeline(self):
        """Voert de volledige analysepijplijn chronologisch uit."""
        packets = self.load_and_chunk_data()
        self.process_and_fit_pulses(packets)
        self.compute_bounded_integrals()
        print(f"[{self.isotope_name}] Full pipeline completed successfully!\n")


# =============================================================================
# UITVOERING VAN DE SCRIPT-PIPELINE
# =============================================================================
if __name__ == "__main__":
    # Configuratie-instellingen per isotoop-bestand
    
    plotsize = 10
    
    config = {
        "Am241": {
            "csv": r"Data\Am241-100626-1.csv",
            "output": r"Data\Am241-050626-1_output",
            "plots": plotsize,
        },
        "Pu239": {
            "csv": r"Data\Pu239-050626-1.csv",
            "output": r"Data\Pu239-050626-1_output",
            "plots": plotsize,
        },
        "Ra226": {
            "csv": r"D:\Ra-080626-1.csv",
            "output": r"Data\Ra-080626-1_output",
            "plots": plotsize,
        },
    }

    # Kies hier de gewenste target-bron om te verwerken
    target = "Am241"  # Opties: "Am241", "Pu239", "Ra226"
    run_cfg = config[target]

    # Initialiseer de OOP processor instantie
    processor = AlphaPulseProcessor(
        csv_file_path=run_cfg["csv"],
        output_folder=run_cfg["output"],
        isotope_name=target,
        max_plots=run_cfg["plots"],
        max_workers=14,  # Pas aan naar aantal beschikbare CPU cores
    )

    # Start de verwerking
    processor.run_pipeline()
