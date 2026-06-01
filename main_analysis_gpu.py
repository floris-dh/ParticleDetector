import os
import glob
import numpy as np
import polars as pl
import torch
from pathlib import Path

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
# Set device to GPU (CUDA or ROCm) if available, otherwise fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================================================================

def fit_pulses_gpu(time_array, voltage_matrix):
    """
    Fits raw waveforms by correcting the baseline and integrating the 
    absolute deviation, ensuring both positive and negative pulses are captured.
    """
    v_tensor = torch.tensor(
        voltage_matrix,
        dtype=torch.float32,
        device=DEVICE
    )

    dt = time_array[1] - time_array[0]


    baseline = torch.mean(v_tensor[:, :200], dim=1, keepdim=True)
    v_corrected = v_tensor - baseline

    # positieve bijdragen leveren aan de integraal
    absolute_pulses = torch.abs(v_corrected)

    integrals_tensor = torch.sum(absolute_pulses, dim=1) * dt

    return integrals_tensor.cpu().numpy()

def process_measurement_single_file(parquet_dir, file_name, output_csv_path):
    """Loads a single combined parquet file, forces float datatypes, fits via GPU, and exports integrals."""
    file_path = os.path.join(parquet_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"WARNING: Combined Parquet file not found: {file_path}")
        return False
        
    print(f"Processing single unified Parquet dataset: {file_name}")
    df = pl.read_parquet(file_path)
    
    # Extract time column name dynamically (regardless of case)
    time_col = None
    for col in df.columns:
        if "time" in col.lower() or col.lower() == "t":
            time_col = col
            break
            
    if time_col is None:
        print(f"ERROR: Could not find time column in {file_name}")
        return False
        
    # Isolate pulse column names
    pulse_columns = [col for col in df.columns if col != time_col]
    
    # FIX: Explicitly cast all pulse data to Float32 to bypass any 'Object' datatype errors
    print("Enforcing strict Float32 numeric casting over all waveform channels...")
    df_numeric = df.with_columns([
        pl.col(col).cast(pl.Float32, strict=False).fill_null(0.0) for col in pulse_columns
    ])
    
    # Extract time vector and waveform matrix safely
    time_data = df_numeric[time_col].cast(pl.Float32).to_numpy()
    voltage_data = df_numeric[pulse_columns].to_numpy().T # Transpose for row-wise pulses (pulses, samples)
    
    # Execute high-performance GPU fitting over the entire dataset at once
    print(f"Executing GPU fitting over {voltage_data.shape[0]} pulses ({voltage_data.shape[1]} samples each)...")
    integrals = fit_pulses_gpu(time_data, voltage_data)
    print("min =", np.min(integrals))
    print("max =", np.max(integrals))
    print("mean =", np.mean(integrals))
    print("std =", np.std(integrals))
    # Save extracted integrals using Polars
    output_df = pl.DataFrame({"integral_Vs": integrals})
    output_df.write_csv(output_csv_path)
    print(f"Successfully saved calculated integrals to: {output_csv_path}")
    return True

if __name__ == "__main__":
    print(f"Using hardware acceleration device: {DEVICE}")
    
    # De basismap van je project
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Het centrale pad naar jouw Data map voor de CSV-outputs
    data_dir = os.path.join(script_dir, "Data")
    os.makedirs(data_dir, exist_ok=True)
    
    # 1. Process Americium-241 Calibration Data
    am241_raw_dir = os.path.join(data_dir, "RawData_Am241")
    am241_output = os.path.join(data_dir, "integrals_Am241.csv")
    
    if os.path.exists(am241_raw_dir):
        print("Starting Americium-241 pulse processing...")
        process_measurement_single_file(am241_raw_dir, "raw_pulses_combined.parquet", am241_output)
    else:
        print(f"Skipping Am241: Directory {am241_raw_dir} does not exist.")
        
    # 2. Process Target Radium-226 Data
    # AANGEPAST: Dit verwijst nu direct naar de map in je hoofdmap (zonder 'Data')
    ra226_raw_dir = os.path.join(script_dir, "RawData_Ra226")
    ra226_output = os.path.join(data_dir, "integrals_Ra226.csv")
    
    if os.path.exists(ra226_raw_dir):
        print("Starting Radium-226 pulse processing from single combined file...")
        process_measurement_single_file(ra226_raw_dir, "combined_raw_ra226.parquet", ra226_output)
    else:
        print(f"Skipping Ra226: Directory {ra226_raw_dir} does not exist.")
        
    print("Main GPU analysis pipeline tasks completed.")
    print("Integral statistics:")
