import os
import pandas as pd
import polars as pl


def extract_full_pulse_data(folder_path):
    all_pulse_dfs = []

    for file_name in sorted(os.listdir(folder_path)):
        if file_name.endswith(".csv") or file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            # --- 1. Comprehensive Metadata Header Parsing ---
            file_meta = {}
            header_line_count = 0
            
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("#"):
                        header_line_count += 1
                        clean_line = line.lstrip("#").strip()
                        
                        if ":" in clean_line:
                            k, v = clean_line.split(":", 1)
                            # Handle duplicate keys cleanly if they appear (e.g., multiple "Voltage" lines)
                            key_name = k.strip()
                            if key_name in file_meta:
                                key_name = f"{key_name}_{header_line_count}"
                            
                            file_meta[key_name] = v.strip()
                        else:
                            # If a metadata line doesn't have a colon, keep it as a note column
                            file_meta[f"Header_Line_{header_line_count}"] = clean_line
                    else:
                        break

            if not file_meta:
                continue

            # Convert metadata dict into a 1-row Polars DataFrame
            df_meta_row = pl.DataFrame([file_meta])

            # --- 2. Read Signals Safely (Using Pandas engine to handle blank lines) ---
            df_pandas = pd.read_csv(file_path, comment="#", skip_blank_lines=True)

            # Clean column whitespace ("Channel 1 (V) " -> "Channel 1 (V)")
            df_pandas.columns = [col.strip() for col in df_pandas.columns]

            # Convert to Polars
            df_signals = pl.from_pandas(df_pandas)

            # Identify the columns dynamically
            time_col = "Time (s)" if "Time (s)" in df_signals.columns else df_signals.columns[0]
            volt_col = (
                "Channel 1 (V)"
                if "Channel 1 (V)" in df_signals.columns
                else df_signals.columns[1]
            )

            # Aggregate columns into horizontal lists/arrays
            df_arrays = df_signals.select(
                [
                    pl.col(time_col).alias("time_array"),
                    pl.col(volt_col).alias("voltage_array"),
                ]
            ).select([pl.all().implode()])

            # --- 3. Combine Metadata with Signal Arrays ---
            df_combined = pl.concat([df_meta_row, df_arrays], how="horizontal").with_columns(
                pl.lit(file_name).alias("Source_File")
            )

            all_pulse_dfs.append(df_combined)

    if not all_pulse_dfs:
        print("No valid data files found.")
        return pl.DataFrame()

    # how="diagonal" is critical here because different files might have 
    # slightly variations in metadata, filling missing fields with nulls safely.
    return pl.concat(all_pulse_dfs, how="diagonal")


# --- Execution ---
df_master = extract_full_pulse_data("RawData")

if not df_master.is_empty():
    df_master.write_parquet("complete_pulses.parquet")
    print("Data successfully processed! Preview of all columns:")
    print(df_master.columns)