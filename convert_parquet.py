import os
import glob
import polars as pl
import numpy as np

def parse_waveforms_csv_to_dataframe(csv_path):
    sample_rate = 4e6
    num_samples = 8192
    
    # 1. Lees de metadata uit de header
    with open(csv_path, 'r') as f:
        for _ in range(15):
            line = f.readline()
            if not line: break
            if line.startswith("#Sample rate:"):
                try: sample_rate = float(line.split(":")[1].strip().replace("Hz", ""))
                except: pass
            elif line.startswith("#Samples:"):
                try: num_samples = int(line.split(":")[1].strip())
                except: pass

    try:
        # 2. Lees het bestand regel voor regel
        with open(csv_path, 'r') as f:
            lines = f.readlines()
        
        voltages_list = []
        for line in lines:
            # Sla commentaarregels en volledig lege regels over
            if line.startswith("#") or not line.strip():
                continue
            
            # FIX: Splits specifiek op de komma die aan het begin van de regel staat
            parts = line.strip().split(",")
            
            # Neem het laatste deel van de splitsing (dit is altijd het getal)
            val_str = parts[-1].strip()
            
            if not val_str:
                continue
                
            try:
                # Zet de string rechtstreeks om naar een Float (Python herkent wetenschappelijke notatie zoals e-05 automatisch!)
                val = float(val_str)
                voltages_list.append(val)
            except ValueError:
                # Sla eventuele tekstkoppen (headers) veilig over
                continue
                
        voltages = np.array(voltages_list, dtype=np.float32)
        
        # DEBUG PRINTER (Alleen voor het allereerste bestand ter controle)
        if not hasattr(parse_waveforms_csv_to_dataframe, "has_printed_stats"):
            print("--- STATISTIEKEN NA NIEUWE PARSING ---")
            print(f"Aantal succesvol geconverteerde getallen: {len(voltages)}")
            print(f"Eerste 5 getallen: {voltages[:5]}")
            print(f"Aantal exacte nullen: {np.sum(voltages == 0.0)}")
            print("------------------------------------------\n")
            parse_waveforms_csv_to_dataframe.has_printed_stats = True
            
    except Exception as e:
        print(f"Fout bij handmatige verwerking van {os.path.basename(csv_path)}: {e}")
        return None, None
                
    # 3. Zorg voor de exacte lengte (8192 samples per puls)
    if len(voltages) < num_samples:
        padded = np.zeros(num_samples, dtype=np.float32)
        padded[:len(voltages)] = voltages
        voltages = padded
    else:
        voltages = voltages[:num_samples]
        
    return voltages, sample_rate
def convert_and_merge_waveforms(csv_dir, parquet_output_dir, output_filename):
    print(f"Scanning for WaveForms CSV files in: {csv_dir}")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        print(f"No CSV files found in {csv_dir}.")
        return
        
    print(f"Found {len(csv_files)} files. Converting...")
    pulse_dict = {}
    shared_sample_rate = None
    
    for idx, csv_path in enumerate(csv_files):
        voltages, sample_rate = parse_waveforms_csv_to_dataframe(csv_path)
        if voltages is None or np.all(voltages == 0.0):
            continue # Sla lege/mislukte bestanden over
            
        if shared_sample_rate is None:
            shared_sample_rate = sample_rate
            
        pulse_dict[f"pulse_{idx}"] = voltages
        
        if (idx + 1) % 250 == 0 or (idx + 1) == len(csv_files):
            print(f"Processed {idx + 1}/{len(csv_files)} files...")

    if pulse_dict:
        first_key = list(pulse_dict.keys())[0]
        actual_length = len(pulse_dict[first_key])
        dt = 1.0 / shared_sample_rate
        time_axis = (np.arange(0, actual_length) * dt).astype(np.float32)
        
        final_dict = {"time": time_axis}
        final_dict.update(pulse_dict)
        
        combined_df = pl.DataFrame(final_dict)
        os.makedirs(parquet_output_dir, exist_ok=True)
        target_path = os.path.join(parquet_output_dir, output_filename)
        
        print(f"Writing structure ({combined_df.shape[1]-1} pulses, {combined_df.shape[0]} samples) to Parquet...")
        combined_df.write_parquet(target_path, compression="snappy")
        print("Conversion completed successfully!")
    else:
        print("ERROR: No valid data extracted (all matrices resolved to zero). Check CSV formatting.")

if __name__ == "__main__":
    csv_ra226_source = r"C:\Users\Nats91\Particle_Detector_PC\ParticleDetector\Data\RawData_Ra226\Ra226-290526"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_ra226_target = os.path.join(script_dir, "RawData_Ra226")
    
    if os.path.exists(csv_ra226_source):
        convert_and_merge_waveforms(csv_ra226_source, parquet_ra226_target, "combined_raw_ra226.parquet")
    else:
        print(f"ERROR: Source folder not found: {csv_ra226_source}")