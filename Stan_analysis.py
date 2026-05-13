import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count # Voor de extra kracht
from tqdm import tqdm # Voor de voortgangsbalk

# 1. We zetten de analyse-logica in een losse functie
def process_single_file(file_path):
    try:
        # Je originele import_data logica
        data = pd.read_csv(file_path, skiprows=9)
        times = data.iloc[:, 0]
        voltage = data.iloc[:, 1]
        #Gradient check
        dydx = np.gradient(voltage)
        min_gradient = dydx.min()
        
        # Oliver gebruikt -20 als grens. 
        # Alles wat 'vlakker' is dan -20 wordt genegeerd.
        if min_gradient > -20: 
            return None # Te flauw, waarschijnlijk ruis
        # De integraal berekenen
        integral = np.trapezoid(voltage, times)
        return integral
    except Exception as e:
        return None

if __name__ == "__main__":
    start_time = time.time()
    
    # 2. Map configureren
    data_folder = "Data"
    # We maken een lijst van alle bestanden die met 'acq' beginnen en eindigen op '.csv'
    all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith("acq") and f.endswith(".csv")]
    
    num_files = len(all_files)
    print(f"Gevonden: {num_files} bestanden.")
    print(f"Analyse starten op {cpu_count()} CPU kernen...")

    # 3. Multiprocessing Pool starten
    # Dit verdeelt de lijst 'all_files' over al je processor-kernen
    with Pool(processes=cpu_count()) as pool:
        # imap_unordered is vaak sneller; tqdm laat een mooie balk zien
        integral_list = list(tqdm(pool.imap_unordered(process_single_file, all_files), total=num_files))

    # Verwijder eventuele foutmeldingen (None resultaten)
    integral_list = [i for i in integral_list if i is not None]

    # --- De rest van je plot-logica ---
    if len(integral_list) > 0:
        plt.figure(figsize=(10, 6))
        counts, bins = np.histogram(
            integral_list,
            bins=int(np.sqrt(len(integral_list))),
            density=True,
        )
        plt.stairs(counts, bins, edgecolor="black", alpha=0.7)
        plt.xlabel("Integral of Voltage over Time")
        plt.ylabel("Density")
        plt.title(f"Histogram of Integrals ({len(integral_list)} files)")
        
        # Zorg dat de map Figures bestaat
        os.makedirs("Figures", exist_ok=True)
        plt.savefig("Figures/integral_histogram.png")
        
        end_time = time.time()
        print(f"\nKlaar! Verwerkt: {len(integral_list)} bestanden in {end_time - start_time:.2f} seconden.")
    else:
        print("Geen data gevonden om te plotten.")