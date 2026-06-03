import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from multiprocessing import Pool, cpu_count 
from tqdm import tqdm # Voor de voortgangsbalk


use_multiprocessing = False # set to True if you want to use all CPU cores for processing
gradient_check = False # set to False if you want to skip the gradient check
# We zetten de analyse-logica in een losse functie
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
        if gradient_check:
            if min_gradient > -20: 
                return None # waarschijnlijk ruis
       
        # De integraal berekenen
        integral = np.trapezoid(np.abs(voltage), times)
        return integral
    except Exception as e:
        return None

def calculate_FWHM(data, bins):
    # Zoek de maximale waarde en de helft daarvan
    max_index = np.argmax(counts)
    max_value = counts[max_index]
    half_max = max_value / 2.0

    # Zoek de linkerkant (loop vanaf de piek terug naar het begin)
    left_bin = max_index
    while left_bin > 0 and counts[left_bin] > half_max:
        left_bin -= 1

    # Zoek de rechterkant (loop vanaf de piek naar het einde)
    right_bin = max_index
    while right_bin < len(counts) - 1 and counts[right_bin] > half_max:
        right_bin += 1
    
    fwhm  = bin_centers[right_bin] - bin_centers[left_bin]
    return fwhm, bin_centers[max_index]

if __name__ == "__main__":
    start_time = time.time()
    
    # Map configureren
    data_folder = "Data"
    # we make a list of all files starting with "acq" and ending with ".csv" in the data_folder
    all_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.startswith("acq") and f.endswith(".csv")]
    
    num_files = len(all_files)
    print(f"Gevonden: {num_files} bestanden.")
    
    

    # Multiprocessing Pool starten
    # Dit verdeelt de lijst 'all_files' over al de processor-kernen
    if use_multiprocessing:
        print(f"Analyse starten op {cpu_count()} CPU kernen...")
        with Pool(processes=cpu_count()) as pool:
            # imap_unordered is vaak sneller; tqdm laat een mooie balk zien
            integral_list = list(tqdm(pool.imap_unordered(process_single_file, all_files), total=num_files))
    # Zonder multiprocessing, gewoon sequentieel
    #tqdm is progressie balk
    else:
        print("Analyse starten zonder multiprocessing...")
        integral_list = []
        for file in tqdm(all_files, total=num_files):
            integral = process_single_file(file)
            integral_list.append(integral)
    # Verwijder eventuele foutmeldingen 
    integral_list = [i for i in integral_list if i is not None]

    # De rest van de analyse en plot
    if len(integral_list) > 0:
        
        num_bins = int(np.sqrt(len(integral_list)))
        
        counts, bin_edges = np.histogram(integral_list, bins=num_bins, density=True,)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        fwhm_waarde, piek_positie = calculate_FWHM(counts, bin_centers)
        waarde_per_bin = (bin_edges[-1] - bin_edges[0]) / num_bins
        sigma_bins = fwhm_waarde / (waarde_per_bin * (2 * np.sqrt(2 * np.log(2))))

        #do the deconvolution
        respons = gauss(len(counts), sigma_bins)
        deconvolution_counts = richard_lucy_deconvolution(counts, respons, iterations=20)
        
        
        plt.figure(figsize=(10, 6))
        plt.stairs(counts, bin_edges, edgecolor="black")
        plt.step(bin_centers, deconvolution_counts, where='mid', color='red', label='Deconvolved')
        plt.xlabel("Integral of Voltage over Time")
        plt.ylabel("Density")
        plt.title(f"Histogram of Integrals ({len(integral_list)} files)")
        plt.grid(True)
        plt.legend()
        # Zorg dat de map Figures bestaat
        os.makedirs("Figures", exist_ok=True)
        plt.savefig("Figures/deconvolution.png")
        print(f"Peak found at: {piek_positie}")
        end_time = time.time()
        print(f"\nKlaar Verwerkt: {len(integral_list)} bestanden in {end_time - start_time:.2f} seconden.")
    else:
        print("Geen data gevonden om te plotten.")