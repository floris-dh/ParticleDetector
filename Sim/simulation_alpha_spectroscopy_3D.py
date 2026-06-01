import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
import os
import time
import multiprocessing as mp

""" We made quite a few assumptions 
 1 stopping power follows a power law (gefit op NIST data) with a fit of a * E^b, 
 also with a certain enery range that can be altered if relevant.
 2 For energy straggling is moddeled as a gaussian which simulates collisions with electrons,
 beacause alpha paricles are much heavier than electrons, which whty we took the value of 0.01 for the statistical noise. 
 3 for the use of Molière theory we approximated the constans as 0.01
 4 we assume the alpha particle travels in homogeneous air with constant density, which is a good approximation for short distances.
 We also make a small angle approximation for the scattering, this is a valid approximation for small angles, 
 therefore I recommend not increasing the distance above 30 mm. All because the scattering angle is inversely proportional to the energy,
 which decreases with distance.
 """

E_START = 5.5           # MeV 
DISTANCE = 20.0           # mm tot detector, would not recommend going above 30 mm explenanation ^^^^ under assumptions
N_PARTICLES = int(1e5)   
STEP_SIZE = 0.05          # mm
DENSITY_AIR = 0.001225    # g/cm^3
# set to True if you want to use all CPU cores for processing
use_multiprocessing = True 

def power_law(E, a, b):
    return a * E**b

def simulation_worker(params):
    """Simuleert één deeltje in 3D met scattering."""
    E_start, distance, step_size, a, b = params
    E = E_start
    x, y, z = 0.0, 0.0, 0.0
    ux, uy, uz = 1.0, 0.0, 0.0 # x is start direction
    
    while E > 0.05 and x < distance:
        st_pow = a * (E**b)
        # Energieverlies + Straggling
        dE = st_pow * step_size + np.random.normal(0, 0.01 * st_pow)
        E -= dE
        
        # Scattering hoek berekenen
        theta_std = 0.01 * (st_pow / max(E, 0.1)) 
        theta = np.random.normal(0, theta_std)
        phi = np.random.uniform(0, 2 * np.pi)
        
        # Richting aanpassen (vector rotatie benadering)
        ux += theta * np.cos(phi)
        uy += theta * np.sin(phi)
        uz += theta * np.sin(phi)
        
        norm = np.sqrt(ux**2 + uy**2 + uz**2)
        ux, uy, uz = ux/norm, uy/norm, uz/norm
        
        x += ux * step_size
    return max(E, 0)

def main():
    
    # Fit
    base_dir = os.path.dirname(__file__)
    file_path = os.path.join(base_dir, 'Data', 'NIST_data.csv')
    
    df = pd.read_csv(file_path, header=None, sep=r'\s+', names=['Energy', 'MassStpPow'])
    df_filtered = df[(df['Energy'] >= 0.5) & (df['Energy'] <= 6.0)].copy()
    y_mm = df_filtered['MassStpPow'] * DENSITY_AIR * 0.1 # correcting units
    
    popt, _ = curve_fit(power_law, df_filtered['Energy'], y_mm)
    a, b = popt
    
    # sim
    start_time = time.time()
    num_cores = mp.cpu_count()
    sim_params = (E_START, DISTANCE, STEP_SIZE, a, b)
    
    energies_at_detector = []
    
    
    if use_multiprocessing:
        print(f"Use multicore processing for {N_PARTICLES} particles...")
        with mp.Pool(num_cores) as pool:
        
            results = pool.imap_unordered(simulation_worker, [sim_params] * N_PARTICLES, chunksize=500)
        
            for i, result in enumerate(results):
                energies_at_detector.append(result)
            
                # progress updater
                if (i + 1) % (N_PARTICLES // 100) == 0:
                    percent = (i + 1) / N_PARTICLES * 100
                    print(f"\rVoortgang: [{'#' * int(percent // 2)}{'.' * (50 - int(percent // 2))}] {percent:.0f}%", end="")
    else:
        print(f"Single core processing for {N_PARTICLES} particles...")
        for i in range(N_PARTICLES):
            result = simulation_worker(sim_params)
            energies_at_detector.append(result)
            
            # progress updater
            if (i + 1) % (N_PARTICLES // 100) == 0:
                percent = (i + 1) / N_PARTICLES * 100
                print(f"\rVoortgang: [{'#' * int(percent // 2)}{'.' * (50 - int(percent // 2))}] {percent:.0f}%", end="")
    duration = time.time() - start_time
    print(f"\n\nKlaar Totale tijd: {duration:.2f} seconden.")
    print(f"Gemiddelde snelheid: {N_PARTICLES / duration:.0f} deeltjes/sec.")

    # plot
    plt.figure(figsize=(10, 6))
    plt.hist(energies_at_detector, bins=100, color='royalblue', edgecolor='black', alpha=0.8)
    plt.title(f"Simulation at {DISTANCE} mm")
    plt.xlabel("Energy [MeV]")
    plt.ylabel("N particles")
    plt.grid(axis='y', alpha=0.3)
    #plt.show()
    plt.savefig("Figures/3D_simulation.png")
if __name__ == '__main__':
    main()