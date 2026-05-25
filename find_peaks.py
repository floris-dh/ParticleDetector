import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def find_min_in_data(file_path):
    # Load data from CSV file
    data = pd.read_csv(file_path, skiprows=10, header=None)  
    data.columns = ['times', 'voltage']  
    
    voltage_data = data['voltage'].values
    return np.min(voltage_data), voltage_data

minima_list = []
i = 1

# Enable interactive mode for live plotting
plt.ion()
fig, ax = plt.subplots()

print("Waiting for new data. Press Ctrl+C to stop and save the final plot...")

try:
    while True:
        filepath = f"RawData/acq{i:04d}.csv"
        
        # Check if the next file in the sequence exists
        if os.path.exists(filepath):
            try:
                min_voltage, voltage_data = find_min_in_data(filepath)
                minima_list.append(np.abs(min_voltage))
                
                print(f"Processed: {filepath} | Min Voltage: {min_voltage:.3e} V", end="\r")
                
                ax.plot(voltage_data, label=f'File {i:04d}')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Voltage (V)')
                ax.set_title(r'Voltage vs Time for $^{241}$Am')
                ax.legend()
                plt.show()
                
                # Update the live histogram every 50 files (adjust as needed to prevent lag)
                # if i % 50 == 0:
                #     ax.clear()
                #     counts, bins = np.histogram(minima_list, bins=50)
                #     ax.stairs(counts, bins)
                #     ax.set_xlabel('Energy (AU)')
                #     ax.set_ylabel('Frequency')
                #     ax.set_title(r'Energy Spectrum of $^{241}$Am')
                #     plt.pause(0.01) # Briefly pause to draw the updated plot
                
                # Increment to look for the next file
                i += 1
                
            except Exception as e:
                # If the file is caught mid-write, it might be empty or locked.
                # Wait briefly and try reading the same file again.
                time.sleep(0.1)
        else:
            # The file isn't there yet. Sleep for 0.5 seconds and check again.
            time.sleep(0.5)

except KeyboardInterrupt:
    # This block runs when you hit Ctrl+C in the terminal
    print("\n\nData collection stopped by user.")
    
    # Generate and save the final, complete histogram
    if minima_list:
        plt.ioff() # Turn off interactive mode
        plt.figure() # Create a clean figure for the final save
        counts, bins = np.histogram(minima_list, bins=50)
        plt.stairs(counts, bins)
        plt.xlabel('Energy (AU)')
        plt.ylabel('Frequency')
        plt.title(r'Energy Spectrum of $^{241}$Am')
        plt.savefig("minima_histogram.png")
        print("Final histogram saved to 'minima_histogram.png'.")
        plt.show()
    else:
        print("No data was collected.")