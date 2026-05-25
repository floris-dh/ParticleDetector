import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data_dir = "RawData"
output_path = "ProcessedData/pulses.h5"

# Ensure output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

if os.path.exists(output_path):
    raise FileExistsError(
        f"{output_path} already exists. Please remove it before running the script."
    )


def read_file(file_path):
    # Skip rows safely; using engine='python' or clear headers handles empty data better
    data = pd.read_csv(file_path, skiprows=11, header=None)
    times = data.iloc[:, 0].to_numpy()
    voltage = data.iloc[:, 1].to_numpy()
    return times, voltage


def compute_energy(voltage, time):
    max_voltage = np.max(voltage)
    if max_voltage <= 0.4:
        return None  # Discard pulses below threshold

    # Trapz computes energy from the area under the curve
    energy = np.trapezoid(voltage, time)

    # Return dictionary to append cleanly in a list later (saves immense RAM/Time)
    return {
        "Voltage": list(voltage),
        "Time": list(time),
        "Max_Voltage": max_voltage,
        "Energy": energy,
    }


if __name__ == "__main__":
    # Get all matching files sorted numerically/alphabetically
    all_files = os.listdir(data_dir)
    csv_files = sorted(
        [f for f in all_files if f.startswith("acq") and f.endswith(".csv")]
    )

    pulse_list = []

    print(f"Found {len(csv_files)} matching files. Processing...")

    for filename in csv_files:
        full_path = os.path.join(data_dir, filename)
        try:
            time, voltage = read_file(full_path)
            pulse_data = compute_energy(voltage, time)

            if pulse_data is not None:
                pulse_list.append(pulse_data)
        except Exception as e:
            print(f"Skipping {filename} due to an error: {e}")

    # Convert the list of results into a single DataFrame at the end
    if pulse_list:
        pulses = pd.DataFrame(pulse_list)

        print(f"Total pulses collected: {len(pulses)}")

        # FIX: Correct method to save HDF5 files in pandas
        pulses.to_hdf(output_path, key="pulses", mode="w")
        print(f"Data saved successfully to {output_path}")

        # Plotting the results
        # Fixed: bins should be a fixed integer (like 50 or 100) instead of len(pulses)
        num_bins = min(50, len(pulses))
        counts, bins = np.histogram(pulses["Energy"], bins=num_bins, density=True)

        plt.stairs(counts, bins)
        plt.title("Energy Distribution of Pulses")
        plt.xlabel("Energy (J)")
        plt.ylabel("Density")
        plt.show()
    else:
        print("No pulses exceeded the 0.4V threshold.")