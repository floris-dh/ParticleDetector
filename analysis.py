import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time

start_time = time.time()


def import_data(file_path):
    # Assuming the data is in CSV format with columns: Time, Voltage
    data = pd.read_csv(file_path, skiprows=9)
    return data["Time (s)"], data["Channel 1 (V)"]


i = "0001"

integral_list = []

while os.path.exists(f"Data/acq{i}.csv"):
    times, voltage = import_data(f"Data/acq{i}.csv")

    integral = np.trapezoid(voltage, times)

    print(f"Integral for acq{i}.csv: {integral:.4f}", end="\r")

    i = str(int(i) + 1).zfill(4)

    integral_list.append(integral)

    # plt.figure(figsize=(10, 6))
    # plt.plot(times, voltage)
    # plt.title(f"Voltage vs Time for acq{i}.csv")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Voltage (V)")
    # plt.savefig(f"Figures/acq/acq{i}.png")


plt.figure(figsize=(10, 6))
counts, bins = np.histogram(
    integral_list,
    bins=int(np.sqrt(len(integral_list))),
    density=True,
)
plt.stairs(counts, bins, edgecolor="black", alpha=0.7)
plt.xlabel("Integral of Voltage over Time")
plt.ylabel("Density")

end_time = time.time()

plt.title("Histogram of Integrals")
plt.savefig("Figures/integral_histogram.png")

print(f"Processed {len(integral_list)} files in {end_time - start_time:.2f} seconds.")
