import polars as pl
import matplotlib.pyplot as plt
import numpy as np

data = pl.read_csv("Data/RawData2705/integrals.csv")

integrals = data["integral_Vs"].to_numpy()

counts, bins = np.histogram(integrals, bins=75)
plt.figure(figsize=(10, 6))
plt.stairs(counts, bins, fill=True, color="steelblue", alpha=0.7)
plt.axvline(5.59)
plt.xlabel("Negative Integral (V*s)")
plt.ylabel("Frequency")
plt.title("Histogram of Negative Integrals")
plt.grid(True, alpha=0.3)
plt.show()
