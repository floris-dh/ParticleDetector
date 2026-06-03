import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import poisson
import polars as pl
def poisson_pmf(k, lambda_, A):
    return A * poisson.pmf(lambda_, k)

Am_data = r"RawData_Ra226\integrals.csv"

integrals_df = pl.read_csv(Am_data, has_header=True)

integrals_df.show()

binsize = np.sqrt(len(integrals_df))
counts, bin_edges = np.histogram(integrals_df['integral_Vs'], bins=int(binsize))
x = (bin_edges[:-1] + bin_edges[1:]) / 2
y = counts

est_max = np.max(counts)
est_index_mean = np.average(x, weights=counts)

popt, pcov = curve_fit(poisson_pmf, x, y)#, p0=[est_index_mean, est_max])


print(popt)

x_array = np.linspace(min(x), max(x), 100)

plt.stairs(counts, bin_edges, fill=True, color='skyblue', edgecolor='black', alpha=0.7)
plt.plot(x_array, poisson_pmf(x_array, *popt), color='red', label='Poisson Fit')
plt.title('Histogram of Integrals with Poisson Fit')
plt.xlabel('Integral (Vs)')
plt.ylabel('Count')
plt.grid(True)
plt.legend()
plt.show()

