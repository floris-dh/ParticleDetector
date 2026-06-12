import polars as pl
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your data
data = pl.read_csv(r"Data\Ra-080626-2_output\fitted_params.csv")
integral_data = pl.read_csv(r"Data\Ra-080626-2_output\pulse_integrals.csv")

# 2. Define your threshold for "very small C"
C_THRESHOLD = 0.001  # Adjust this value based on your actual baseline scale

# 3. Filter the fitted parameters where |C| is small, then join or filter integral data
# We get the chunk_ids that meet the baseline criteria
valid_baseline_ids = data.filter(pl.col("C").abs() < C_THRESHOLD).get_column("chunk_id")
# Filter integral data so we only sample from pulses with a small C
filtered_integrals = integral_data.filter(pl.col("chunk_id").is_in(valid_baseline_ids))
filtered_integrals = filtered_integrals.sort("pulse_integral", descending=True)

ids = []

def pulse_model(t, t0, a, tau_1, tau_2, b, tau_3, c) -> np.ndarray:
    with np.errstate(over="ignore"):
        dt = t - t0
        result = (
            c
            - (a + b) * np.exp(-dt / tau_1)
            + a * np.exp(-dt / tau_2)
            + b * np.exp(-dt / tau_3)
        )
    return np.where(dt >= 0, result, c)

# 4. Pick 7 representative pulses from the filtered subset
if filtered_integrals.height > 0:
    sample_indices = np.linspace(2500, filtered_integrals.height - 1, 7, dtype=int)[:6]
    ids = [filtered_integrals["chunk_id"][int(i)] for i in sample_indices]
    
pulses = data.filter(pl.col("chunk_id").is_in(ids))

# 5. Plotting setup
fig, ax = plt.subplots(figsize=(8, 5), facecolor='#182731')

t = np.linspace(-0.00013, 0.001, 1000)

for i, row in enumerate(pulses.rows(named=True)):
    t0, A, tau_1, tau_2, B, tau_3, C = row["t0"], row["A"], row["tau_1"], row["tau_2"], row["B"], row["tau_3"], row["C"]
    fitted_curve = pulse_model(t, t0, A, tau_1, tau_2, B, tau_3, C)

    # Label with C value to verify the filter worked
    if i == 0:
        label = f"Pulses"
    else:
        label = None  # Only label the first one for legend clarity
        
    ax.plot(t - t0, np.where(fitted_curve <= C_THRESHOLD, fitted_curve, np.nan), color='#EFCD88', alpha=0.8, label=label)

ax.axhline(-0.05, label="Trigger (-50 mV)", color='#7F8550', linestyle='--', alpha=0.8)

ax.annotate(
    text="Pulse from Po214",
    xy=(0.0004, -0.145),
    textcoords="data",
    color='#BBC2C6', fontsize=9,          
    ha='center', va='bottom', weight='bold'
)

handles, labels = ax.get_legend_handles_labels()

fig.legend(
    handles, 
    labels, 
    loc='lower center',       # Positions it at the bottom middle
    ncol=3,                   # Forces the items to sit side-by-side horizontally
    bbox_to_anchor=(0.5, -0.05), # Fine-tunes positioning (X=center, Y=slightly below plots)
    facecolor='inherit',
    frameon=False,              # True or False depending on if you want a box border
    labelcolor='#BBC2C6',                     # Changes the text color
    prop={'size': 12, 'weight': 'bold'}       # Changes the font properties
)

plt.suptitle("Pulses collected from Ra226", fontsize=14, weight='bold', color="#F5FDFF")
ax.axhline(0, color='#BBC2C6', linestyle='--', alpha=0.5)
ax.set_xlabel('Time (s)', color='#BBC2C6')
ax.set_ylabel('Voltage (V)', color='#BBC2C6')
ax.grid(alpha=0.3, color='#BBC2C6', linestyle='--')
ax.set_facecolor('#223441')
ax.tick_params(colors='#BBC2C6', which='both')
plt.savefig(r"Figures\informative_pulses.png", dpi=300, bbox_inches='tight')