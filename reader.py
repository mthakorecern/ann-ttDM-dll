import uproot
import numpy as np
import matplotlib.pyplot as plt

# Define file names
signal_files = ["output_signal1.root"]
background_files = ["output_background1.root", "output_background2.root"]

# Initialize lists to store histograms
hist_signal_list = []
hist_background_list = []

# Loop over signal files
for file_name in signal_files:
    file = uproot.open(file_name)
    branch_signal = file["output/tDM"]
    hist_signal_list.append(branch_signal.array())
    file.close()

# Loop over background files
for file_name in background_files:
    file = uproot.open(file_name)
    branch_background = file["output/tDM"]
    hist_background_list.append(branch_background.array())
    file.close()

# Combine histograms from different files
hist_signal = np.concatenate(hist_signal_list)
hist_background = np.concatenate(hist_background_list)

# Determine bin edges based on the range of data
min_value = min(np.min(hist_signal), np.min(hist_background))
max_value = max(np.max(hist_signal), np.max(hist_background))
bins = np.linspace(min_value, max_value, 201)  # Adjust the number of bins as needed

# Plot the histograms
plt.figure(figsize=(8, 6))
plt.hist(hist_signal, bins=bins, alpha=0.5, color='blue', label='Signal', density=True)
plt.hist(hist_background, bins=bins, alpha=0.5, color='orange', label='Background', density=True)
plt.xlabel("")
plt.ylabel("Normalized Counts")
plt.title("Signal vs Background")
plt.legend()
plt.grid(True)

# Calculate and display signal and background entries, mean, and standard deviation
signal_entries = len(hist_signal)
background_entries = len(hist_background)
mean_signal = np.mean(hist_signal)
std_signal = np.std(hist_signal)
mean_background = np.mean(hist_background)
std_background = np.std(hist_background)

# Add text annotations at the bottom of the histogram
plt.text(0.05, -0.12, f'Signal Entries: {signal_entries}, Background Entries: {background_entries}, Signal Mean: {mean_signal:.2f}, Signal Std: {std_signal:.2f}\nBackground Mean: {mean_background:.2f}, Background Std: {std_background:.2f}', transform=plt.gca().transAxes)

plt.savefig("SvsB.png")
plt.show()
