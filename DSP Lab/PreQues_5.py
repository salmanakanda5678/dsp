import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin

# Parameters
fs      = 500          # Sampling frequency (Hz) — must be > 2×100 Hz
T       = 1.0          # Duration (seconds)
N       = int(fs * T)
t       = np.arange(N) / fs

# Signals
clean   = np.sin(2 * np.pi * 10  * t)   # 10 Hz — desired signal
noise   = np.sin(2 * np.pi * 100 * t)   # 100 Hz — noise
noisy   = clean + noise                  # Noisy signal

# FIR Low-Pass Filter (Hamming window, cutoff = 20 Hz)
numtaps = 101                            # More taps = sharper cutoff
cutoff  = 20                             # Hz
h       = firwin(numtaps, cutoff, fs=fs, window='hamming')

# Filter using convolution
filtered = np.convolve(noisy, h, mode='same')

# Plot 1: Noisy Signal
plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(t, noisy, color='gray')
plt.title("Noisy Signal (10 Hz + 100 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Plot 2: Filtered Signal
plt.subplot(3, 1, 2)
plt.plot(t, filtered, color='steelblue')
plt.title("Filtered Signal (After FIR Low-Pass, cutoff=20 Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

# Plot 3: Noisy vs Filtered (Comparison) ✅ — এটাই exam চেয়েছে
plt.subplot(3, 1, 3)
plt.plot(t, noisy,    color='gray',      alpha=0.6, label='Noisy')
plt.plot(t, filtered, color='steelblue', linewidth=2, label='Filtered')
plt.title("Comparison: Noisy vs Filtered")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Observation:")
print("The FIR low-pass filter (Hamming, cutoff=20 Hz, 101 taps)")
print("successfully removes the 100 Hz noise.")
print("The filtered output closely matches the original 10 Hz sine wave.")