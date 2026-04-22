import numpy as np
import matplotlib.pyplot as plt

fs = 100                          # Sampling frequency (Hz)
N  = fs                           # Total samples (1 second)
t  = np.arange(N) / fs            # Time axis

# Generate signals
x1 = np.sin(2 * np.pi * 30 * t)  # 30 Hz sine wave
x2 = np.sin(2 * np.pi * 70 * t)  # 70 Hz sine wave

# Compute DFT
X1 = np.fft.fft(x1)
X2 = np.fft.fft(x2)

# Positive frequencies only
freq  = np.fft.fftfreq(N, d=1/fs)
half  = N // 2

mag1 = (2/N) * np.abs(X1[:half])  # Normalized magnitude
mag2 = (2/N) * np.abs(X2[:half])

# Plot
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.stem(freq[:half], mag1)
plt.title("Magnitude Spectrum — 30 Hz Sine Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)|")
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.stem(freq[:half], mag2)
plt.title("Magnitude Spectrum — 70 Hz Sine Wave")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|X(f)|")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Observation (print or write in viva)
print("Observation:")
print("Both spectra show a single spike at 30 Hz.")
print("70 Hz aliases to 30 Hz because fs=100 Hz → 100-70 = 30 Hz (Aliasing!)")