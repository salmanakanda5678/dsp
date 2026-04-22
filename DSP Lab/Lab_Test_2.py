import numpy as np
import matplotlib.pyplot as plt

# Given specifications
fs = 8000          # Sampling frequency
fc = 1000          # Cutoff frequency
N = 21             # Filter length

# Normalized cutoff frequency (in radians)
wc = 2 * np.pi * fc / fs

# Generate n
n = np.arange(N)

# Center index
M = (N - 1) / 2

# Ideal impulse response (sinc function)
h_ideal = np.zeros(N)
for i in range(N):
    if n[i] == M:
        h_ideal[i] = wc / np.pi
    else:
        h_ideal[i] = np.sin(wc * (n[i] - M)) / (np.pi * (n[i] - M))

# Hamming window
hamming_window = np.hamming(N)

# Apply window
h = h_ideal * hamming_window

# Frequency response
H = np.fft.fft(h, 1024)
H_shifted = np.fft.fftshift(H)
freq = np.linspace(-fs/2, fs/2, len(H_shifted))

# Magnitude and phase
magnitude = np.abs(H_shifted)
phase = np.angle(H_shifted)

# ================== Plotting ==================

plt.figure(figsize=(12, 8))

# Impulse Response
plt.subplot(3,1,1)
plt.stem(n, h)
plt.title("Impulse Response (Windowed)")
plt.xlabel("n")
plt.ylabel("h[n]")

# Magnitude Response
plt.subplot(3,1,2)
plt.plot(freq, magnitude)
plt.title("Magnitude Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("|H(f)|")

# Phase Response
plt.subplot(3,1,3)
plt.plot(freq, phase)
plt.title("Phase Response")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")

plt.tight_layout()
plt.show()