import numpy as np
import matplotlib.pyplot as plt

fs = 100
f = 10

t = np.arange(0, 1, 1/fs)

# Original clean signal
signal = np.sin(2*np.pi*f*t)

# Add noise
noise = np.random.normal(0, 0.5, len(t))
noisy_signal = signal + noise

# 6-point moving average filter
h = np.ones(6) / 6
filtered_signal = np.convolve(noisy_signal, h, mode='same')

# Plot
plt.figure(figsize=(10,6))

plt.plot(t, signal, label="Original Signal", linestyle='dashed')
plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.6)
plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)

plt.legend()
plt.title("Noise Removal using Moving Average Filter")
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.grid()

plt.show()