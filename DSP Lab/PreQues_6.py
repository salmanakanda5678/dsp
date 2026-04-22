import numpy as np
import matplotlib.pyplot as plt

# Parameters
fs = 500
T  = 1.0
N  = int(fs * T)
t  = np.arange(N) / fs

# Signals
a = np.sin(2 * np.pi * 10 * t)          # 10 Hz sine
b = np.sign(np.sin(2 * np.pi * 10 * t)) # 10 Hz square wave
c = np.sin(2 * np.pi * 20 * t)          # 20 Hz sine

# Correlations
auto_a   = np.correlate(a, a, mode='full')
cross_ab = np.correlate(a, b, mode='full')
cross_ac = np.correlate(a, c, mode='full')

lags = np.arange(-(N-1), N) / fs        # Lag axis in seconds ✅

# --- Figure 1: Signal plots ---
plt.figure(figsize=(12, 7))

plt.subplot(3, 1, 1)
plt.plot(t, a, color='steelblue')
plt.title("Signal (a): 10 Hz Sine Wave")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(t, b, color='darkorange')
plt.title("Signal (b): 10 Hz Square Wave")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(t, c, color='seagreen')
plt.title("Signal (c): 20 Hz Sine Wave")
plt.xlabel("Time (s)"); plt.ylabel("Amplitude")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# --- Figure 2: Correlation plots ---
plt.figure(figsize=(12, 7))

plt.subplot(3, 1, 1)
plt.plot(lags, auto_a, color='steelblue')
plt.title("Auto-correlation of (a) — 10 Hz Sine")
plt.xlabel("Lag (s)"); plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(lags, cross_ab, color='purple')
plt.title("Cross-correlation: (a) Sine vs (b) Square — Same Frequency")
plt.xlabel("Lag (s)"); plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(lags, cross_ac, color='crimson')
plt.title("Cross-correlation: (a) 10 Hz vs (c) 20 Hz — Different Frequency")
plt.xlabel("Lag (s)"); plt.ylabel("Correlation")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

