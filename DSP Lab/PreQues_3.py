import numpy as np
import matplotlib.pyplot as plt

fs = 100
t = np.arange(0, 1, 1/fs)

# Original signal
x = np.sin(2*np.pi*5*t)

# -------- Proper Delay (zero padding) --------
delay = 10
x_delayed = np.concatenate((np.zeros(delay), x[:-delay]))

# -------- Cross-correlation --------
corr = np.correlate(x, x_delayed, mode='full')
lags = np.arange(-len(x)+1, len(x))

# Detect delay
estimated_delay = lags[np.argmax(corr)]

# -------- Plot signals --------
plt.figure(figsize=(10,6))
plt.plot(t, x, label="Original Signal")
plt.plot(t, x_delayed, label="Delayed Signal")
plt.legend()
plt.title("Original vs Delayed Signal")
plt.grid()
plt.show()

# -------- Plot correlation --------
plt.figure(figsize=(10,6))
plt.plot(lags, corr)
plt.title("Cross-Correlation")
plt.xlabel("Lag")
plt.ylabel("Correlation")
plt.grid()
plt.show()

print("Actual Delay:", delay)
print("Estimated Delay:", estimated_delay)