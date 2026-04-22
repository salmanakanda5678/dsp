import numpy as np
import matplotlib.pyplot as plt

# Input signals
x1 = np.array([1, 2, 3, 4])
x2 = np.array([2, 1, 0, 1])

# Length for FFT
N = len(x1) + len(x2) - 1

# -------------------------------
# Time-domain convolution
# -------------------------------
conv_time = np.convolve(x1, x2)

# -------------------------------
# Frequency-domain multiplication
# -------------------------------
X1 = np.fft.fft(x1, N)
X2 = np.fft.fft(x2, N)

Y = X1 * X2
conv_freq = np.fft.ifft(Y)
conv_freq = np.real(conv_freq)

# -------------------------------
# Difference check
# -------------------------------
error = conv_time - conv_freq

# -------------------------------
# Print all outputs
# -------------------------------
print("x1 =", x1)
print("x2 =", x2)

print("\nTime-domain Convolution:")
print(conv_time)

print("\nFrequency-domain Result (IFFT):")
print(conv_freq)

print("\nDifference (should be ~0):")
print(error)

# -------------------------------
# Plot all outputs together
# -------------------------------
plt.figure(figsize=(10,6))

plt.subplot(3,1,1)
plt.stem(conv_time)
plt.title("Time Domain Convolution")

plt.subplot(3,1,2)
plt.stem(conv_freq)
plt.title("Frequency Domain (IFFT Result)")

plt.subplot(3,1,3)
plt.stem(error)
plt.title("Error (Difference)")

plt.tight_layout()
plt.show()