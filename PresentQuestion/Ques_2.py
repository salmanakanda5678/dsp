import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Given Signal
# -------------------------------
x = [1, 2, 3, 4]
N = len(x)

# -------------------------------
# (i) Manual DFT
# -------------------------------
X = [0]*N

for k in range(N):
    real = 0
    imag = 0
    for n in range(N):
        angle = -2 * np.pi * k * n / N
        real += x[n] * np.cos(angle)
        imag += x[n] * np.sin(angle)
    X[k] = complex(real, imag)

print("Manual DFT X[k]:")
for i in range(N):
    print(f"X[{i}] = {X[i]}")

# -------------------------------
# (ii) NumPy DFT (for verification)
# -------------------------------
X_np = np.fft.fft(x)
print("\nNumPy DFT:")
print(X_np)

# -------------------------------
# (iii) Magnitude & Phase
# -------------------------------
magnitude = [0]*N
phase = [0]*N

for k in range(N):
    real = X[k].real
    imag = X[k].imag
    
    magnitude[k] = (real**2 + imag**2)**0.5
    phase[k] = np.arctan2(imag, real)

# Plot Magnitude Spectrum
plt.figure()
plt.stem(range(N), magnitude)
plt.title("Magnitude Spectrum |X[k]|")
plt.xlabel("k")
plt.ylabel("|X[k]|")
plt.grid()
plt.show()

# Plot Phase Spectrum
plt.figure()
plt.stem(range(N), phase)
plt.title("Phase Spectrum ∠X[k]")
plt.xlabel("k")
plt.ylabel("Phase (radians)")
plt.grid()
plt.show()

# -------------------------------
# (iv) Manual IDFT
# -------------------------------
x_reconstructed = [0]*N

for n in range(N):
    value = 0
    for k in range(N):
        angle = 2 * np.pi * k * n / N
        value += (X[k].real * np.cos(angle) - X[k].imag * np.sin(angle))
    x_reconstructed[n] = value / N

print("\nReconstructed Signal (IDFT):")
for i in range(N):
    print(f"x[{i}] = {x_reconstructed[i]}")

# Plot Reconstructed Signal
plt.figure()
plt.stem(range(N), x_reconstructed)
plt.title("Reconstructed Signal using IDFT")
plt.xlabel("n")
plt.ylabel("x[n]")
plt.grid()
plt.show()