import numpy as np
import matplotlib.pyplot as plt

fs = 100
f = 10

# -------- Case 1 --------
t1 = np.arange(0, 1, 1/fs)
x1 = np.sin(2*np.pi*f*t1)

X1 = np.fft.fft(x1)
X1 = np.abs(X1) / len(x1)
freq1 = np.fft.fftfreq(len(x1), 1/fs)

# -------- Case 2 --------
t2 = np.arange(0, 0.95, 1/fs)
x2 = np.sin(2*np.pi*f*t2)

X2 = np.fft.fft(x2)
X2 = np.abs(X2) / len(x2)
freq2 = np.fft.fftfreq(len(x2), 1/fs)

# -------- Hamming --------
w = np.hamming(len(x2))
x2w = x2 * w

X2w = np.fft.fft(x2w)
X2w = np.abs(X2w) / len(x2w)

# -------- Plot only positive freq --------
plt.figure(figsize=(12,8))

plt.subplot(3,1,1)
plt.title("DFT (1 sec - integer cycles)")
plt.plot(freq1[:len(freq1)//2], X1[:len(X1)//2])
plt.grid(True)

plt.subplot(3,1,2)
plt.title("DFT (0.95 sec - leakage)")
plt.plot(freq2[:len(freq2)//2], X2[:len(X2)//2])
plt.grid(True)

plt.subplot(3,1,3)
plt.title("DFT with Hamming Window")
plt.plot(freq2[:len(freq2)//2], X2w[:len(X2w)//2])
plt.grid(True)

plt.tight_layout()
plt.show()