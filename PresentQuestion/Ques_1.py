import numpy as np
import matplotlib.pyplot as plt

# Define length
N = 20

# -------------------------------
# Impulse Signal δ[n]
# -------------------------------
impulse = np.zeros(N)
impulse[0] = 1  # δ[n]

# Impulse response h[n]
h = np.zeros(N)
for n in range(N):
    h[n] = 0.5 * impulse[n]
    if n-1 >= 0:
        h[n] += 0.3 * impulse[n-1]
    if n-2 >= 0:
        h[n] += 0.2 * impulse[n-2]

# Plot impulse response
plt.stem(h)
plt.title("Impulse Response h[n]")
plt.xlabel("n")
plt.ylabel("h[n]")
plt.show()

# -------------------------------
# Unit Step Signal u[n]
# -------------------------------
x = np.ones(N)

# -------------------------------
# Manual Convolution (NO FUNCTION)
# -------------------------------
y = np.zeros(N)

for n in range(N):
    for k in range(n+1):
        if k < len(x) and (n-k) < len(h):
            y[n] += x[k] * h[n-k]

# Plot output
plt.stem(y)
plt.title("Output y[n] (Step Input)")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.show()