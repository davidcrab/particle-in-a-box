import numpy as np
from scipy import linalg

# set up parameters
L = 1.0  # box length
m = 1.0  # particle mass
N = 100  # number of points in the box

# create a grid of points inside the box
x = np.linspace(0, L, N)

# create the potential energy function
V = np.zeros(N)  # potential is zero inside the box

# create the second-derivative matrix
h = L / N  # step size
D2 = -2.0*np.eye(N) + np.eye(N, k=-1) + np.eye(N, k=1)
D2 /= h**2

# solve the Schrödinger equation
H = -(1 / (2 * m)) * D2 + np.diag(V)  # Hamiltonian matrix
energies, wavefunctions = linalg.eigh(H)

# normalize the wavefunctions
wavefunctions = wavefunctions / np.sqrt(h)

import matplotlib.pyplot as plt

# plot the lowest few energy levels and wavefunctions
for n in range(4):  # number of levels to plot
    plt.figure()
    plt.plot(x, energies[n] + wavefunctions[:, n])
    plt.title(f"Energy level {n+1}")
    plt.xlabel("x")
    plt.ylabel("Energy + Ψ(x)")
    plt.grid(True)

plt.show()

# plot the probability densities for the first few energy levels
for i in range(4):  # number of energy levels to plot
    plt.plot(x, np.abs(wavefunctions[:, i])**2, label=f"n = {i+1}")
plt.legend()
plt.xlabel("Position (x)")
plt.ylabel("Probability Density |Ψ(x)|^2")
plt.title("Probability densities for a particle in a box")
plt.grid(True)
plt.show()
