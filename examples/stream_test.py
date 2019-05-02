# streamfolding.py

import pyfolding as pf
import numpy as np
import matplotlib.pyplot as plt


def data_generator(n: int, speed: int = 1e-3, std: float = 1.0):
    mu0 = np.array([-5, 5])
    v0 = np.array([1, -1]) / np.sqrt(2)
    mu1 = np.array([5, 5])
    v1 = np.array([-1, -1]) / np.sqrt(2)
    X = np.random.multivariate_normal([0, 0], std * np.eye(2), n)
    y = np.zeros(n)
    e = np.random.uniform(-1, 1, n)
    for i in range(n):
        if e[i] > 0:
            y[i] = 1
            X[i] += mu0 + v0 * speed * i
        else:
            y[i] = -1
            X[i] += mu1 + v1 * speed * i
    return X, y


# create StreamFolding object with window size equal to 500
depth = 500
sf = pf.StreamFolding(depth)

# container to retrieve some data during the exp.
x_phi = []
phi = []

# data
n_data = 20000
colors = np.zeros(n_data)
data, c = data_generator(n_data, speed=0.001)

# run
for i, x in enumerate(data):
    sf.update(x)
    if i > depth and i % 50 == 0:
        r = sf.folding_test()
        x_phi.append(i)
        phi.append(r.folding_statistics)

# plotting stuff
unimodal_indices = [i for i, k in enumerate(phi) if k > 1]

plt.figure(figsize=(10, 6))
plt.subplots_adjust(hspace=0.5)
ax1 = plt.subplot(2, 2, (1, 2))

ax1.plot(x_phi, phi, label="Folding statistics")
ax1.plot([0, n_data], [1, 1], ls='--', lw=3, color='tab:red')
ax1.axvspan(x_phi[unimodal_indices[0]], x_phi[unimodal_indices[-1]],
            facecolor='tab:green', alpha=0.2, label="Unimodal")
ax1.axvspan(0, x_phi[unimodal_indices[0]], facecolor='tab:green', alpha=0.1, label="Multimodal")
ax1.axvspan(x_phi[unimodal_indices[0]], x_phi[-1], facecolor='tab:green', alpha=0.1)
ax1.set_xlabel('iterations')
ax1.set_ylabel('Φ(X)')
ax1.legend()

it = 2500
ax2 = plt.subplot(2, 2, 3)
ax2.scatter(data[it:it + depth, 0],
            data[it:it + depth, 1],
            c=c[it:it + depth], alpha=0.8, cmap='tab20')
ax2.axis('equal')
ax2.set_title(f'Iteration {it}')

it = 8000
ax3 = plt.subplot(2, 2, 4)
ax3.scatter(data[it:it + depth, 0],
            data[it:it + depth, 1],
            c=c[it:it + depth], alpha=0.8, cmap='tab20')
ax3.axis('equal')
ax3.set_title(f'Iteration {it}')

# save the figure
plt.savefig("../assets/streamfolding.svg")
