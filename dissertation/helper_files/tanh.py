import numpy as np
import matplotlib.pyplot as plt

lim = 3
size = 6
fig, ax = plt.subplots(figsize=(size, size))
xs = np.linspace(-lim, lim)
ax.plot(xs, np.tanh(xs), label="y = tanh(x)", color="#830000", linewidth=2)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks([-2,-1, 0, 1, 2])
ax.set_yticks([-1, 0, 1])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid("on")
ax.legend()
fig.tight_layout()
fig.savefig("tanh.pdf", transparent=False)
