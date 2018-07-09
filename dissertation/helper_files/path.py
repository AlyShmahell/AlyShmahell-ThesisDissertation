import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):                                        
	return 1 / (1 + np.exp(-x))

def leaky_relu(x):
	c = np.zeros_like(x)
	slope = 1e-1
	c[x>0] = x[x>0]
	c[x<=0] = slope*x[x<=0]
	return c

def pathFunc(x):
	results = []
	lim = 3
	for val1 in np.linspace(-lim, lim):
		for val2 in np.linspace(-lim, lim):
			results.append(sigmoid(leaky_relu(sigmoid(x)*val1)*val2))
	return results
	
lim = 3
size = 6
fig, ax = plt.subplots(figsize=(size, size))
xs = np.linspace(-lim, lim)
for arr in pathFunc(xs):
	ax.plot(xs, arr , color="#830000", linewidth=2)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.set_xlabel("x")
ax.set_ylabel("∀x ∈ X, X = [-1, +1]: Y = sigmoid(leakyRelu(sigmoid(X) ∗ x) ∗ x)")
ax.grid("on")
fig.tight_layout()
fig.savefig("sigmoid_leakyRelu_sigmoid.pdf", transparent=False)
