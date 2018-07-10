# -*- coding: utf-8 -*-
"""Neumerical Analysis of different activation function paths for neurencoder."""
__author__ = "Aly Shmahell"
__copyright__ = "Copyright © 2018, Aly Shmahell"
__license__ = "All Rights Reserved"
__version__ = "TDPR1"
__maintainer__ = "Aly Shmahell"
__email__ = "aly.shmahell@gmail.com"
__status__ = "Thesis Defense PreRelease"

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 10})

lim = 3

def sigmoid(x):                                        
	return 1 / (1 + np.exp(-x))

def leaky_relu(x):
	y = np.zeros_like(x)
	slope = 1e-1
	y[x>0] = x[x>0]
	y[x<=0] = slope*x[x<=0]
	return y

def pathFunc(x):
	results = []
	for val1 in np.linspace(-lim, lim):
		for val2 in np.linspace(-lim, lim):
			results.append(sigmoid(leaky_relu(sigmoid(x)*val1)*val2))
	return results
	

fig, ax = plt.subplots(figsize=(lim*2, lim*2))
linspace = np.linspace(-lim, lim)
for arr in pathFunc(linspace):
	ax.plot(linspace, arr , color="#830000", linewidth=2)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks([x for x in range(-lim,lim+1)])
ax.set_yticks([y for y in range(-lim,lim+1)])
ax.set_xlabel("x")
ax.set_ylabel("∀x ∈ X, X = [-1, +1]: Y = sigmoid(leakyRelu(sigmoid(X) ∗ x) ∗ x)")
ax.grid("on")
fig.tight_layout()
fig.savefig("sigmoid_leakyRelu_sigmoid.pdf", transparent=False)
