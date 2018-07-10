# -*- coding: utf-8 -*-
"""
	Generates a tanh plot for use in the dissertation.
	Inspired by: https://is.wikipedia.org/wiki/Mynd:Sinh_cosh_tanh.svg
"""
__author__ = "Aly Shmahell"
__copyright__ = "Copyright © 2018, Aly Shmahell"
__license__ = "All Rights Reserved"
__version__ = "TDPR1"
__maintainer__ = "Aly Shmahell"
__email__ = "aly.shmahell@gmail.com"
__status__ = "Thesis Defense PreRelease"

import numpy as np
import matplotlib.pyplot as plt


lim = 3
fig, ax = plt.subplots(figsize=(lim*2, lim*2))
linspace = np.linspace(-lim, lim)
ax.plot(linspace, np.tanh(linspace), label="y = tanh(x)", color="#830000", linewidth=2)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xticks([x for x in range(-lim,lim+1)])
ax.set_yticks([y for y in range(-lim,lim+1)])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid("on")
ax.legend()
fig.tight_layout()
fig.savefig("tanh.pdf", transparent=False)
