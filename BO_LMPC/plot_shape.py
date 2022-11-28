import numpy as np
from FTOCP_casadi import FTOCP
from LMPC import LMPC
import matplotlib
from tqdm import tqdm
from control import dlqr

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy

from objective_functions_lqr import get_params
from args import Q, R, R_delta, compute_uncertainty, A, B, Ad, Bd

import arguments

n_params = 2
length = 100
samples = np.linspace(1, length, length)
x = np.linspace(1, length, length)
y = np.linspace(1, length, length)
xx, yy = np.meshgrid(x, y)
all_res = np.load('./all_res.npy')
font = {'family': 'Times New Roman', 'size': 16}
fig = plt.figure(figsize=[13, 9])
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(xx, yy, all_res[:, :, 0], cmap=plt.get_cmap('rainbow'))
ax.set_title('Iteration 1', font)
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(xx, yy, all_res[:, :, 1], cmap=plt.get_cmap('rainbow'))
ax.set_title('Iteration 2', font)
ax = fig.add_subplot(133, projection='3d')
ax.plot_surface(xx, yy, all_res[:, :, 2], cmap=plt.get_cmap('rainbow'))
ax.set_title('Iteration 3', font)
plt.subplots_adjust(wspace=0.6)
plt.savefig('./figs/similarity.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()
a = 1


