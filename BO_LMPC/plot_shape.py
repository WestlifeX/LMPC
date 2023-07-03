import numpy as np
from FTOCP_casadi import FTOCP
from LMPC import LMPC
import matplotlib
from tqdm import tqdm
from control import dlqr
import seaborn as sns
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from objective_functions_lqr import get_params
from args import Q, R, R_delta, compute_uncertainty, A, B, Ad, Bd

n_params = 2
length = 100
v = 0
samples = np.linspace(1, length, length)
x = np.linspace(1, length, length)
y = np.linspace(1, length-v, length-v)
xx, yy = np.meshgrid(x, y)
all_res = np.load('./all_res.npy')
font = {'family': 'Times New Roman', 'size': 16}
fig = plt.figure(figsize=[13, 9])
ax = fig.add_subplot(131, projection='3d')
ax.plot_surface(xx, yy, all_res[v:, :, 0], cmap=plt.get_cmap('rainbow'))
ax.set_title('(a) Iteration 1', font, y=-0.25)
ax = fig.add_subplot(132, projection='3d')
ax.plot_surface(xx, yy, all_res[v:, :, 1], cmap=plt.get_cmap('rainbow'))
ax.set_title('(b) Iteration 2', font, y=-0.25)
ax = fig.add_subplot(133)
# ax = sns.heatmap()
im = ax.imshow(all_res[v:, :, 1]-all_res[v:, :, 0], cmap=plt.cm.rainbow)
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=.05)
plt.colorbar(im, cax=cax)
# ax.plot_surface(xx, yy, all_res[v:, :, 0] - all_res[v:, :, 1], cmap=plt.get_cmap('rainbow'))
ax.set_title('(c) Variation between two iterations', font, y=-0.25)
plt.subplots_adjust(wspace=0.6)
plt.savefig('./figs/similarity.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()
a = 1


