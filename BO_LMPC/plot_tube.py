import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from mrpi.polyhedron import polyhedron, plot_polygon_list


xcl_tvbo = np.load('tvbo_3_xcl.npy')
xcl_lmpc = np.load('own_xcl.npy')
v_tvbo = np.load('./vertices/tvbo_2/vertices_49.npy', allow_pickle=True)
v_lmpc = np.load('./vertices/own/vertices_49.npy', allow_pickle=True)

plt.plot(xcl_tvbo[:, 0], xcl_tvbo[:, 1], label='transfer learning bo')
plt.scatter(xcl_tvbo[:, 0], xcl_tvbo[:, 1], s=10)
plt.plot(xcl_lmpc[:, 0], xcl_lmpc[:, 1], label='lmpc')
plt.scatter(xcl_lmpc[:, 0], xcl_lmpc[:, 1], s=10)
for i in range(xcl_tvbo.shape[0]):
    v_tb = v_tvbo[i]
    ax = plt.gca()
    hull = ConvexHull(v_tb)
    v_tb = v_tb[hull.vertices, :]
    v_tb = v_tb + xcl_tvbo[i]  # 整体偏移
    patch = plt.Polygon(v_tb, color='blue', linestyle='solid', fill=False)
    ax.add_patch(patch)

for i in range(xcl_lmpc.shape[0]):
    v_lm = v_lmpc[i]
    ax = plt.gca()
    hull = ConvexHull(v_lm)
    v_lm = v_lm[hull.vertices, :]
    v_lm = v_lm + xcl_lmpc[i]  # 整体偏移
    patch = plt.Polygon(v_lm, color='orange', linestyle='solid', fill=False)
    ax.add_patch(patch)
plt.legend()
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.savefig('./figs/x_tube.png', dpi=600)
plt.show()

ucl_tvbo = np.load('tvbo_3_ucl_true.npy')
uv_tvbo = np.load('./vertices/tvbo_3/Kxs_49.npy', allow_pickle=True)
ucl_lmpc = np.load('own_ucl.npy')
uv_lm = np.load('./vertices/own/Kxs_49.npy', allow_pickle=True)
fig, ax = plt.subplots(1, 2, figsize=[13, 5])
tb = []
x_tb = np.linspace(1, ucl_tvbo.shape[0], ucl_tvbo.shape[0])
for i in range(ucl_tvbo.shape[0]):
    tb.append(np.max(uv_tvbo[i]))
ax[0].errorbar(x_tb, ucl_tvbo, tb, fmt='o-', ecolor='blue', elinewidth=2, capsize=4, label='transfer learning bo')
ax[0].set_title('transfer learning bo')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('u')
lm = []
x_lm = np.linspace(1, ucl_lmpc.shape[0], ucl_lmpc.shape[0])
for i in range(ucl_lmpc.shape[0]):
    lm.append(np.max(uv_lm[i]))
ax[1].errorbar(x_lm, ucl_lmpc, lm, fmt='o-', ecolor='r', elinewidth=2, capsize=4, label='lmpc')
ax[1].set_title('lmpc')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('u')
# plt.plot(ucl_tvbo, label='ucl tvbo')
# # plt.scatter(ucl_tvbo, s=10)
# plt.legend()

plt.savefig('./figs/u_tube.png', dpi=600)
plt.show()