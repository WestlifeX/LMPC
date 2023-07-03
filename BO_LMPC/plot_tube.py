import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from mrpi.polyhedron import polyhedron, plot_polygon_list
from matplotlib.patches import ConnectionPatch
font = {'family': 'Times New Roman', 'size': 16}
colors = ['sandybrown', 'purple']
xcl_tvbo = np.load('tvbo_3_xcl.npy')
xcl_lmpc = np.load('own_xcl.npy')
v_tvbo = np.load('./vertices/tvbo_2/vertices_49.npy', allow_pickle=True)
v_lmpc = np.load('./vertices/own/vertices_49.npy', allow_pickle=True)

plt.plot(xcl_lmpc[:, 0], xcl_lmpc[:, 1], lw=2, label='RLMPC', color=colors[0])
plt.scatter(xcl_lmpc[:, 0], xcl_lmpc[:, 1], lw=2, s=20, color=colors[0])
plt.plot(xcl_tvbo[:, 0], xcl_tvbo[:, 1], lw=2, label='Efficient BO', color=colors[1], linestyle='--')
plt.scatter(xcl_tvbo[:, 0], xcl_tvbo[:, 1], s=20, color=colors[1])
plt.grid()
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
for i in range(xcl_lmpc.shape[0]):
    v_lm = v_lmpc[i]
    ax = plt.gca()
    hull = ConvexHull(v_lm)
    v_lm = v_lm[hull.vertices, :]
    v_lm = v_lm + xcl_lmpc[i]  # 整体偏移
    patch = plt.Polygon(v_lm, color=colors[0], linestyle='solid', fill=False)
    ax.add_patch(patch)

for i in range(xcl_tvbo.shape[0]):
    v_tb = v_tvbo[i]
    ax = plt.gca()
    hull = ConvexHull(v_tb)
    v_tb = v_tb[hull.vertices, :]
    v_tb = v_tb + xcl_tvbo[i]  # 整体偏移
    patch = plt.Polygon(v_tb, color=colors[1], linestyle='solid', fill=False)
    ax.add_patch(patch)

plt.legend(prop=font)
plt.xlabel('x[0]', font)
plt.ylabel('x[1]', font)
plt.savefig('./figs/x_tube.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()

ucl_tvbo = np.load('tvbo_3_ucl_true.npy')
uv_tvbo = np.load('./vertices/tvbo_3/Kxs_49.npy', allow_pickle=True)
ucl_lmpc = np.load('own_ucl.npy')
uv_lm = np.load('./vertices/own/Kxs_49.npy', allow_pickle=True)
fig, ax = plt.subplots(1, 2, figsize=[13, 6])

lm = []
x_lm = np.linspace(1, ucl_lmpc.shape[0], ucl_lmpc.shape[0])
for i in range(ucl_lmpc.shape[0]):
    lm.append(np.max(uv_lm[i]))
ax[0].scatter(x_lm, ucl_lmpc, color=colors[0], s=20)
ax[0].errorbar(x_lm, ucl_lmpc, lm, fmt='none', ecolor=colors[0], elinewidth=2, capsize=4, label='lmpc')
ax[0].set_title('RLMPC', font)
ax[0].set_xlabel('Iteration', font)
ax[0].set_ylabel('u', font)
ax[0].set_ylim(-1.1, 1.1)
# ax = ax[0].gca()
ax[0].spines['right'].set_color('none')
ax[0].spines['top'].set_color('none')
ax[0].grid()
ax[0].tick_params(labelsize=12)
tb = []
x_tb = np.linspace(1, ucl_tvbo.shape[0], ucl_tvbo.shape[0])
for i in range(ucl_tvbo.shape[0]):
    tb.append(np.max(uv_tvbo[i]))
ax[1].scatter(x_tb, ucl_tvbo, color=colors[1], s=20)
ax[1].errorbar(x_tb, ucl_tvbo, tb, fmt='none', ecolor=colors[1], elinewidth=2, capsize=4, label='transfer learning bo')
ax[1].set_title('Efficient BO', font)
ax[1].set_xlabel('Iteration', font)
ax[1].set_ylabel('u', font)
ax[1].set_ylim(-1.1, 1.1)
# ax = ax[1].gca()
ax[1].spines['right'].set_color('none')
ax[1].spines['top'].set_color('none')
ax[1].grid()
for i in range(5):
    p1 = (i+1, ucl_lmpc[i]+lm[i])
    p2 = (i+1, ucl_lmpc[i]-lm[i])
    con1 = ConnectionPatch(xyA=p1, xyB=p1, coordsA='data', coordsB='data', axesA=ax[0], axesB=ax[1], color='gray',
                    linestyle='dashed')
    con2 = ConnectionPatch(xyA=p2, xyB=p2, coordsA='data', coordsB='data', axesA=ax[0], axesB=ax[1], color='gray',
                    linestyle='dashed')
    ax[1].add_artist(con1)
    ax[1].add_artist(con2)

    l1 = ucl_lmpc[i] + lm[i] - ucl_tvbo[i] - tb[i]
    l2 = ucl_tvbo[i] - tb[i] - ucl_lmpc[i] + lm[i]
    ax[1].plot([i+1, i+1], [ucl_lmpc[i]+lm[i], ucl_tvbo[i]+tb[i]], c='r', linestyle='dotted')
    ax[1].plot([i+1, i+1], [ucl_lmpc[i]-lm[i], ucl_tvbo[i]-tb[i]], c='r', linestyle='dotted')
    # ax[1].arrow(i+1, ucl_tvbo[i]+tb[i], 0, l1, color='r', head_width=0.2, head_length=0.0045)
    # ax[1].arrow(i+1, ucl_lmpc[i]+lm[i], 0, -l1, color='r', head_width=0.01)
    # #
    # ax[1].arrow(i+1, ucl_tvbo[i]-tb[i], 0, -l2, color='r', head_width=0.01)
    # ax[1].arrow(i+1, ucl_lmpc[i]-lm[i], 0, l2, color='r', head_width=0.01)
# plt.plot(ucl_tvbo, label='ucl tvbo')
# # plt.scatter(ucl_tvbo, s=10)
# plt.legend()
plt.tick_params(labelsize=12)
plt.subplots_adjust(wspace=1.2)
plt.savefig('./figs/u_tube.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()