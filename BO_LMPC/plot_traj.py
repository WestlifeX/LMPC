import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from mrpi.polyhedron import polyhedron, plot_polygon_list
# xcl_tvbo = np.load('tvbo_3_xcl.npy')
# xcl_lmpc = np.load('own_xcl_true.npy')
# xcl_lmpc = np.load('own_xcl.npy')
# v_tvbo = np.load('./vertices/tvbo_2/vertices_49.npy', allow_pickle=True)
# v_lmpc = np.load('./vertices/own/vertices_49.npy', allow_pickle=True)

# plt.plot(xcl_tvbo[:, 0], xcl_tvbo[:, 1], label='tvbo')
# plt.scatter(xcl_tvbo[:, 0], xcl_tvbo[:, 1], s=10)
# plt.plot(xcl_lmpc[:, 0], xcl_lmpc[:, 1], label='lmpc')
# plt.scatter(xcl_lmpc[:, 0], xcl_lmpc[:, 1], s=10)
# for i in range(xcl_tvbo.shape[0]):
#     v_tb = v_tvbo[i]
#     ax = plt.gca()
#     hull = ConvexHull(v_tb)
#     v_tb = v_tb[hull.vertices, :]
#     v_tb = v_tb + xcl_tvbo[i]  # 整体偏移
#     patch = plt.Polygon(v_tb, color='blue', linestyle='solid', fill=False)
#     ax.add_patch(patch)
#
# for i in range(xcl_lmpc.shape[0]):
#     v_lm = v_lmpc[i]
#     ax = plt.gca()
#     hull = ConvexHull(v_lm)
#     v_lm = v_lm[hull.vertices, :]
#     v_lm = v_lm + xcl_lmpc[i]  # 整体偏移
#     patch = plt.Polygon(v_lm, color='orange', linestyle='solid', fill=False)
#     ax.add_patch(patch)
# plt.legend()
# plt.show()

ucl_tvbo = np.load('tvbo_3_ucl_true.npy')
uv_tvbo = np.load('./vertices/tvbo_2/Kxs.npy', allow_pickle=True)
ucl_lmpc = np.load('own_ucl.npy')
uv_lm = np.load('./vertices/own/Kxs.npy', allow_pickle=True)

tb = []
x_tb = np.linspace(1, ucl_tvbo.shape[0], ucl_tvbo.shape[0])
for i in range(ucl_tvbo.shape[0]):
    tb.append(np.max(uv_tvbo[i]))
plt.errorbar(x_tb, ucl_tvbo, tb, fmt='o-', ecolor='blue', elinewidth=2, capsize=4)


lm = []
x_lm = np.linspace(1, ucl_lmpc.shape[0], ucl_lmpc.shape[0])
for i in range(ucl_lmpc.shape[0]):
    lm.append(np.max(uv_lm[i]))
plt.errorbar(x_lm, ucl_lmpc, lm, fmt='o-', ecolor='r', elinewidth=2, capsize=4)


# plt.plot(ucl_tvbo, label='ucl tvbo')
# # plt.scatter(ucl_tvbo, s=10)
# plt.legend()
plt.show()

from FTOCP_casadi import FTOCP
# from FTOCP import FTOCP
from LMPC import LMPC
# points = self.vertices
#         ax = plt.gca()
# hull = ConvexHull(points)
#             points = points[hull.vertices, :]
#             patch = plt.Polygon(points, color=color, linestyle=linestyle,
#                                 fill=fill, linewidth=linewidth)
#             ax.add_patch(patch)
#             ax.autoscale_view()
#             ax.relim()
#             ax.grid(color=(0, 0, 0), linestyle='--', linewidth=0.3)
#             ax.set_title(title)
#             plt.xlabel('c\,(m)', fontsize='19')
#             plt.ylabel('\dot{c}\,({m}/{s})', fontsize='19')
#             plt.show()


