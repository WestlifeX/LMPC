import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from mrpi.polyhedron import polyhedron, plot_polygon_list

ys = np.array([839.7065714])
xs = np.ones(1) * 1
plt.scatter(xs, ys, marker='*', s=20, c='black', label='current point')
for i in range(29):
    ys = np.load('./ys/tvbo/y_{}.npy'.format(i))
    xs = np.ones(10) * (i+2)
    plt.scatter(xs, ys, marker='*', s=20, c='black')

with open('./new_results/robust_tvbo_3.md') as f:
    lines = f.readlines()
    lines = [float(i.strip().strip('[[').strip(']]')) for i in lines]
    lines = np.array(lines)
    current_best = np.zeros_like(lines)
    vmin = np.inf
    for j in range(lines.shape[0]):
        if lines[j] < vmin:
            current_best[j] = lines[j]
            vmin = lines[j]
        else:
            current_best[j] = vmin
    plt.plot(np.linspace(1, 30, 30), current_best[:30], color='purple', label='current best performance')

font = {'family': 'Times New Roman', 'size': 16}
plt.xlabel('Iterations', font)
plt.ylabel('Cost', font)
plt.grid()
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
plt.legend(prop=font)
plt.savefig('./figs/data_points.jpg', dpi=1200, bbox_inches='tight', pad_inches=0)
plt.show()
