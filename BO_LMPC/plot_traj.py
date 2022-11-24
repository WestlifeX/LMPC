import matplotlib.pyplot as plt
import numpy as np

xcl_tvbo = np.load('tvbo_3_xcl.npy')
xcl_lmpc = np.load('own_xcl.npy')
v_tvbo = np.load('./vertices/tvbo_2/vertices_49.npy', allow_pickle=True)
v_lmpc = np.load('./vertices/own/vertices_49.npy', allow_pickle=True)

plt.plot(xcl_tvbo[:, 0], xcl_tvbo[:, 1], label='transfer learning bo')
plt.scatter(xcl_tvbo[:, 0], xcl_tvbo[:, 1], s=10)
plt.xlabel('x[0]')
plt.ylabel('x[1]')
plt.savefig('./figs/traj.png', dpi=600)
plt.show()




