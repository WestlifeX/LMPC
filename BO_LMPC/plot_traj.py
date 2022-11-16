import matplotlib.pyplot as plt
import numpy as np

xcl_tvbo = np.load('tvbo_3_xcl_true.npy')
xcl_lmpc = np.load('own_xcl_true.npy')
plt.plot(xcl_tvbo[:, 0], xcl_tvbo[:, 1], label='tvbo')
plt.scatter(xcl_tvbo[:, 0], xcl_tvbo[:, 1], s=10)

plt.plot(xcl_lmpc[:, 0], xcl_lmpc[:, 1], label='lmpc')
plt.scatter(xcl_lmpc[:, 0], xcl_lmpc[:, 1], s=10)

plt.legend()
plt.show()
v_tvbo = np.load('./vertices/tvbo_2/vertices_49.npy', allow_pickle=True)
v_lmpc = np.load('./vertices/own/vertices_49.npy', allow_pickle=True)
