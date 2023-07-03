import matplotlib.pyplot as plt
import numpy as np

xcl_tvbo = np.load('tvbo_3_xcl.npy')
xcl_lmpc = np.load('own_xcl.npy')
v_tvbo = np.load('./vertices/tvbo_1/vertices_29.npy', allow_pickle=True)
v_lmpc = np.load('./vertices/own/vertices_29.npy', allow_pickle=True)
colors = ['sandybrown', 'purple']
plt.plot(xcl_lmpc[:, 0], xcl_lmpc[:, 1], lw=2, label='RLMPC', color=colors[0])
plt.scatter(xcl_lmpc[:, 0], xcl_lmpc[:, 1], lw=2, s=20, color=colors[0])
plt.plot(xcl_tvbo[:, 0], xcl_tvbo[:, 1], lw=2, label='Efficient BO', color=colors[1], linestyle='dashed')
plt.scatter(xcl_tvbo[:, 0], xcl_tvbo[:, 1], s=20, color=colors[1])
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
font = {'family': 'Times New Roman', 'size': 16}
plt.grid()
plt.xlabel('x[0]', font)
plt.ylabel('x[1]', font)
plt.tick_params(labelsize=12)
plt.legend(prop=font)
plt.savefig('./figs/traj_cstr.jpg', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()




