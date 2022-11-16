import matplotlib.pyplot as plt
import numpy as np

xcl = np.load('tvbo_3_xcl_true.npy')
plt.plot(xcl[:, 0], xcl[:, 1])
plt.scatter(xcl[:, 0], xcl[:, 1], s=10)
plt.show()