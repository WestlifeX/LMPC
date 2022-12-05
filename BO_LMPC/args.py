import numpy as np
Q = np.diag([2., 2.])
R = np.eye(1) * 1
R_delta = np.eye(1) * 1

Ad = np.array([[1.2, 1.5], [0, 1.3]])
Bd = np.array([[0.], [1.]])
# Ad = np.array([[0.995, 0.095], [-0.095, 0.900]])
# Bd = np.array([[0.048], [0.95]])
# A = np.array([[1, 1], [0, 1]])
# B = np.array([[0], [1]])
# Q = np.eye(4) * 10  # np.eye(2) 非线性下真实的Q
# R = np.eye(1)  # np.array([[1]]) 非线性下真实的R
A = np.vstack((np.hstack((Ad, Bd)), np.hstack((np.zeros((Bd.shape[1], Ad.shape[1])), np.eye(Bd.shape[1])))))
B = np.vstack((Bd, np.eye(Bd.shape[1])))
<<<<<<< HEAD
x0 = [4., 1.]
=======
x0 = [-6, -0.6]
>>>>>>> 29931cf41752ed03c8653bef4114a2007c032d90
coef = 0.04
totalIterations = 50
def compute_uncertainty(xt):
    return [np.clip(np.sign(xt[0]) * (np.exp(xt[0] ** 2 / 10) - 1), -0.03, 0.03),
        np.clip(np.sign(xt[1]) * (-np.exp(xt[1] ** 2 / 10) + 1), -0.03, 0.03)]

    # return [0.02 * xt[0],
    #     0.02 * xt[1]]