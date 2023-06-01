import numpy as np
Q = np.diag([1., 1.]) * 2
R = np.eye(1) * 1
R_delta = np.eye(1) * 1

Ad = np.array([[0.745, -0.002], [5.610, 0.780]])
Bd = np.array([[5.6e-6], [0.464]])
# Ad = np.array([[0.995, 0.095], [-0.095, 0.900]])
# Bd = np.array([[0.048], [0.95]])
# A = np.array([[1, 1], [0, 1]])
# B = np.array([[0], [1]])
# Q = np.eye(4) * 10  # np.eye(2) 非线性下真实的Q
# R = np.eye(1)  # np.array([[1]]) 非线性下真实的R
A = np.vstack((np.hstack((Ad, Bd)), np.hstack((np.zeros((Bd.shape[1], Ad.shape[1])), np.eye(Bd.shape[1])))))
B = np.vstack((Bd, np.eye(Bd.shape[1])))
x0 = [-0.5, -6.]
coef = 0.4
totalIterations = 20

# G = np.array([[-0.0002, 0.0893], [0.1390, 1.2267]])
def compute_uncertainty(xt):
    # return [np.clip(np.sign(xt[0]) * (np.exp(xt[0] ** 2 / 10) - 1), -0.03, 0.03),
    #         np.clip(np.sign(xt[1]) * (-np.exp(xt[1] ** 2 / 10) + 1), -0.03, 0.03)]
    return [np.clip(np.random.uniform(-0.05, 0.05), -0.05, 0.05),
        np.clip(np.random.uniform(-0.05, 0.05), -0.05, 0.05)]

    # return np.dot(G, np.array([np.clip(np.random.uniform(-2, 2), -2, 2),
    #     np.clip(np.random.uniform(-0.1, 0.1), -0.1, 0.1)]).reshape(-1, 1)).reshape(-1,)

    # return [0.02 * xt[0],
    #     0.02 * xt[1]]