import numpy as np
Q = np.diag([1., 1.])
R = np.eye(1) * 1
R_delta = np.eye(1) * 1

Ad = np.array([[0.7776, -0.0045], [26.6185, 1.8555]])
Bd = np.array([[-0.0004], [0.2907]])
# Ad = np.array([[0.995, 0.095], [-0.095, 0.900]])
# Bd = np.array([[0.048], [0.95]])
# A = np.array([[1, 1], [0, 1]])
# B = np.array([[0], [1]])
# Q = np.eye(4) * 10  # np.eye(2) 非线性下真实的Q
# R = np.eye(1)  # np.array([[1]]) 非线性下真实的R
A = np.vstack((np.hstack((Ad, Bd)), np.hstack((np.zeros((Bd.shape[1], Ad.shape[1])), np.eye(Bd.shape[1])))))
B = np.vstack((Bd, np.eye(Bd.shape[1])))
x0 = [-0.3, 5.]
coef = 0.4
totalIterations = 30

G = np.array([[-0.0002, 0.0893], [0.1390, 1.2267]])
def compute_uncertainty(xt):
    return np.dot(G, np.array([np.clip(np.random.uniform(-2, 2), -2, 2),
        np.clip(np.random.uniform(-0.1, 0.1), -0.1, 0.1)]).reshape(-1, 1)).reshape(-1,)

    # return [0.02 * xt[0],
    #     0.02 * xt[1]]