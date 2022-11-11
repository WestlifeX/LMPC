import numpy as np
Q = np.diag([1., 1.])
R = np.eye(1) * 1
R_delta = np.eye(1) * 1

def compute_uncertainty(xt):
    return [np.clip(np.sign(xt[0]) * (np.exp(xt[0] ** 2 / 600) - 1), -0.2, 0.2),
        np.clip(np.sign(xt[1]) * (-np.exp(xt[1] ** 2 / 600) + 1), -0.2, 0.2)]