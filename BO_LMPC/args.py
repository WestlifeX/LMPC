import numpy as np
Q = np.diag([1., 1.])
R = np.eye(1) * 1
R_delta = np.eye(1) * 1

def compute_uncertainty(xt):
    return [np.clip(np.sign(xt[0]) * (np.exp(xt[0] ** 2 / 50) - 1), -0.03, 0.03),
        np.clip(np.sign(xt[1]) * (-np.exp(xt[1] ** 2 / 50) + 1), -0.03, 0.03)]

    # return [0.02 * xt[0],
    #     0.02 * xt[1]]