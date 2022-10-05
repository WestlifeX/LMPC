from pyMPC.mpc import MPCController
import numpy as np
from objective_functions_lqr import get_params, get_linearized_model, inv_pendulum

np.random.seed(1)
Ts = 0.1
params = get_params()
linear_model = get_linearized_model(params, Ts)
# Define system dynamics and cost
(A, B, _, _) = linear_model
x0 = np.array([1, 0, 0.1, -0.01])
n = 4
d = 1
# Q = np.diag([9.58310635, 5.37833632, 6.94958343, 3.22360475])
Q = np.eye(4) * 10
R = np.array([[1]])
ct = MPCController(A, B, Np=10, x0=x0, xref=np.zeros(n), uref=np.zeros(d),
                   Qx=Q, QxN=Q, Qu=R, QDu=np.zeros((d, d)))

ct.setup()

xt = x0
x_traj = [x0]
u_traj = []
for i in range(10):
    ut = ct.output()
    xt = A.dot(xt) + B.dot(ut)
    ct.update(xt)
    x_traj.append(xt)
    u_traj.append(ut)

def J(x, u, N):
    c = 0
    Q = np.eye(4) * 10
    R = np.eye(1)
    for k in range(N):
        c += np.dot(np.dot(x[:, k].reshape(1, -1), Q), x[:, k].reshape(-1, 1))
        c += np.dot(np.dot(u[:, k].reshape(1, -1), R), u[:, k].reshape(-1, 1))
    c += np.dot(np.dot(x[:, N].reshape(1, -1), Q), x[:, N].reshape(-1, 1))
    return c


# 验证了mpc第一步不一定是最优的，这么做优化只能保证整体是最优的，所以需要调节参数？
print(J(np.array(x_traj).T, np.array(u_traj).T, 10))