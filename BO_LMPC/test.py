import numpy as np
import pdb
import scipy
from cvxpy import *
from objective_functions_lqr import get_params, get_linearized_model, inv_pendulum

np.random.seed(1)
Ts = 0.1
params = get_params()
linear_model = get_linearized_model(params, Ts)
# Define system dynamics and cost
(A, B, _, _) = linear_model
Q = np.diag([9.58310635, 5.37833632, 6.94958343, 3.22360475])
# Q = np.eye(4) * 10
R = np.array([[1]])

N = 100
n = 4
d = 1
x0 = [1, 0, 0.1, -0.01]
x_traj = [np.array(x0)]
u_traj = []
for it in range(100):
    cost = 0
    x = Variable((n, N + 1))
    u = Variable((d, N))
    for i in range(N):
        # cost += quad_form(x[:, i], Q) + norm((R ** 0.5) @ u[:, i]) ** 2
        cost += quad_form(x[:, i], Q) + quad_form(u[:, i], R)
    cost += quad_form(x[:, N], Q)

    constr = [x[:, 0] == x0[:]]  # initializing condition
    for i in range(N):
        constr += [x[:, i + 1] == A @ x[:, i] + B @ u[:, i]]

    problem = Problem(Minimize(cost), constr)
    problem.solve(verbose=False, solver=ECOS)
    xPred = x.value
    uPred = u.value
    u_traj.append(uPred[:, 0])
    x_traj.append(xPred[:, 1])
    x0 = xPred[:, 1]


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
print(J(np.array(x_traj).T, np.array(u_traj).T, 100))
