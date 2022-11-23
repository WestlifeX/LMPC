import numpy as np
from FTOCP_casadi import FTOCP
from LMPC import LMPC
import matplotlib
from tqdm import tqdm
from control import dlqr

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy

from objective_functions_lqr import get_params
from args import Q, R, R_delta, compute_uncertainty, A, B, Ad, Bd

import arguments

def iters_once(x0, lmpc, Ts, params, K, SS=None, Qfun=None):
    # for it in range(0, totalIterations):
    # Set initial condition at each iteration
    xcl = [x0]
    ucl = []
    xcl_true = [x0]
    ucl_true = []
    time = 0
    st = x0
    # time Loop (Perform the task until close to the origin)
    while np.dot(st, st) > 10 ** (-6):
    # for time in range(20):
        # Read measurement
        st = xcl[time]
        xt = xcl_true[time]
        bias = np.dot(K, (np.array(xt)-np.array(st)).reshape(-1, 1))[0][0]
        # Solve FTOCP

        if SS is not None and Qfun is not None:
            lmpc.solve(st, time=time, verbose=False, SS=SS, Qfun=Qfun)
        else:
            lmpc.solve(st, time=time, verbose=False)
        # Read optimal input
        # Read optimal input
        try:
            vt = lmpc.uPred[:, 0][0]
        except TypeError:
            return None
        ucl.append(vt)

        ut = bias + vt
        if abs(ut) > 1 or abs(xt[0]) > 10 or abs(xt[1]) > 10:
            a = 1
        # Apply optimal input to the system
        ucl_true.append(ut)
        # z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
        # xcl.append(z[1])
        # xcl.append(lmpc.xPred[:, 1])
        # xcl.append([a + b * Ts for a, b in zip(xt, inv_pendulum(xt, 0, ut, params))])

        xcl.append(np.array(lmpc.ftocp.model(st, vt)))
        xcl_true.append(np.array(lmpc.ftocp.model(xt, ut)))
        uncertainty = compute_uncertainty(xt)
        xcl_true[-1] = [a + b for a, b in zip(xcl_true[-1], uncertainty)]
        time += 1
        if time >= 50:
            break
    # Add trajectory to update the safe set and value function

    return lmpc.computeCost(xcl_true, ucl_true, Q, R, R_delta)[0], xcl, ucl, xcl_true, ucl_true
args = arguments.get_args()
np.random.seed(1)
Ts = 0.1
params = get_params()
K, _, _ = dlqr(Ad, Bd, Q, R)
K = -K
print("Computing a first feasible trajectory")
x0 = [4., 1.]
# Initialize FTOCP object
N_feas = 10
# 产生初始可行解的时候应该Q、R随便
ftocp_for_mpc = FTOCP(N_feas, Ad, Bd, 0.01 * Q, R, R_delta, K, params)
# ====================================================================================
# Run simulation to compute feasible solution
# ====================================================================================
xcl_feasible = [x0]
ucl_feasible = []
xcl_feasible_true = [x0]
ucl_feasible_true = []
st = x0
time = 0
# time Loop (Perform the task until close to the origin)
while np.dot(st, st) > 10 ** (-10):
    st = xcl_feasible[time]
    xt = xcl_feasible_true[time]  # Read measurements
    bias = np.dot(K, (np.array(xt) - np.array(st)).reshape(-1, 1))[0][0]

    ftocp_for_mpc.solve(st, time=time, verbose=False)  # Solve FTOCP

    vt = ftocp_for_mpc.uPred[:, 0][0]
    ucl_feasible.append(vt)
    ut = bias + vt
    ucl_feasible_true.append(ut)
    # z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
    # xcl_feasible.append(z[1])
    xcl_feasible.append(ftocp_for_mpc.model(st, vt))
    xcl_feasible_true.append(ftocp_for_mpc.model(xt, ut))
    uncertainty = compute_uncertainty(xt)
    xcl_feasible_true[-1] = [a + b for a, b in zip(xcl_feasible_true[-1], uncertainty)]  # uncertainties
    # xcl_feasible.append([a + b * Ts for a, b in zip(xt, inv_pendulum(xt, 0, ut, params))])
    time += 1
    if time >= 50:
        break
# ====================================================================================
# Run LMPC
N_LMPC = 3  # horizon length
ftocp = FTOCP(N_LMPC, Ad, Bd, copy.deepcopy(Q), copy.deepcopy(R), copy.deepcopy(R_delta), K, params) # ftocp solved by LMPC，这里的Q和R在后面应该要一直变，初始值可以先用Q，R
lmpc = LMPC(ftocp, CVX=True)  # Initialize the LMPC (decide if you wanna use the CVX hull)
lmpc.addTrajectory(xcl_feasible, ucl_feasible, xcl_feasible_true, ucl_feasible_true)  # Add feasible trajectory to the safe set
bayes = True
totalIterations = 3  # Number of iterations to perform
n_params = 2
length = 100
samples = np.linspace(1, length, length)
x = np.linspace(1, length, length)
y = np.linspace(1, length, length)
xx, yy = np.meshgrid(x, y)
all_res = np.zeros((length, length, totalIterations))
fig = plt.figure()
# fig, ax = plt.subplots(1, 3, figsize=[13, 6])
for it in range(0, totalIterations):
    for i in tqdm(range(length)):
        for j in range(length):
            theta = [samples[i], samples[j]]
            lmpc.theta_update(theta)
            K, _, _ = dlqr(Ad, Bd, lmpc.Q, lmpc.R)
            K = -K
            lmpc.ftocp.K = K
            lmpc.ftocp.compute_mrpi()
            res, _, _, _, _ = iters_once(x0, lmpc, Ts, params, K=K)
            all_res[i, j, it] = res
    if it == 0:
        ax = fig.add_subplot(131, projection='3d')
        ax.plot_surface(xx, yy, all_res[:, :, it], cmap=plt.get_cmap('rainbow'))
    elif it == 1:
        ax = fig.add_subplot(132, projection='3d')
        ax.plot_surface(xx, yy, all_res[:, :, it], cmap=plt.get_cmap('rainbow'))
    else:
        ax = fig.add_subplot(133
                             , projection='3d')
        ax.plot_surface(xx, yy, all_res[:, :, it], cmap=plt.get_cmap('rainbow'))

    train_x = np.random.uniform(1., 100.,
                                size=(2, ))
    lmpc.theta_update(train_x.tolist())
    K, _, _ = dlqr(Ad, Bd, lmpc.Q, lmpc.R)
    K = -K
    lmpc.ftocp.K = K
    lmpc.ftocp.compute_mrpi()
    _, xcl, ucl, xcl_true, ucl_true = \
        iters_once(x0, lmpc, Ts, params, K=K)
    lmpc.addTrajectory(xcl, ucl, xcl_true, ucl_true)
plt.subplots_adjust(wspace=1.0)
plt.savefig('./figs/similarity.png', dpi=600, bbox_inches='tight', pad_inches=0)
plt.show()
a = 1


