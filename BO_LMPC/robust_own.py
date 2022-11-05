import numpy as np
import torch

from FTOCP_robust import FTOCP
# from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import matplotlib
from scipy.integrate import odeint
from tqdm import tqdm

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import pickle
from objective_functions_lqr import get_params, get_linearized_model, inv_pendulum
from bayes_opt_mine import get_model, step

from acq_func import opt_acquision
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import cvxpy
import time as ti
from control import dlqr
def main():
    np.random.seed(1)
    Ts = 0.1
    params = get_params()
    Ad = np.array([[1., 1.], [0, 1.]])
    Bd = np.array([[0.], [1.]])
    Q = np.eye(Ad.shape[0]) * 10
    R = np.eye(1) * 10
    # A = np.array([[1, 1], [0, 1]])
    # B = np.array([[0], [1]])
    # Q = np.eye(4) * 10  # np.eye(2) 非线性下真实的Q
    # R = np.eye(1)  # np.array([[1]]) 非线性下真实的R
    K, _, _ = dlqr(Ad, Bd, Q, R)
    K = -K
    # K = np.array([0.6865, 2.1963, 16.7162, 1.4913]).reshape(1, -1)
    print("Computing a first feasible trajectory")
    # Initial Condition
    # x0 = [1, 0, 0.25, -0.01]
    x0 = [4., 0.]
    # Initialize FTOCP object
    N_feas = 5
    # 产生初始可行解的时候应该Q、R随便
    ftocp_for_mpc = FTOCP(N_feas, Ad, Bd, 0.01 * Q, R, K, params)
    # ====================================================================================
    # Run simulation to compute feasible solution
    # ====================================================================================
    xcl_feasible = [x0]
    ucl_feasible = []
    xcl_feasible_true = [x0]
    ucl_feasible_true = []
    xt = x0
    time = 0
    # time Loop (Perform the task until close to the origin)
    while np.dot(xt, xt) > 10 ** (-6):
        st = xcl_feasible[time]
        xt = xcl_feasible_true[time]  # Read measurements
        bias = np.dot(K, (np.array(xt) - np.array(st)).reshape(-1, 1))[0][0]

        ftocp_for_mpc.solve(st, verbose=False)  # Solve FTOCP

        vt = ftocp_for_mpc.uPred[:, 0][0]
        ucl_feasible.append(vt)
        ut = bias + vt
        ucl_feasible_true.append(ut)
        # z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
        # xcl_feasible.append(z[1])
        xcl_feasible.append(ftocp_for_mpc.model(st, vt))
        xcl_feasible_true.append(ftocp_for_mpc.model(xt, ut))
        xcl_feasible_true[-1] = [a + np.exp(a**2/200)-1 for a in xcl_feasible_true[-1]]  # uncertainties
        # xcl_feasible.append([a + b * Ts for a, b in zip(xt, inv_pendulum(xt, 0, ut, params))])
        time += 1
    # ====================================================================================
    # Run LMPC
    # ====================================================================================

    # Initialize LMPC object
    # 这个horizon length设置成3的时候会出现infeasible的情况
    # 理论上不应该无解，已经生成可行解了，不可能无解，可能是求解器的问题
    N_LMPC = 3  # horizon length
    ftocp = FTOCP(N_LMPC, Ad, Bd, copy.deepcopy(Q), R, K, params)  # ftocp solved by LMPC，这里的Q和R在后面应该要一直变，初始值可以先用Q，R
    lmpc = LMPC(ftocp, CVX=True)  # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible, xcl_feasible_true, ucl_feasible_true)  # Add feasible trajectory to the safe set
    bayes = False
    totalIterations = 50  # Number of iterations to perform
    n_params = 2
    # lmpc.theta_update([1000, 1e-10, 1e-10, 1e-10])
    # run simulation
    # iteration loop
    print("Starting LMPC")
    returns = []
    times = []
    for it in range(0, totalIterations):
        start = ti.time()
        iters_once(x0, lmpc, Ts, params, K=K)
        # if not bayes:

        end = ti.time()
        print('time: ', end - start)
        times.append(end-start)
        returns.append(lmpc.Qfun_true[it][0])
        # ====================================================================================
        # Compute optimal solution by solving a FTOCP with long horizon
        # ====================================================================================

    tag = 'bayes' if bayes else 'no_bayes'
    np.save('./returns_' + tag + '.npy', returns)
    np.save('./times_npy', times)
    N = 100  # Set a very long horizon to fake infinite time optimal control problem
    ftocp_opt = FTOCP(N, Ad, Bd, copy.deepcopy(Q), R, K, params)
    ftocp_opt.solve(x0)
    xOpt = ftocp_opt.xPred
    uOpt = ftocp_opt.uPred
    lmpc.theta_update([1] * n_params)
    costOpt = lmpc.computeCost(xOpt.T.tolist(), uOpt.T.tolist())
    print("Optimal cost is: ", costOpt[0])
    # Store optimal solution in the lmpc object
    lmpc.optCost = costOpt[0]
    lmpc.xOpt = xOpt

    # Save the lmpc object
    filename = 'lmpc_object.pkl'
    filehandler = open(filename, 'wb')
    pickle.dump(lmpc, filehandler)


def iters_once(x0, lmpc, Ts, params, K, res=False):
    # for it in range(0, totalIterations):
    # Set initial condition at each iteration
    xcl = [x0]
    ucl = []
    xcl_true = [x0]
    ucl_true = []
    time = 0
    # time Loop (Perform the task until close to the origin)
    while np.dot(xcl_true[time], xcl_true[time]) > 10 ** (-6):
    # for time in range(20):
        # Read measurement
        st = xcl[time]
        xt = xcl_true[time]
        bias = np.dot(K, (np.array(xt)-np.array(st)).reshape(-1, 1))[0][0]
        # Solve FTOCP
        lmpc.solve(st, verbose=False)
        # Read optimal input
        vt = lmpc.uPred[:, 0][0]
        ucl.append(vt)

        ut = bias + vt

        # Apply optimal input to the system
        ucl_true.append(ut)

        xcl.append(np.array(lmpc.ftocp.model(st, vt)))

        # uncertainty = np.vstack((np.zeros((2, 1)),
        #                          np.clip(np.random.randn(2, 1) * 1e-4, -0.01, 0.01)))
        # uncertainty = np.clip(np.random.randn(4, 1) * 1e-3, -0.1, 0.1)
        xcl_true.append(np.array(lmpc.ftocp.model(xt, ut)))
        xcl_true[-1] = [a + np.exp(a**2/200)-1 for a in xcl_true[-1]]
        time += 1

    # Add trajectory to update the safe set and value function
    if not res:
        # if np.dot(xcl[time], xcl[time]) <= 10 ** (-6):
        lmpc.addTrajectory(xcl, ucl, xcl_true, ucl_true)
    # 这里对Q参数赋值，计算的是真实轨迹下真实回报,return这个值单纯是为了计算实际cost
    return lmpc.computeCost(xcl_true, ucl_true, np.eye(2) * 10)[0]


if __name__ == "__main__":
    main()
