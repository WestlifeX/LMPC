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

from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel
import gaussian_process as gp
import kernel as kn
from acq_func import opt_acquision
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import cvxpy
import time as ti
def main():
    np.random.seed(4)
    Ts = 0.1
    params = get_params()
    linear_model = get_linearized_model(params, Ts)
    # Define system dynamics and cost
    (Ad, Bd, A, B) = linear_model
    # A = np.array([[1, 1], [0, 1]])
    # B = np.array([[0], [1]])
    Q = np.eye(4) * 10
    R = np.eye(1)
    # K_m = np.array([0.0100,  1.0403 , 10.4141,  0.0238]).reshape(1, -1) # robust MPC fixed gain
    K = np.array([0.6865, 2.1963, 16.7162, 1.4913]).reshape(1, -1)  # robust MPC fixed gain
    # K = np.array([-1.8592e3, -0.1574e3, -0.3240e3, -0.0180e3]).reshape(1, -1)
    print("Computing a first feasible trajectory")
    # Initial Condition

    # simple model condition
    x0 = [2, 0, 0.25, -0.01]  # 99.20
    # Initialize FTOCP object
    N_feas = 10
    # 产生初始可行解的时候应该Q、R随便
    ftocp_for_mpc = FTOCP(N_feas, Ad, Bd, Q, R, K, params)
    # ====================================================================================
    # Run simulation to compute feasible solution
    # ====================================================================================
    xcl_feasible = [x0]
    ucl_feasible = []
    xt = x0
    time = 0
    # time Loop (Perform the task until close to the origin)
    while np.dot(xt, xt) > 10 ** (-6):
        xt = xcl_feasible[time]  # Read measurements

        ftocp_for_mpc.solve(xt, verbose=False)  # Solve FTOCP

        # ucl_feasible = ftocp_for_mpc.uPred.T.tolist()

        # Apply optimal input to the system
        # ucl.append(ut)
        # Read input and apply it to the system
        ut = ftocp_for_mpc.uPred[:, 0][0]
        ucl_feasible.append(ut)
        # z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
        # xcl_feasible.append(z[1])
        # xcl_feasible.append(ftocp_for_mpc.xPred[:, 1])
        # xcl_feasible.append([a + b * Ts for a, b in zip(xt, inv_pendulum(xt, 0, ut, params))])
        xcl_feasible.append(ftocp_for_mpc.model(xcl_feasible[time], ut))
        time += 1
    # ====================================================================================
    # Run LMPC
    # ====================================================================================

    # Initialize LMPC object
    # 这个horizon length设置成3的时候会出现infeasible的情况
    # 理论上不应该无解，已经生成可行解了，不可能无解，可能是求解器的问题
    N_LMPC = 5  # horizon length
    ftocp = FTOCP(N_LMPC, Ad, Bd, copy.deepcopy(Q), R, K, params)  # ftocp solved by LMPC，这里的Q和R在后面应该要一直变，初始值可以先用Q，R
    lmpc = LMPC(ftocp, CVX=True)  # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible)  # Add feasible trajectory to the safe set
    bayes = False
    totalIterations = 0  # Number of iterations to perform
    n_params = 4
    theta_bounds = np.array([[0.1, 1000]] * n_params)
    # lmpc.theta_update([1000, 1e-10, 1e-10, 1e-10])
    # run simulation
    # iteration loop
    print("Starting LMPC")
    returns = []
    n_inital_points = 5
    n_iters = 5
    # train_x = torch.FloatTensor(n_inital_points, len(theta)).uniform_(theta_bounds[0][0], theta_bounds[0][1])
    thresh = 1e-7
    last_params = np.array([1] * n_params).reshape(1, -1)
    times = []
    for it in range(0, totalIterations):
        start = ti.time()
        if not bayes:
            # pass
            iters_once(x0, lmpc, Ts, params, K=K)
        else:
            # bayes opt
            print("Initializing")
            x_init = np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1],
                                        size=(n_inital_points, theta_bounds.shape[0]))
            train_x = []
            train_y = []
            n_None = 0
            for i in tqdm(range(n_inital_points)):
                lmpc.theta_update(x_init[i].tolist())
                train_obj = iters_once(x0, lmpc, Ts, params, K=K, res=True)  # 这里取个负号，因为我们的目标是取最小，而这个BO是找最大点
                if train_obj is not None:
                    train_x.append(x_init[i])
                    train_y.append(train_obj)
                else:
                    n_None += 1
            train_x = np.array(train_x).reshape(-1, 4)
            train_y = np.array(train_y).reshape(-1, 1)

            model = GaussianProcessRegressor(kernel=kernels.RBF(), n_restarts_optimizer=5, normalize_y=False)
            model.fit(train_x, train_y)
            # model.fit(train_x, train_y)
            # model, mll = get_model(train_x, train_y)
            print('bayes opt for {} iteration'.format(it + 1))

            for idx in tqdm(range(n_iters)):
                beta = 2 * np.log((idx + 1) ** 2 * 2 * np.pi ** 2 / (3 * 0.05)) + \
                       2 * n_params * np.log(
                    (idx + 1) ** 2 * n_params * 1e-4 * 1000 * np.sqrt(np.log(4 * n_params * 0.1 / 0.05)))
                beta = np.sqrt(beta)
                next_sample = opt_acquision(model, theta_bounds, beta=beta, ts=False)
                # 避免出现重复数据影响GP的拟合
                if np.any(np.abs(next_sample - train_x) <= thresh):
                    next_sample = np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1], theta_bounds.shape[0])
                lmpc.theta_update(next_sample.tolist())
                new_res = iters_once(x0, lmpc, Ts, params, K=K, res=True)
                if new_res is not None:
                    train_y = np.vstack((train_y, new_res))
                    train_x = np.vstack((train_x, next_sample.reshape(1, -1)))

                    model.fit(train_x, train_y)
                else:
                    n_None += 1

            theta = train_x[-(n_inital_points+n_iters-n_None):][np.argmin(train_y[-(n_inital_points+n_iters-n_None):], axis=0)]
            lmpc.theta_update(theta.tolist()[0])
            iters_once(x0, lmpc, Ts, params, K=K)
            last_params = copy.deepcopy(theta.reshape(1, -1))
            print('optimized theta: ', last_params)

            # mean_module = model.mean_module
            # covar_module = model.covar_module
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
    # while np.dot(xcl[time], xcl[time]) > 10 ** (-6):
    for time in range(100):
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
        # z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
        # xcl.append(z[1])
        # xcl.append(lmpc.xPred[:, 1])
        # xcl.append([a + b * Ts for a, b in zip(xt, inv_pendulum(xt, 0, ut, params))])

        xcl.append(np.array(lmpc.ftocp.model(st, vt)))

        # uncertainty = np.vstack((np.zeros((2, 1)),
        #                          np.clip(np.random.randn(2, 1) * 1e-4, -0.01, 0.01)))
        uncertainty = np.clip(np.random.randn(4, 1) * 1e-3, -0.1, 0.1)
        uncertainty[1] = 0
        uncertainty[3] = 0
        xcl_true.append(np.array(lmpc.ftocp.model(xt, ut)) + uncertainty.reshape(-1, ))
        time += 1

    # Add trajectory to update the safe set and value function
    if not res:
        # if np.dot(xcl[time], xcl[time]) <= 10 ** (-6):
        lmpc.addTrajectory(xcl, ucl, xcl_true, ucl_true)
    # 这里对Q参数赋值，计算的是真实轨迹下真实回报,return这个值单纯是为了计算实际cost
    return lmpc.computeCost(xcl_true, ucl_true, np.eye(4) * 10)[0]


if __name__ == "__main__":
    main()
