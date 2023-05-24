import numpy as np
import torch
from args import Q, R, R_delta, compute_uncertainty, A, B, Ad, Bd
from FTOCP_casadi import FTOCP
# from FTOCP_robust import FTOCP
from LMPC import LMPC
import pdb
import matplotlib
from scipy.integrate import odeint
from tqdm import tqdm
from control import dlqr
import cvxpy
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import pickle
from objective_functions_lqr import get_params, get_linearized_model, inv_pendulum

from acq_func import opt_acquision
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
import time as tim
from scipy.linalg import block_diag

def main():
    # args = arguments.get_args()
    np.random.seed(1)
    Ts = 0.1
    params = get_params()

    K, _, _ = dlqr(Ad, Bd, Q, R)
    K = -K
    # K = np.array([1.7, 3.3]).reshape(1, -1)
    # K = -K
    print("Computing a first feasible trajectory")
    # Initial Condition
    # x0 = [1, 0, 0.25, -0.01]
    # x0 = [-2., 6.]
    x0 = [4., 1.]
    # Initialize FTOCP object
    N_feas = 10
    # 产生初始可行解的时候应该Q、R随便
    ftocp_for_mpc = FTOCP(N_feas, Ad, Bd, 0.04 * Q, R, R_delta, K, params)
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
    # ====================================================================================

    # Initialize LMPC object
    # 这个horizon length设置成3的时候会出现infeasible的情况
    # 理论上不应该无解，已经生成可行解了，不可能无解，可能是求解器的问题
    N_LMPC = 3  # horizon length
    ftocp = FTOCP(N_LMPC, Ad, Bd, copy.deepcopy(Q), copy.deepcopy(R), copy.deepcopy(R_delta), K, params)  # ftocp solved by LMPC，这里的Q和R在后面应该要一直变，初始值可以先用Q，R
    lmpc = LMPC(ftocp, CVX=True)  # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible, xcl_feasible_true, ucl_feasible_true)  # Add feasible trajectory to the safe set
    bayes = True
    totalIterations = 50  # Number of iterations to perform
    n_params = 3
    theta_bounds = np.array([[1., 100.]] * (n_params))
    # lmpc.theta_update([5.23793828, 50.42607759, 30.01345335, 30.14379343])
    # run simulation
    print("Starting LMPC")
    returns = []
    # train_x = torch.FloatTensor(n_inital_points, len(theta)).uniform_(theta_bounds[0][0], theta_bounds[0][1])
    thresh = 1e-7
    last_params = np.array([1] * (n_params)).reshape(1, -1)
    times = []
    xcls = []
    ucls = []
    xcls_true = []
    ucls_true = []

    for it in range(0, totalIterations):
        start = tim.time()
        vertices = []
        Kx = []
        # bayes opt
        # theta_bounds[:n_params-1, 0] = last_params[0, :n_params-1] / 3
        # theta_bounds[:n_params-1, 1] = last_params[0, :n_params-1] * 3
        # theta_bounds = np.clip(theta_bounds, 0, 100)
        print("Initializing")
        if it == 0:
            n_inital_points = 10
            n_iters = 0
            train_x = np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1],
                                        size=(n_inital_points, theta_bounds.shape[0]))
            train_y = []

            for i in tqdm(range(n_inital_points)):
                lmpc.theta_update(train_x[i].tolist())
                K, _, _ = dlqr(Ad, Bd, lmpc.Q, lmpc.R)
                K = -K
                lmpc.ftocp.K = K
                lmpc.ftocp.compute_mrpi()

                train_obj, xcl, ucl, xcl_true, ucl_true = \
                    iters_once(x0, lmpc, Ts, params, K=K)
                train_y.append(train_obj)
                xcls.append(xcl)
                ucls.append(ucl)
                xcls_true.append(xcl_true)
                ucls_true.append(ucl_true)

            train_y = np.array(train_y).reshape(-1, 1)
        else:
            n_inital_points = 0
            n_iters = 10
            
        if train_x.shape[0] > 50:
            train_x = train_x[-50:, :]
            train_y = train_y[-50:, :]
        # model = gp.GaussianProcess(kernel, 0.001)
        model = GaussianProcessRegressor(kernel=kernels.RBF())
        model.fit(train_x, train_y)
        # model.fit(train_x, train_y)
        # model, mll = get_model(train_x, train_y)
        print('bayes opt for {} iteration'.format(it + 1))
        for idx in tqdm(range(n_iters)):
            # beta = 1
            # beta = np.sqrt(beta)
            beta = 1
            next_sample = opt_acquision(model, theta_bounds, beta=beta, ts=False)
            # 避免出现重复数据影响GP的拟合
            if np.any(np.abs(next_sample - train_x) <= thresh):
                next_sample = np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1], theta_bounds.shape[0])
            lmpc.theta_update(next_sample.tolist())
            K, _, _ = dlqr(Ad, Bd, lmpc.Q, lmpc.R)
            K = -K
            lmpc.ftocp.K = K
            lmpc.ftocp.compute_mrpi()
            try:
                new_res, xcl, ucl, xcl_true, ucl_true = \
                    iters_once(x0, lmpc, Ts, params, K=K)
            except AttributeError:
                a = 1

            xcls.append(xcl)
            ucls.append(ucl)
            xcls_true.append(xcl_true)
            ucls_true.append(ucl_true)
            train_y = np.append(train_y, new_res).reshape(-1, 1)
            train_x = np.vstack((train_x, next_sample.reshape(1, -1)))

            model = GaussianProcessRegressor(kernel=kernels.RBF())
            model.fit(train_x, train_y)

        theta = train_x[-(n_inital_points+n_iters):][np.argmin(train_y[-(n_inital_points+n_iters):], axis=0)[0]]
        # theta = train_x[:][np.argmin(train_y[:], axis=0)[0]]
        lmpc.theta_update(theta.tolist())
        K, _, _ = dlqr(Ad, Bd, lmpc.Q, lmpc.R)
        K = -K
        lmpc.ftocp.K = K
        lmpc.ftocp.compute_mrpi()
        res, xcl, ucl, xcl_true, ucl_true = \
            iters_once(x0, lmpc, Ts, params, K=K)
        lmpc.addTrajectory(xcl, ucl, xcl_true, ucl_true)
        # train_y[np.argmin(train_y[:], axis=0)] = res

        # lmpc.addTrajectory(xcls[np.argmin(train_y[:], axis=0)[0]],
        #                    ucls[np.argmin(train_y[:], axis=0)[0]],
        #                    xcls_true[np.argmin(train_y[:], axis=0)[0]],
        #                    ucls_true[np.argmin(train_y[:], axis=0)[0]])
        last_params = copy.deepcopy(theta.reshape(1, -1))
        print('optimized theta: ', last_params)

        end = tim.time()
        print('consumed time: ', end - start)
        times.append(end-start)
        returns.append(lmpc.Qfun_true[it][0])
        # 存一下每次迭代最好的那个点的tube，画个图
        for i in range(len(lmpc.ftocp.F_list)):
            vertices.append(lmpc.ftocp.F_list[i].vertices)
            Kx.append(lmpc.ftocp.Kxs[i].vertices)
        np.save('./vertices/tvbo_3/vertices_{}.npy'.format(it), vertices)
        np.save('./vertices/tvbo_3/Kxs_{}.npy'.format(it), Kx)
        # ====================================================================================
        # Compute optimal solution by solving a FTOCP with long horizon
        # ====================================================================================
    print(min(returns))
    print(np.argmin(returns))
    tag = 'bayes' if bayes else 'no_bayes'
    np.save('./returns_' + tag + '.npy', returns)
    np.save('tvbo_3_xcl_true.npy', xcls_true[np.argmin(returns[:], axis=0)])
    np.save('tvbo_3_ucl_true.npy', ucls_true[np.argmin(returns[:], axis=0)])
    np.save('tvbo_3_xcl.npy', xcls[np.argmin(returns[:], axis=0)])
    np.save('tvbo_3_ucl.npy', ucls[np.argmin(returns[:], axis=0)])
    N = 100  # Set a very long horizon to fake infinite time optimal control problem
    # K, _, _ = dlqr(Ad, Bd, Q, R)
    # K = -K
    ftocp_opt = FTOCP(N, Ad, Bd, copy.deepcopy(Q), R, R_delta, K, params)
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


def iters_once(x0, lmpc, Ts, params, K, SS=None, Qfun=None):
    # for it in range(0, totalIterations):
    # Set initial condition at each iteration
    Ki = np.array([[-0.52746546, -1.82539112]])
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
        bias1 = np.dot(K, (np.array(xt)-np.array(st)).reshape(-1, 1))[0][0]
        bias2 = np.dot(Ki, (np.array(xt)-np.array(st)).reshape(-1, 1))[0][0] + 1e3
        if abs(bias1) < abs(bias2):
            bias = bias1
        else:
            bias = bias2

        # Solve FTOCP
        if SS is not None and Qfun is not None:
            lmpc.solve(st, time=time, verbose=False, SS=SS, Qfun=Qfun)
        else:
            lmpc.solve(st, time=time, verbose=False)
        # Read optimal input
        # Read optimal input
        try:
            vt = lmpc.uPred[:, 0][0]
        except IndexError:
            return None
        ucl.append(vt)

        ut = bias + vt
        if abs(ut) > 1 or abs(xt[0]) > 10 or abs(xt[1]) > 10:
            a = 1
            # raise AttributeError
        # Apply optimal input to the system
        ucl_true.append(ut)
        # z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
        # xcl.append(z[1])
        # xcl.append(lmpc.xPred[:, 1])
        # xcl.append([a + b * Ts for a, b in zip(xt, inv_pendulum(xt, 0, ut, params))])

        xcl.append(np.array(lmpc.ftocp.model(st, vt)))
        # uncertainty = xcl[-1] ** 2 * 1e-3
        # uncertainty = np.clip(np.random.randn(4, 1) * 1e-3, -0.1, 0.1)
        # uncertainty[1] = 0
        # uncertainty[3] = 0
        xcl_true.append(np.array(lmpc.ftocp.model(xt, ut)))
        uncertainty = compute_uncertainty(xt)
        xcl_true[-1] = [a + b for a, b in zip(xcl_true[-1], uncertainty)]
        time += 1
        if time >= 50:
            break
    # Add trajectory to update the safe set and value function

    return lmpc.computeCost(xcl_true, ucl_true, Q, R, R_delta)[0], xcl, ucl, xcl_true, ucl_true


if __name__ == "__main__":
    main()
