import numpy as np
import torch

from FTOCP import FTOCP
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
import time as tim


def main():
    np.random.seed(4)
    Ts = 0.1
    params = get_params()
    linear_model = get_linearized_model(params, Ts)
    # Define system dynamics and cost
    (Ad, Bd, A, B) = linear_model
    # A = np.array([[1, 1], [0, 1]])
    # B = np.array([[0], [1]])
    Q = np.eye(4) * 10  # np.eye(2) 非线性下真实的Q
    R = np.eye(1)  # np.array([[1]]) 非线性下真实的R

    print("Computing a first feasible trajectory")
    # Initial Condition
    x0 = [1, 0, 0.2, -0.01]

    # Initialize FTOCP object
    N_feas = 10
    # 产生初始可行解的时候应该Q、R随便
    # 求解MPC应该也是用线性模型，因为MPC是为了求解u，而求u应该用不准确的模型，否则就没有误差了，但是得到u之后求下一步x用非线性的
    # 修改了初值的位置，因为x离原点太远了，把cart调回原点需要太多步了（x的初值应该是可以改的，因为线性化只针对phi）
    # 增大了采样时间0.02 --> 0.1，同上，采样时间越小需要的步数就越多
    # 增大了Q（原来是0.01*Q），Q大一点应该可以快点收敛吧
    ftocp_for_mpc = FTOCP(N_feas, Ad, Bd, Q, R, params)
    # ====================================================================================
    # Run simulation to compute feasible solution
    # ====================================================================================
    xcl_feasible = [x0]
    ucl_feasible = []
    xt = x0
    time = 0
    # time Loop (Perform the task until close to the origin)
    while np.dot(xt, xt) > 10 ** (-3):
        xt = xcl_feasible[time]  # Read measurements

        ftocp_for_mpc.solve(xt, verbose=False)  # Solve FTOCP

        # ucl_feasible = ftocp_for_mpc.uPred.T.tolist()

        # Apply optimal input to the system
        # ucl.append(ut)
        # for i in range(len(ucl_feasible)):
        # Read input and apply it to the system
        ut = ftocp_for_mpc.uPred[:, 0][0]
        ucl_feasible.append(ut)
        # z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
        # xcl_feasible.append(z[1])
        xcl_feasible.append(ftocp_for_mpc.model(xcl_feasible[time], ut))
        # xcl_feasible.append([a + b * Ts for a, b in zip(xt, inv_pendulum(xt, 0, ut, params))])
        time += 1

    # print(np.round(np.array(xcl_feasible).T, decimals=2))
    # print(np.round(np.array(ucl_feasible).T, decimals=2))

    # ====================================================================================
    # Run LMPC
    # ====================================================================================

    # Initialize LMPC object
    # 这个horizon length设置成3的时候会出现infeasible的情况
    # 理论上不应该无解，已经生成可行解了，不可能无解，可能是求解器的问题
    N_LMPC = 5  # horizon length
    ftocp = FTOCP(N_LMPC, Ad, Bd, copy.deepcopy(Q), R, params)  # ftocp solved by LMPC，这里的Q和R在后面应该要一直变，初始值可以先用Q，R
    lmpc = LMPC(ftocp, CVX=True)  # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible)  # Add feasible trajectory to the safe set
    bayes = True
    totalIterations = 200  # Number of iterations to perform
    n_params = 4
    theta_bounds = np.array([[0.5, 2]] * n_params)
    # lmpc.theta_update([5.23793828, 50.42607759, 30.01345335, 30.14379343])
    # run simulation
    print("Starting LMPC")
    returns = []

    n_inital_points = 1
    n_iters = 3
    # train_x = torch.FloatTensor(n_inital_points, len(theta)).uniform_(theta_bounds[0][0], theta_bounds[0][1])
    thresh = 1e-7
    last_params = np.array([1] * n_params).reshape(1, -1)
    mu_init = 1
    tau_init = 1e10 - 1
    tau_s = [tau_init]
    mu_s = [mu_init]
    times = []

    for it in range(0, totalIterations):
        start = tim.time()
        if not bayes:
            # pass
            iters_once(x0, lmpc, Ts, params)
        else:

            # bayes opt
            theta_bounds[:, 0] = last_params / 2
            theta_bounds[:, 1] = last_params * 2
            theta_bounds = np.clip(theta_bounds, 0, 1000)
            xcls = []
            ucls = []
            print("Initializing")
            if it == 0:
                train_x = np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1],
                                            size=(n_inital_points, theta_bounds.shape[0]))
                train_y = []
                for i in tqdm(range(n_inital_points)):
                    lmpc.theta_update(train_x[i].tolist())
                    train_obj, xcl, ucl = iters_once(x0, lmpc, Ts, params, res=True)  # 这里取个负号，因为我们的目标是取最小，而这个BO是找最大点
                    train_y.append(train_obj)
                    xcls.append(xcl)
                    ucls.append(ucl)
                train_y = np.array(train_y).reshape(-1, 1)
            else:
                mu_s.append(mu_init)
                tau_s.append(tau_init)
                train_x = np.vstack((train_x, np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1],
                                                                size=(n_inital_points, theta_bounds.shape[0]))))
                y_t = []
                bias = 0
                for i in tqdm(range(n_inital_points)):
                    lmpc.theta_update(train_x[i - n_inital_points].tolist())
                    train_obj, xcl, ucl = iters_once(x0, lmpc, Ts, params, res=True)  # 新数据
                    xcls.append(xcl)
                    ucls.append(ucl)

                    source_obj, _, _ = iters_once(x0, lmpc, Ts, params, res=True, SS=lmpc.SS[:-1],
                                                  Qfun=lmpc.Qfun[:-1])  # 原数据
                    bias += (train_obj - source_obj) ** 2
                    mu_s[:-1] = [m + bias / 2 for m in mu_s[:-1]]
                    tau_s[-2] += 1 / 2
                    # mu_s = [m + bias / 2 for m in mu_s]
                    # tau_s[-(i+1):] = [t + 1 / 2 for t in tau_s[-(i+1):]]
                    y_t.append(train_obj)
                y_t = np.array(y_t).reshape(-1, 1)
                # y_t = np.squeeze(y_t, axis=1)
                train_y = np.vstack([train_y, y_t])

            alpha = np.ones(train_x.shape[0]) * 1e-10
            for i in range(len(mu_s) - 1):
                alpha[i * (n_inital_points + n_iters):(i + 1) * (n_inital_points + n_iters)] = mu_s[i] / (1 + tau_s[i])
            # alpha[-n_inital_points:] = mu_s[-1] / (1 + tau_s[-1])
            # if train_x.shape[0] > 100:
            #     train_x = train_x[-100:, :]
            #     train_y = train_y[-100:, :]
            # model = gp.GaussianProcess(kernel, 0.001)
            model = GaussianProcessRegressor(kernel=kernels.RBF(), alpha=alpha, n_restarts_optimizer=5,
                                             normalize_y=False)
            model.fit(train_x, train_y)
            # model.fit(train_x, train_y)
            # model, mll = get_model(train_x, train_y)
            print('bayes opt for {} iteration'.format(it + 1))
            for idx in tqdm(range(n_iters)):
                beta = 2 * np.log((idx + 1) ** 2 * 2 * np.pi ** 2 / (3 * 0.05)) + \
                       2 * n_params * np.log(
                    (idx + 1) ** 2 * n_params * 1e-4 * 1000 * np.sqrt(np.log(4 * n_params * 0.1 / 0.05)))
                beta = np.sqrt(beta)
                # beta = 5
                next_sample = opt_acquision(model, theta_bounds, beta=beta, ts=False, prior=last_params)
                # 避免出现重复数据影响GP的拟合
                if np.any(np.abs(next_sample - train_x) <= thresh):
                    next_sample = np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1], theta_bounds.shape[0])
                lmpc.theta_update(next_sample.tolist())
                new_res, xcl, ucl = iters_once(x0, lmpc, Ts, params, res=True)
                xcls.append(xcl)
                ucls.append(ucl)
                train_y = np.vstack((train_y, new_res))
                train_x = np.vstack((train_x, next_sample.reshape(1, -1)))
                if it != 0:
                    source_obj, _, _ = iters_once(x0, lmpc, Ts, params, res=True, SS=lmpc.SS[:-1],
                                                  Qfun=lmpc.Qfun[:-1])  # 原数据
                    bias += (new_res - source_obj) ** 2
                    mu_s[:-1] = [m + bias / 2 for m in mu_s[:-1]]
                    tau_s[-2] += 1 / 2
                    # mu_s = [m + bias / 2 for m in mu_s]
                    # tau_s[-(idx+n_inital_points+1):] = [t + 1 / 2 for t in tau_s[-(idx+n_inital_points+1):]]
                    alpha = np.ones(train_x.shape[0]) * 1e-10
                    for i in range(len(mu_s) - 1):
                        alpha[i * (n_inital_points + n_iters):(i + 1) * (n_inital_points + n_iters)] = mu_s[i] / (
                                1 + tau_s[i])
                    # alpha[-(n_inital_points+idx+1):] = mu_s[-1] / (1 + tau_s[-1])
                else:
                    alpha = np.ones(train_x.shape[0]) * 1e-10

                model = GaussianProcessRegressor(kernel=kernels.RBF(), alpha=alpha, n_restarts_optimizer=5,
                                                 normalize_y=False)
                model.fit(train_x, train_y)
            # next_sample = opt_acquision(model, theta_bounds, beta=5, ts=False)
            # res = iters_once(x0, lmpc, Ts, params)

            # lmpc.theta_update([1, 1, 1, 1])
            # print('theoretical: ', iters_once(x0, lmpc, Ts, params, res=True))

            lmpc.theta_update(last_params.tolist()[0])
            result, xcl, ucl = iters_once(x0, lmpc, Ts, params, res=True)
            if result[0][0] < np.min(train_y[-(n_inital_points + n_iters):], axis=0)[0]:
                lmpc.addTrajectory(xcl, ucl)
            else:
                theta = train_x[-(n_inital_points + n_iters):][
                    np.argmin(train_y[-(n_inital_points + n_iters):], axis=0)]
                # lmpc.theta_update(theta.tolist()[0])
                # iters_once(x0, lmpc, Ts, params)
                lmpc.addTrajectory(xcls[np.argmin(train_y[-(n_inital_points + n_iters):], axis=0)[0]],
                                   ucls[np.argmin(train_y[-(n_inital_points + n_iters):], axis=0)[0]])
                last_params = copy.deepcopy(theta.reshape(1, -1))
            print('optimized theta: ', last_params)


            # mean_module = model.mean_module
            # covar_module = model.covar_module
        end = tim.time()
        print('consumed time: ', end - start)
        times.append(end-start)
        returns.append(lmpc.Qfun_true[it][0])
        # ====================================================================================
        # Compute optimal solution by solving a FTOCP with long horizon
        # ====================================================================================

    tag = 'bayes' if bayes else 'no_bayes'
    np.save('./returns_' + tag + '.npy', returns)
    N = 100  # Set a very long horizon to fake infinite time optimal control problem
    ftocp_opt = FTOCP(N, Ad, Bd, copy.deepcopy(Q), R, params)
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


def iters_once(x0, lmpc, Ts, params, res=False, SS=None, Qfun=None):
    # for it in range(0, totalIterations):
    # Set initial condition at each iteration
    xcl = [x0]
    ucl = []
    time = 0
    # time Loop (Perform the task until close to the origin)
    # while np.dot(xcl[time], xcl[time]) > 10 ** (-5):
    for time in range(100):
        # Read measurement
        xt = xcl[time]

        # Solve FTOCP
        if SS is not None and Qfun is not None:
            lmpc.solve(xt, verbose=False, SS=SS, Qfun=Qfun)
        else:
            lmpc.solve(xt, verbose=False)
        # Read optimal input
        ut = lmpc.uPred[:, 0][0]

        # Apply optimal input to the system
        ucl.append(ut)
        z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
        xcl.append(z[1])
        # xcl.append(np.array(lmpc.ftocp.model(xt, ut)) + np.clip(np.random.randn(4) * 1e-3, -0.01, 0.01))
        # xcl.append(np.array(lmpc.ftocp.model(xt, ut)))
        # xcl.append([a + b * Ts for a, b in zip(xt, inv_pendulum(xt, 0, ut, params))])
        time += 1

    # Add trajectory to update the safe set and value function
    if not res:
        lmpc.addTrajectory(xcl, ucl)

    return lmpc.computeCost(xcl, ucl, np.eye(4) * 10)[0], xcl, ucl  # 这里对Q参数赋值，计算的是真实轨迹下真实回报，而不是


if __name__ == "__main__":
    main()
