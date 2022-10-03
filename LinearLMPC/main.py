import numpy as np
from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import matplotlib
from tqdm import tqdm
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import pickle
from objective_functions_lqr import get_params, get_linearized_model, inv_pendulum
from bayes_opt_mine import get_model, step
import torch
from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from acq_func import opt_acquision
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
def main():
    # Define system dynamics and cost
    np.random.seed(1)
    A = np.array([[1, 1], [0, 1]])
    B = np.array([[0], [1]])
    Q = np.diag([1.0, 1.0])  # np.eye(2)
    R = np.array([[1]])  # np.array([[1]])

    print("Computing first feasible trajectory")

    # Initial Condition
    x0 = [-15.0, 0.0]

    # Initialize FTOCP object
    N_feas = 5
    ftocp_for_mpc = FTOCP(N_feas, A, B, 0.01 * Q, R)

    # ====================================================================================
    # Run simulation to compute feasible solution
    # ====================================================================================
    xcl_feasible = [x0]
    ucl_feasible = []
    xt = x0
    time = 0

    # time Loop (Perform the task until close to the origin)
    while np.dot(xt, xt) > 10 ** (-15):
        xt = xcl_feasible[time]  # Read measurements

        ftocp_for_mpc.solve(xt, verbose=False)  # Solve FTOCP

        # Read input and apply it to the system
        ut = ftocp_for_mpc.uPred[:, 0][0]
        ucl_feasible.append(ut)
        xcl_feasible.append(ftocp_for_mpc.model(xcl_feasible[time], ut))
        time += 1

    print(np.round(np.array(xcl_feasible).T, decimals=2))
    print(np.round(np.array(ucl_feasible).T, decimals=2))
    # ====================================================================================

    # ====================================================================================
    # Run LMPC
    # ====================================================================================
    Q = np.diag([1.0, 1.0])
    # Initialize LMPC object
    N_LMPC = 3  # horizon length
    ftocp = FTOCP(N_LMPC, A, B, Q, R)  # ftocp solved by LMPC
    lmpc = LMPC(ftocp, CVX=True)  # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible)  # Add feasible trajectory to the safe set

    totalIterations = 20  # Number of iterations to perform
    n_params = 2
    theta_bounds = np.array([[0.5, 2]] * n_params)
    # run simulation
    # iteration loop
    print("Starting LMPC")
    returns = []
    n_inital_points = 10
    n_iters = 10
    thresh = 1e-7
    prior = np.array([1] * n_params).reshape(1, -1)
    last_params = copy.deepcopy(prior)
    for it in range(0, totalIterations):

        # lmpc.theta_update(theta)
        # bayes opt
        print("Initializing")
        # if it == 0:
        train_x = np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1],
                                    size=(n_inital_points, theta_bounds.shape[0]))
        train_y = []
        for i in tqdm(range(n_inital_points)):
            lmpc.theta_update(train_x[i].tolist())
            train_obj = iters_once(x0, lmpc, res=True)  # 这里取个负号，因为我们的目标是取最小，而这个BO是找最大点
            train_y.append(train_obj)
        train_y = np.squeeze(train_y, axis=1)
        model = GaussianProcessRegressor(kernel=kernels.Matern(), n_restarts_optimizer=5, normalize_y=False)
        model.fit(train_x, train_y)
        # model.fit(train_x, train_y)
        # model, mll = get_model(train_x, train_y)
        print('bayes opt for {} iteration'.format(it + 1))
        for _ in tqdm(range(n_iters)):
            next_sample = opt_acquision(model, theta_bounds, beta=25, ts=False)
            # prior = next_sample.reshape(1, -1)  # 仅用在时变里，但是没什么用
            # 避免出现重复数据影响GP的拟合
            if np.any(np.abs(next_sample - train_x) <= thresh):
                next_sample = np.random.uniform(theta_bounds[:, 0], theta_bounds[:, 1], theta_bounds.shape[0])
            lmpc.theta_update(next_sample.tolist())
            new_res = iters_once(x0, lmpc, res=True)
            train_y = np.vstack((train_y, new_res))
            train_x = np.vstack((train_x, next_sample))

            # model.fit(train_x, train_y)
            model.fit(train_x, train_y)

        # next_sample = opt_acquision(model, theta_bounds, beta=5, ts=False, prior=prior)
        # prior = next_sample.reshape(1, -1)  # 仅用在时变里
        lmpc.theta_update(last_params.tolist()[0])
        result = iters_once(x0, lmpc, res=True)
        # # res = iters_once(x0, lmpc, Ts, params)
        if result[0][0] < np.min(train_y[-(n_inital_points + n_iters):, ], axis=0)[0]:
            iters_once(x0, lmpc)
            print('optimized theta: ', last_params)
            # prior = last_params
        else:
            theta = train_x[-(n_inital_points + n_iters):][np.argmin(train_y[-(n_inital_points + n_iters):], axis=0)]
            lmpc.theta_update(theta.tolist()[0])
            iters_once(x0, lmpc)
        # prior = theta.reshape(1, -1)
        #     last_params = copy.deepcopy(theta.reshape(1, -1))
            print('optimized theta: ', theta)
        # mean_module = model.mean_module
        # covar_module = model.covar_module
        returns.append(lmpc.Qfun_true[it][0])

    # =====================================================================================

    # ====================================================================================
    # Compute optimal solution by solving a FTOCP with long horizon
    # ====================================================================================
    N = 100  # Set a very long horizon to fake infinite time optimal control problem
    ftocp_opt = FTOCP(N, A, B, Q, R)
    ftocp_opt.solve(x0)
    xOpt = ftocp_opt.xPred
    uOpt = ftocp_opt.uPred
    costOpt = lmpc.computeCost(xOpt.T.tolist(), uOpt.T.tolist(), np.eye(2))
    print("Optimal cost is: ", costOpt[0])
    # Store optimal solution in the lmpc object
    lmpc.optCost = costOpt[0]
    lmpc.xOpt = xOpt

    # Save the lmpc object
    filename = 'lmpc_object.pkl'
    filehandler = open(filename, 'wb')
    pickle.dump(lmpc, filehandler)

def iters_once(x0, lmpc, res=False):
    # Set initial condition at each iteration
    xcl = [x0]
    ucl = []
    time = 0
    # time Loop (Perform the task until close to the origin)
    while np.dot(xcl[time], xcl[time]) > 10 ** (-10):
        # Read measurement
        xt = xcl[time]

        # Solve FTOCP
        lmpc.solve(xt, verbose=False)
        # Read optimal input
        ut = lmpc.uPred[:, 0][0]

        # Apply optimal input to the system
        ucl.append(ut)
        xcl.append(lmpc.ftocp.model(xcl[time], ut))
        time += 1

    if not res:
        # Add trajectory to update the safe set and value function
        lmpc.addTrajectory(xcl, ucl)

    return lmpc.computeCost(xcl, ucl, np.eye(2))[0]  # 这里对Q参数赋值，计算的是真实轨迹下真实回报，而不是

if __name__ == "__main__":
    main()
