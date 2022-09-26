import numpy as np
import torch

from FTOCP import FTOCP
from LMPC import LMPC
import pdb
import matplotlib
from scipy.integrate import odeint

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import copy
import pickle
from objective_functions_lqr import get_params, get_linearized_model, inv_pendulum
from bayes_opt import get_model, step

from botorch.optim import optimize_acqf
from botorch.acquisition import UpperConfidenceBound
from gpytorch.means import ConstantMean
from gpytorch.kernels import MaternKernel
import gaussian_process as gp
import kernel as kn
from acq_func import opt_acquision
from sklearn.gaussian_process import GaussianProcessRegressor
def main():
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
    x0 = [1, 0, 0.1, -0.01]

    # Initialize FTOCP object
    N_feas = 10
    # 产生初始可行解的时候应该Q、R随便
    # 求解MPC应该也是用线性模型，因为MPC是为了求解u，而求u应该用不准确的模型，否则就没有误差了，但是得到u之后求下一步x用非线性的
    # 修改了初值的位置，因为x离原点太远了，把cart调回原点需要太多步了（x的初值应该是可以改的，因为线性化只针对phi）
    # 增大了采样时间0.02 --> 0.1，同上，采样时间越小需要的步数就越多
    # 增大了Q（原来是0.01*Q），Q大一点应该可以快点收敛吧
    ftocp_for_mpc = FTOCP(N_feas, Ad, Bd, Q, R)
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

        # Read input and apply it to the system
        ut = ftocp_for_mpc.uPred[:, 0][0]
        ucl_feasible.append(ut)
        z = odeint(inv_pendulum, xt, [Ts*time, Ts*(time+1)], args=(ut, params))  # 用非线性连续方程求下一步
        xcl_feasible.append(z[1])
        # xcl_feasible.append(ftocp_for_mpc.model(xcl_feasible[time], ut))
        time += 1

    print(np.round(np.array(xcl_feasible).T, decimals=2))
    print(np.round(np.array(ucl_feasible).T, decimals=2))
    # ====================================================================================

    # ====================================================================================
    # Run LMPC
    # ====================================================================================

    # Initialize LMPC object
    # 这个horizon length设置成3的时候会出现infeasible的情况
    # 理论上不应该无解，已经生成可行解了，不可能无解，可能是求解器的问题
    N_LMPC = 5  # horizon length
    ftocp = FTOCP(N_LMPC, Ad, Bd, copy.deepcopy(Q), R)  # ftocp solved by LMPC，这里的Q和R在后面应该要一直变，初始值可以先用Q，R
    lmpc = LMPC(ftocp, CVX=True)  # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible)  # Add feasible trajectory to the safe set
    bayes = True
    totalIterations = 20  # Number of iterations to perform
    theta = [1, 1, 1, 1]  # 填theta初始值，等后续确定了theta范围再填
    theta_bounds = [[0.2, 0.2, 0.2, 0.2], [5, 5, 5, 5]]
    # run simulation
    # iteration loop
    print("Starting LMPC")
    returns = []
    kernel = kn.Matern32Kernel(1.0, 1.0)
    for it in range(0, totalIterations):
        if not bayes:
            # pass
            iters_once(x0, lmpc, Ts, params)
        else:
            # lmpc.theta_update(theta)
            # bayes opt
            print('bayes opt for {} iteration'.format(it))
            n_inital_points = 10
            train_x = torch.FloatTensor(n_inital_points, 4).uniform_(theta_bounds[0][0], theta_bounds[1][0])
            train_y = []
            for i in range(n_inital_points):
                lmpc.theta_update(train_x[i].tolist())
                train_obj = - iters_once(x0, lmpc, Ts, params, res=True)  # 这里取个负号，因为我们的目标是取最小，而这个BO是找最大点
                train_y.append(train_obj)
            train_y = torch.tensor(np.array(train_y), dtype=torch.float32).squeeze(1)
            # train_y = (train_y - torch.mean(train_y)) / torch.std(train_y)  # 做个标准化（标准化好像对结果有反作用）

            # model = gp.GaussianProcess(kernel, 0.001)
            model = GaussianProcessRegressor()
            model.fit(train_x.detach().numpy(), train_y.detach().numpy())
            # model, mll = get_model(train_x, train_y)
            for i in range(10):
                new_point_analytic = opt_acquision(train_x, model, theta_bounds, beta=5)
                point = new_point_analytic.tolist()
                lmpc.theta_update(point)
                new_res = - iters_once(x0, lmpc, Ts, params, res=True)
                train_y = torch.cat([torch.tensor(new_res, dtype=torch.float32), train_y])
                train_x = torch.cat([torch.tensor(point).unsqueeze(0), train_x])

                # model.fit(train_x, train_y)
                model.fit(train_x.detach().numpy(), train_y.detach().numpy())
            new_point_analytic = opt_acquision(train_x, model, theta_bounds, beta=5)
            theta = new_point_analytic.tolist()
            lmpc.theta_update(theta)
            iters_once(x0, lmpc, Ts, params)
            # mean_module = model.mean_module
            # covar_module = model.covar_module
        returns.append(lmpc.Qfun_true[it][0])
        # ====================================================================================
        # Compute optimal solution by solving a FTOCP with long horizon
        # ====================================================================================
    tag = 'bayes' if bayes else 'no_bayes'
    np.save('./returns_' + tag + '.npy', returns)
    N = 100  # Set a very long horizon to fake infinite time optimal control problem
    ftocp_opt = FTOCP(N, Ad, Bd, copy.deepcopy(Q), R)
    ftocp_opt.solve(x0)
    xOpt = ftocp_opt.xPred
    uOpt = ftocp_opt.uPred
    lmpc.theta_update([1, 1, 1, 1])
    costOpt = lmpc.computeCost(xOpt.T.tolist(), uOpt.T.tolist())
    print("Optimal cost is: ", costOpt[0])
    # Store optimal solution in the lmpc object
    lmpc.optCost = costOpt[0]
    lmpc.xOpt = xOpt

    # Save the lmpc object
    filename = 'lmpc_object.pkl'
    filehandler = open(filename, 'wb')
    pickle.dump(lmpc, filehandler)

def iters_once(x0, lmpc, Ts, params, res=False):
    # for it in range(0, totalIterations):
        # Set initial condition at each iteration
    xcl = [x0]
    ucl = []
    time = 0
    # time Loop (Perform the task until close to the origin)
    while np.dot(xcl[time], xcl[time]) > 10 ** (-3):
        # Read measurement
        xt = xcl[time]

        # Solve FTOCP
        lmpc.solve(xt, verbose=False)
        # Read optimal input
        ut = lmpc.uPred[:, 0][0]

        # Apply optimal input to the system
        ucl.append(ut)
        z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
        xcl.append(z[1])
        time += 1

    # Add trajectory to update the safe set and value function
    if not res:
        lmpc.addTrajectory(xcl, ucl)

    return lmpc.computeCost(xcl, ucl, np.eye(4)*10)[0]  # 这里对Q参数赋值，计算的是真实轨迹下真实回报，而不是


if __name__ == "__main__":
    main()
