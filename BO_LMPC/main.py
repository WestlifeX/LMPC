import numpy as np
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


def main():
    Ts = 0.02
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
    x0 = [4, 0, 0.1, -0.01]

    # Initialize FTOCP object
    N_feas = 10
    # 产生初始可行解的时候应该Q、R随便
    # 求解MPC应该也是用线性模型，因为MPC是为了求解u，而求u应该用不准确的模型，否则就没有误差了，但是得到u之后求下一步x用非线性的
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
    N_LMPC = 3  # horizon length
    ftocp = FTOCP(N_LMPC, A, B, Q, R)  # ftocp solved by LMPC，这里的Q和R在后面应该要一直变，初始值可以先用Q，R
    lmpc = LMPC(ftocp, CVX=True)  # Initialize the LMPC (decide if you wanna use the CVX hull)
    lmpc.addTrajectory(xcl_feasible, ucl_feasible)  # Add feasible trajectory to the safe set

    totalIterations = 20  # Number of iterations to perform
    theta = []  # 填theta初始值，等后续确定了theta范围再填
    # run simulation
    # iteration loop
    print("Starting LMPC")
    for it in range(0, totalIterations):
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
            z = odeint(inv_pendulum, xt, [Ts * time, Ts * (time + 1)], args=(ut, params))  # 用非线性连续方程求下一步
            xcl.append(z[1])
            time += 1

        # Add trajectory to update the safe set and value function
        lmpc.addTrajectory(xcl, ucl)

    # =====================================================================================

    # ====================================================================================
    # Compute optimal solution by solving a FTOCP with long horizon
    # ====================================================================================
    N = 100  # Set a very long horizon to fake infinite time optimal control problem
    ftocp_opt = FTOCP(N, A, B, Q, R)
    ftocp_opt.solve(xcl[0])
    xOpt = ftocp_opt.xPred
    uOpt = ftocp_opt.uPred
    costOpt = lmpc.computeCost(xOpt.T.tolist(), uOpt.T.tolist())
    print("Optimal cost is: ", costOpt[0])
    # Store optimal solution in the lmpc object
    lmpc.optCost = costOpt[0]
    lmpc.xOpt = xOpt

    # Save the lmpc object
    filename = 'lmpc_object.pkl'
    filehandler = open(filename, 'wb')
    pickle.dump(lmpc, filehandler)


if __name__ == "__main__":
    main()
