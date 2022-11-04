import cvxpy.error
import numpy as np
from numpy import linalg as la
import pdb
import copy
import itertools


class LMPC(object):
    """Learning Model Predictive Controller (LMPC)
	Inputs:
		- ftocp: Finite Time Optimal Control Prolem object used to compute the predicted trajectory
	Methods:
		- addTrajectory: adds a trajectory to the safe set SS and update value function
		- computeCost: computes the cost associated with a feasible trajectory
		- solve: uses ftocp and the stored data to comptute the predicted trajectory"""

    def __init__(self, ftocp, CVX):
        # Initialization
        self.ftocp = ftocp
        self.SS = []
        self.uSS = []
        self.Qfun = []
        self.Q = ftocp.Q
        self.R = ftocp.R
        self.it = 0
        self.CVX = CVX
        self.Q_true = np.eye(2)
        self.Qfun_true = []

        self.last_SS = []
        self.last_Qfun = []
    def theta_update(self, theta):
        # theta0, theta1 = theta
        self.ftocp.Q = np.eye(2) * np.diag(theta)
        # self.ftocp.Q[2:4, 2:4] = 10 * np.eye(2) * np.diag(theta)
        # self.ftocp.R = np.eye(1) * theta
        self.Q = np.eye(2) * np.diag(theta)
        # self.Q[2:4, 2:4] = 10 * np.eye(2) * np.diag(theta)
        # self.R = np.eye(1) * theta
        self.Qfun = []
        for i in range(len(self.SS)):
            self.Qfun.append(self.computeCost(self.SS[i], self.uSS[i]))

    def addTrajectory(self, x, u, x_true=None, u_true=None):
        # Add the feasible trajectory x and the associated input sequence u to the safe set
        self.SS.append(copy.copy(x))
        self.uSS.append(copy.copy(u))

        # Compute and store the cost associated with the feasible trajectory
        cost = self.computeCost(x, u)
        self.Qfun.append(cost)

        if x_true is not None and u_true is not None:
            cost_true = self.computeCost(x_true, u_true, self.Q_true)
            self.Qfun_true.append(cost_true)
        else:
            cost_true = self.computeCost(x, u, self.Q_true)
            self.Qfun_true.append(cost_true)

        # Initialize zVector
        # self.zt = np.array(x[self.ftocp.N], dtype=object)

        # Augment iteration counter and print the cost of the trajectories stored in the safe set
        self.it = self.it + 1
        print("Trajectory added to the Safe Set. Current Iteration: ", self.it)
        print("Performance stored trajectories: \n")
        for i in range(self.it):
            print(self.Qfun_true[i][0])

    def computeCost(self, x, u, Q=None):
        # Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
        if Q is None:
            Q = self.Q
        cost = []
        for i in range(0, len(x)):
            idx = len(x) - 1 - i
            if i == 0:
                cost = [np.dot(np.dot(x[idx], Q), x[idx])]
            else:
                cost.append(np.dot(np.dot(x[idx], Q), x[idx]) + np.dot(np.dot(u[idx], self.R), u[idx]) + cost[-1])

        # Finally flip the cost to have correct order
        return np.flip(np.array(cost, dtype=object)).tolist()

    def solve(self, xt, verbose=False, SS=None, Qfun=None):
        if SS is None:
            SS = self.SS
        if Qfun is None:
            Qfun = self.Qfun
        # Build SS and cost matrices used in the ftocp
        # NOTE: it is possible to use a subset of the stored data to reduce computational complexity while having all guarantees on safety and performance improvement
        SS_vector = np.squeeze(
            np.array(list(itertools.chain.from_iterable(SS)), dtype=object)).T  # From a 3D list to a 2D array
        Qfun_vector = list(itertools.chain.from_iterable(Qfun))
        Qfun_vector = np.array(Qfun_vector, dtype=object)
        Qfun_vector = np.expand_dims(Qfun_vector, 0)
        # Qfun_vector = np.expand_dims(np.array(list(itertools.chain.from_iterable(self.Qfun))),
        # 0)  # From a 2D list to a 1D array

        # Solve the FTOCP.
        # try:
        self.ftocp.solve(xt, verbose, SS_vector, Qfun_vector, self.CVX)
        # except cvxpy.error.SolverError:
        #     print('solver error')

        # Update predicted trajectory
        self.xPred = self.ftocp.xPred
        self.uPred = self.ftocp.uPred
