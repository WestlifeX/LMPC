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
        # self.R_delta = ftocp.R_delta
        self.it = 0
        self.CVX = CVX
        self.Q_true = np.diag([2., 2.])
        self.R_true = np.eye(1) * 1
        # self.R_delta_true = np.eye(1) * 1
        self.Qfun_true = []

        self.last_SS = []
        self.last_Qfun = []

        self.u1 = 1.
        self.u2 = 1.
    def theta_update(self, theta):
        # theta0, theta1 = theta
        self.ftocp.Q = self.Q_true * np.diag(theta[:2])
        self.Q = self.Q_true * np.diag(theta[:2])
        self.ftocp.R = self.R_true * theta[2]
        self.R = self.R_true * theta[2]
        # self.ftocp.R_delta = self.R_delta_true * theta[3]
        # self.R_delta = self.R_delta_true * theta[3]
        # self.ftocp.N = int(round(theta[4]))
        # self.Q[0, 1] = theta[3]
        # self.Q[1, 0] = theta[3]
        # self.ftocp.Q[0, 1] = theta[3]
        # self.ftocp.Q[1, 0] = theta[3]
        # self.ftocp.A[0, 0] = theta[0]
        # self.ftocp.A[0, 1] = theta[1]
        # self.ftocp.A[1, 0] = theta[2]
        # self.ftocp.A[1, 1] = theta[3]
        # self.ftocp.B[0, 0] = theta[4]
        # self.ftocp.B[1, 0] = theta[5]
        # self.ftocp.compute_mrpi()
        # self.ftocp.N = int(round(theta[6]))
        # self.u1 = theta[4]
        # self.u2 = theta[5]

        self.Qfun = []
        for i in range(len(self.SS)):
            self.Qfun.append(self.computeCost(self.SS[i], self.uSS[i]))

    def addTrajectory(self, x, u):
        # Add the feasible trajectory x and the associated input sequence u to the safe set
        self.SS.append(copy.copy(x))
        self.uSS.append(copy.copy(u))
        # if len(self.SS) > 10:
        #     self.SS = [self.SS[0]] + self.SS[-10:]
        #     self.uSS = [self.uSS[0]] + self.uSS[-10:]
        # Compute and store the cost associated with the feasible trajectory
        cost = self.computeCost(x, u)
        self.Qfun.append(cost)

        cost_true = self.computeCost(x, u, self.Q_true, self.R_true)
        self.Qfun_true.append(cost_true)

        # Initialize zVector
        # self.zt = np.array(x[self.ftocp.N], dtype=object)

        # Augment iteration counter and print the cost of the trajectories stored in the safe set
        self.it = self.it + 1
        print("Trajectory added to the Safe Set. Current Iteration: ", self.it)
        print("Performance stored trajectories: \n")
        for i in range(self.it):
            print(self.Qfun_true[i][0])

    def computeCost(self, x, u, Q=None, R=None, R_delta=None):
        # Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
        if Q is None and R is None:
            Q = self.Q
            R = self.R
            # R_delta = self.R_delta
        cost = []
        for i in range(0, len(x)):
            idx = len(x) - 1 - i
            if i == 0:
                cost = [np.dot(np.dot(x[idx], Q), x[idx])]
            else:
                if i == len(x) - 1:
                    cost.append(np.dot(np.dot(x[idx], Q), x[idx]) +
                                np.dot(np.dot(u[idx], R), u[idx])[0][0] +
                                # np.dot(np.dot(u[idx], R_delta), u[idx])[0][0] +
                                cost[-1])
                else:
                    cost.append(np.dot(np.dot(x[idx], Q), x[idx]) +
                                np.dot(np.dot(u[idx], R), u[idx])[0][0] +
                                # np.dot(np.dot(u[idx] - u[idx - 1], R_delta), u[idx] - u[idx - 1])[0][0] +
                                cost[-1])

        # Finally flip the cost to have correct order
        return np.flip(cost).tolist()

    def solve(self, xt, time=0, verbose=False, SS=None, Qfun=None):
        if SS is None:
            SS = self.SS
        if Qfun is None:
            Qfun = self.Qfun
        # Build SS and cost matrices used in the ftocp
        # NOTE: it is possible to use a subset of the stored data to reduce computational complexity while having all guarantees on safety and performance improvement
        SS_vector = np.squeeze(
            np.array(list(itertools.chain.from_iterable(SS)))).T  # From a 3D list to a 2D array
        Qfun_vector = list(itertools.chain.from_iterable(Qfun))
        Qfun_vector = np.array(Qfun_vector)
        Qfun_vector = np.expand_dims(Qfun_vector, 0)
        # Qfun_vector = np.expand_dims(np.array(list(itertools.chain.from_iterable(self.Qfun))),
        # 0)  # From a 2D list to a 1D array

        # Solve the FTOCP.
        # try:
        res = self.ftocp.solve(xt, time, verbose, SS_vector, Qfun_vector, self.CVX)
        # except cvxpy.error.SolverError:
        #     print('solver error')
        if res == 0:
            self.xPred = self.xPred[1:, ]
            self.uPred = self.uPred[1:, ]
        else:
        # Update predicted trajectory
            self.xPred = self.ftocp.xPred
            self.uPred = self.ftocp.uPred
