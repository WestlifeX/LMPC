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
        self.Q_true = np.eye(2) * 10
        self.Qfun_true = []
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
    def addTrajectory(self, x, u):
        # Add the feasible trajectory x and the associated input sequence u to the safe set
        self.SS.append(copy.copy(x))
        self.uSS.append(copy.copy(u))

        # Compute and store the cost associated with the feasible trajectory
        cost = self.computeCost(x, u)
        self.Qfun.append(cost)

        cost_true = self.computeCost(x, u, self.Q_true)
        self.Qfun_true.append(cost_true)

        # Initialize zVector
        self.zt = np.array(x[self.ftocp.N])

        # Augment iteration counter and print the cost of the trajectories stored in the safe set
        self.it = self.it + 1
        print("Trajectory added to the Safe Set. Current Iteration: ", self.it)
        print("Performance stored trajectories: \n", [self.Qfun_true[i][0] for i in range(0, self.it)])

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
        return np.flip(cost).tolist()

    def solve(self, xt, verbose=False):

        # Build SS and cost matrices used in the ftocp
        # NOTE: it is possible to use a subset of the stored data to reduce computational complexity while having all guarantees on safety and performance improvement
        SS_vector = np.squeeze(list(itertools.chain.from_iterable(self.SS))).T  # From a 3D list to a 2D array
        Qfun_vector = np.expand_dims(np.array(list(itertools.chain.from_iterable(self.Qfun))),
                                     0)  # From a 2D list to a 1D array
        # SS_vector = np.squeeze(
        #     np.array(list(itertools.chain.from_iterable(self.SS)), dtype=object)).T  # From a 3D list to a 2D array
        # Qfun_vector = list(itertools.chain.from_iterable(self.Qfun))
        # Qfun_vector = np.array(Qfun_vector, dtype=object)
        # Qfun_vector = np.expand_dims(Qfun_vector, 0)

        # Solve the FTOCP.
        self.ftocp.solve(xt, verbose, SS_vector, Qfun_vector, self.CVX)

        # Update predicted trajectory
        self.xPred = self.ftocp.xPred
        self.uPred = self.ftocp.uPred
