import numpy as np
import pdb
import scipy
from cvxpy import *
from scipy.integrate import odeint
from objective_functions_lqr import inv_pendulum, get_params
import casadi as ca
import pypolycontain as pp
import copy
from pytope import Polytope
from eps_MRPI import eps_MRPI
class FTOCP(object):
    """ Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""

    def __init__(self, N, A, B, Q, R, K, params):
        # Define variables
        self.N = N  # Horizon Length

        # System Dynamics (x_{k+1} = A x_k + Bu_k)
        self.A = A
        self.B = B
        self.n = A.shape[1]
        self.d = B.shape[1]

        # Cost (h(x,u) = x^TQx +u^TRu)
        self.Q = Q
        self.R = R
        self.K = K
        self.F = np.vstack((np.eye(4), np.zeros((1, 4))))
        self.G = np.zeros((5, 1))
        self.G[4] = 1
        self.f = np.array([3, 3, 1, 2, 5])
        self.phi = self.F + np.dot(self.G, self.K)

        W_A = np.array([[0, 0, 0, -1], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, -1, 0]])
        W_b = np.array([0.1, 0.1, 0.1, 0.1])
        # W = Polytope(W_A, W_b)
        Ak = self.A + np.dot(self.B, self.K)
        # F_alpha_s, result = eps_MRPI(Ak, W, 0.01)
        W = pp.H_polytope(W_A, W_b)

        rho = 0.8
        rho_W = pp.H_polytope(W_A, rho * W_b)
        mRPI = copy.deepcopy(W)
        for i in range(1, 50):
            Ai = Ak ** i
            AW = pp.AH_polytope(np.zeros((4, 1)), Ai, W)
            if pp.check_subset(AW, rho_W):
                break
            mRPI = pp.minkowski_sum(mRPI, AW)
        self.mRPI = 1 / (1 - rho) * mRPI
        # Initialize Predicted Trajectory
        self.xPred = []
        self.uPred = []

        self.bias = 0
    def solve(self, x0, verbose=False, SS=None, Qfun=None, CVX=None):
        """This method solves an FTOCP given:
			- x0: initial condition
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
		"""
        # Initialize Variables
        # x = ca.SX.sym('x', self.n, self.N + 1, boolean=True)
        x = Variable((self.n, self.N + 1))
        u = Variable((self.d, self.N))

        # State Constraints
        constr = [x[:, 0] == x0[:]]  # initializing condition
        for i in range(self.N):
            constr += [self.F @ x[:, i] + self.G @ u[:, i] <= self.f -
                       np.abs(np.dot(self.phi, np.array([0.1, 0, 0.1, 0]).reshape(-1, )))]
            constr += [self.F @ x[:, i] + self.G @ u[:, i] >= -self.f +
                       np.abs(np.dot(self.phi, np.array([0.1, 0, 0.1, 0]).reshape(-1, )))]
            # constr += [u[:, i] >= -5.0 + np.abs(self.bias),
            #            u[:, i] <= 5.0 - np.abs(self.bias),
            #            x[0, i] >= -5.0,
            #            x[0, i] <= 5.0,
            #            x[1, i] >= -5.0,
            #            x[1, i] <= 5.0,
            #            x[2, i] >= -0.5+1e-5,
            #            x[2, i] <= 0.5-1e-5,
            #            x[3, i] >= -1+1e-5,
            #            x[3, i] <= 1-1e-5, ]
            constr += [x[:, i + 1] == self.A @ x[:, i] + self.B @ u[:, i], ]
            # constr += [x[:, i + 1] == odeint(inv_pendulum, x[:, i], [0, 0.1], args=(u[:, i], params))[1]]
        # Cost Function
        cost = 0
        for i in range(self.N):
            # Running cost h(x,u) = x^TQx + u^TRu
            cost += quad_form(x[:, i], self.Q) + quad_form(u[:, i], self.R)

        # cost += quad_form(x[:, self.N], self.Q)
        # cost += norm(self.Q**0.5*x[:,i])**2 + norm(self.R**0.5*u[:,i])**2

        # If SS is given initialize lambdaVar multipliers used to enforce terminal constraint
        # SS的shape应该是 n × SS中点的个数
        # 如果SS是None，末项就是简单的二次型，否则是SS中各个点value的加权和
        if SS is not None:
            if CVX:
                lambVar = Variable((SS.shape[1], 1), boolean=False)  # Initialize vector of variables
            else:
                lambVar = Variable((SS.shape[1], 1), boolean=True)  # Initialize vector of variables
            # cost += quad_form(x[:, self.N], self.Q)
            # Terminal Constraint if SS not empty --> enforce the terminal constraint
            # 改变Q的话，xN就会不一样，xN不一样约束这里就会不一样
            constr += [SS @ lambVar[:, 0] == x[:, self.N],  # Terminal state \in ConvHull(SS)
                       np.ones((1, SS.shape[1])) @ lambVar[:, 0] == 1,  # Multiplies \lambda sum to 1
                       lambVar >= 0]  # Multiplier are positive definite

            # Terminal cost if SS not empty
            cost += Qfun[0, :] @ lambVar[:, 0]  # Its terminal cost is given by interpolation using \lambda

            self.lamb = lambVar.value
        else:
            cost += quad_form(x[:, self.N], self.Q)  # If SS is not given terminal cost is quadratic

        # Solve the Finite Time Optimal Control Problem
        problem = Problem(Minimize(cost), constr)
        if CVX:
            # problem.solve(verbose=verbose)
            problem.solve(verbose=verbose, solver=MOSEK)  # I find that ECOS is better please use it when solving QPs 3q
        else:
            problem.solve(verbose=verbose)
        self.xPred = x.value
        self.uPred = u.value
        # Store the open-loop predicted trajectory

    def model(self, x, u):
        #     # Compute state evolution
        return (np.dot(self.A, x) + np.squeeze(np.dot(self.B, u))).tolist()
