import numpy as np
import pdb
import scipy
from cvxpy import *
from scipy.integrate import odeint
from objective_functions_lqr import inv_pendulum, get_params
import casadi as ca
import copy
from mrpi.polyhedron import polyhedron
from mrpi.mRPI_set import compute_mRPI
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
        # Initialize Predicted Trajectory
        self.xPred = []
        self.uPred = []

        self.bias = 0

        self.len_conx = 0
        self.len_conu = 0
        W_A = np.array([[1., 0], [-1., 0], [0, 1.], [0, -1.]])
        W_b = np.array([0.2, 0.2, 0.25, 0.25]).reshape(-1, 1)
        W = polyhedron(W_A, W_b)
        eps = 1e-5
        # 加了N这个参数，所以求的已经不是mrpi而是前五步的mrpi，已经够用了
        _, self.F_list = compute_mRPI(eps, W, self.A, self.B, self.K, self.N)

        X_A = np.array([[1., 0], [-1., 0], [0, 1.], [0, -1.]])
        X_b = np.array([10., 10., 10., 10.]).reshape(-1, 1)
        X = polyhedron(X_A, X_b)
        U_A = np.array([[1.], [-1.]])
        U_b = np.array([1., 1.]).reshape(-1, 1)
        U = polyhedron(U_A, U_b)
        self.constr_x = []
        self.constr_u = []
        for i in range(self.N+1):
            self.constr_x.append(X.minkowskiDiff(self.F_list[i]))
            if i > 0:
                self.constr_x[i].minVrep()
            self.constr_x[i].compute_Hrep()
            self.len_conx += self.constr_x[i].A.shape[0]
        for i in range(self.N):
            self.constr_u.append(U.minkowskiDiff(self.F_list[i].affineMap(self.K)))
            self.constr_u[i].compute_Hrep()
            self.len_conu += self.constr_u[i].A.shape[0]

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
            constr += [self.constr_x[i].A @ x[:, i] <= self.constr_x[i].b.reshape(-1, )]
            constr += [self.constr_u[i].A @ u[:, i] <= self.constr_u[i].b.reshape(-1, )]
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
        constr += [self.constr_x[self.N].A @ x[:, self.N] <= self.constr_x[self.N].b.reshape(-1, )]
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
            problem.solve(verbose=verbose, solver=ECOS)  # I find that ECOS is better please use it when solving QPs 3q
        else:
            problem.solve(verbose=verbose, solver=ECOS)
        self.xPred = x.value
        self.uPred = u.value
        # Store the open-loop predicted trajectory

    def model(self, x, u):
        #     # Compute state evolution
        return (np.dot(self.A, x) + np.squeeze(np.dot(self.B, u))).tolist()
