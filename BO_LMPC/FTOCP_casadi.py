import numpy as np
import pdb
import scipy
from cvxpy import *
from scipy.integrate import odeint
from objective_functions_lqr import inv_pendulum, get_params
from casadi import *
from mrpi.polyhedron import polyhedron
from mrpi.mRPI_set import compute_mRPI
class FTOCP(object):
    """ Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""

    def __init__(self, N, A, B, Q, R, K, args):
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
        self.args = args
        self.len_conx = 0
        self.len_conu = 0
        W_A = np.array([[1., 0], [-1., 0], [0, 1.], [0, -1.]])
        W_b = np.array([0.2, 0.2, 0.2, 0.2]).reshape(-1, 1)
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

        a = 1
    def solve(self, x0, verbose=False, SS=None, Qfun=None, CVX=None):
        """This method solves an FTOCP given:
			- x0: initial condition
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
		"""
        # Initialize Variables
        # x = ca.SX.sym('x', self.n, self.N + 1, boolean=True)
        x = MX.sym('x', self.n*(self.N+1))
        u = MX.sym('u', self.d*self.N)
        # State Constraints
        constraints = []
        for i in range(self.N+1):
            constraints = vertcat(constraints,
                                  mtimes(self.constr_x[i].A, x[self.n * i:self.n * (i + 1)])-self.constr_x[i].b)
        for i in range(self.N):
            constraints = vertcat(constraints, mtimes(self.constr_u[i].A,
                                                      u[self.d * i:self.d * (i + 1)])-self.constr_u[i].b)
        constraints = vertcat(constraints, x[:self.n]-np.array(x0))
        cost = 0
        for i in range(self.N):
            constraints = vertcat(constraints,
                                  x[self.n * (i + 1):self.n * (i + 2)] -
                                  mtimes(self.A, x[self.n * i:self.n * (i + 1)]) -
                                  mtimes(self.B, u[self.d * i:self.d * (i + 1)]))
            cost = cost + mtimes(mtimes(x[self.n * i:self.n * (i + 1)].T, self.Q),
                                 x[self.n * i:self.n * (i + 1)]) + \
                    mtimes(mtimes(u[self.d * i:self.d * (i + 1)].T, self.R), u[self.d * i:self.d * (i + 1)])

        # SS的shape应该是 n × SS中点的个数
        # 如果SS是None，末项就是简单的二次型，否则是SS中各个点value的加权和
        if SS is not None:
            lambVar = MX.sym('lam', SS.shape[1])
            # Terminal Constraint if SS not empty --> enforce the terminal constraint
            for i in range(self.n):
                constraints = vertcat(constraints, x[self.n * self.N + i] - dot(SS[i], lambVar))
            constraints = vertcat(constraints, dot(np.ones(SS.shape[1]), lambVar) - 1)
            # constraints = vertcat(constraints, lambVar - 0)
            # Terminal cost if SS not empty
            cost = cost + dot(Qfun[0], lambVar)  # Its terminal cost is given by interpolation using \lambda
            options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                       "ipopt.mu_strategy": "adaptive",
                       "ipopt.mu_init": 0.1, "ipopt.mu_min": 1e-11,
                       "ipopt.barrier_tol_factor": 10}
            nlp = {'x': vertcat(x, u, lambVar), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            lbg = [-np.inf]*(self.len_conx + self.len_conu) + [0] * (self.n * (self.N + 1)) + [0] * self.n + [0] * 1
            ubg = [0]*(self.len_conx + self.len_conu) + [0] * (self.n * (self.N + 1)) + [0] * self.n + [0] * 1
            lbx = [-10.] * (self.n * (self.N + 1)) + [-1] * (self.d * self.N) + [0.] * SS.shape[1]
            ubx = [10.] * (self.n * (self.N + 1)) + [1] * (self.d * self.N) + [1.] * SS.shape[1]
            sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            g = np.array(sol['g'][-self.n * (self.N + 1)-self.n-1:])
            if np.sum(g) > 0.01:
                a = 1
        else:
            cost = cost + mtimes(mtimes(x[self.n * self.N:self.n * (self.N + 1)].T, self.Q),
                                 x[self.n * self.N:self.n * (self.N + 1)])  # If SS is not given terminal cost is quadratic
            options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                       "ipopt.mu_strategy": "adaptive",
                       "ipopt.mu_init": 0.1, "ipopt.mu_min": 1e-11,
                       "ipopt.barrier_tol_factor": 10}
            nlp = {'x': vertcat(x, u), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            lbg = [-np.inf]*(self.len_conx + self.len_conu) + [0] * (self.n * (self.N + 1))
            ubg = [0]*(self.len_conx + self.len_conu) + [0] * (self.n * (self.N + 1))
            lbx = [-10.] * (self.n * (self.N + 1)) + [-1] * (self.d * self.N)
            ubx = [10.] * (self.n * (self.N + 1)) + [1] * (self.d * self.N)
            sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            g = np.array(sol['g'][-self.n * (self.N + 1):])
            if np.sum(g) > 0.01:
                a = 1
        res = np.array(sol['x'])

        xSol = res[0:(self.N + 1) * self.n].reshape((self.N + 1, self.n)).T
        uSol = res[(self.N + 1) * self.n:((self.N + 1) * self.n + self.d * self.N)].reshape(
            (self.N, self.d)).T
        self.xPred = xSol
        self.uPred = uSol

        # Store the open-loop predicted trajectory

    def model(self, x, u):
        #     # Compute state evolution
        return (np.dot(self.A, x) + np.squeeze(np.dot(self.B, u))).tolist()
