import numpy as np
import pdb
import scipy
from cvxpy import *
from scipy.integrate import odeint
from objective_functions_lqr import inv_pendulum, get_params
from casadi import *
from mrpi.polyhedron import polyhedron, plot_polygon_list
from mrpi.mRPI_set import compute_mRPI
from scipy.spatial import ConvexHull
class FTOCP(object):
    """ Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""

    def __init__(self, N, A, B, Q, R, R_delta, K, args):
        # Define variables
        self.N = N  # Horizon Length
        # System Dynamics (x_{k+1} = A x_k + Bu_k)
        # self.A_true = np.array([[0.995, 0.095], [-0.095, 0.900]])
        # self.B_true = np.array([[0.048], [0.95]])
        self.A = A
        self.B = B
        self.n = A.shape[1]
        self.d = B.shape[1]

        # Cost (h(x,u) = x^TQx +u^TRu)
        self.Q = Q
        self.R = R
        self.R_delta = R_delta
        self.K = K
        # Initialize Predicted Trajectory
        self.xPred = []
        self.uPred = []
        self.bias = 0
        self.args = args
        self.len_conx = 0
        self.len_conu = 0
        W_A = np.array([[1., 0], [-1., 0], [0, 1.], [0, -1.]])
        W_b = np.array([0.03, 0.03, 0.03, 0.03]).reshape(-1, 1)
        self.W = polyhedron(W_A, W_b)
        X_A = np.array([[1., 0], [-1., 0], [0, 1.], [0, -1.]])
        X_b = np.array([10., 10., 10., 10.]).reshape(-1, 1)
        self.X = polyhedron(X_A, X_b)
        U_A = np.array([[1.], [-1.]])
        U_b = np.array([1., 1.]).reshape(-1, 1)
        self.U = polyhedron(U_A, U_b)
        self.compute_mrpi()

    def compute_mrpi(self):
        # 计算60个F_list，但在MPC里50步就终止
        self.mrpi, self.F_list = compute_mRPI(1e-10, self.W, self.A, self.B, self.K, 60)
        self.constr_x = []
        self.constr_u = []
        self.Kxs = []
        self.s = len(self.F_list)
        # 之所以s+1是要把最后一个留给mrpi
        for i in range(self.s+1):
            if i < self.s:
                self.F_list[i].minVrep()
                self.constr_x.append(self.X.minkowskiDiff(self.F_list[i]))
            else:
                self.mrpi.minVrep()
                self.constr_x.append(self.X.minkowskiDiff(self.mrpi))
            # if i > 0:
            self.constr_x[i].minVrep()
            try:
                self.constr_x[i].vertices = np.round(self.constr_x[i].vertices, 3)
                self.constr_x[i].compute_Hrep()
            except RuntimeError:
                self.constr_x[i].vertices = np.round(self.constr_x[i].vertices)
                self.constr_x[i].compute_Hrep()
                a = 1

        for i in range(self.s+1):
            if i < self.s:
                Kx = self.F_list[i].affineMap(self.K)
                # 防止错误的顶点误导
                Kx.vertices = np.array([np.max(Kx.vertices), -np.max(Kx.vertices)]).reshape(-1, 1)
                self.constr_u.append(self.U.minkowskiDiff(Kx))
            else:
                Kx = self.mrpi.affineMap(self.K)
                # 防止错误的顶点误导
                Kx.vertices = np.array([np.max(Kx.vertices), -np.max(Kx.vertices)]).reshape(-1, 1)
                self.constr_u.append(self.U.minkowskiDiff(Kx))

            self.constr_u[i].vertices = np.round(self.constr_u[i].vertices, 3)
            self.constr_u[i].compute_Hrep()

            self.Kxs.append(Kx)
        # 加了N这个参数，所以求的已经不是mrpi而是前五步的mrpi，已经够用了

            # self.len_conu += self.constr_u[i].A.shape[0]

        a = 1
    def solve(self, x0, time=0, verbose=False, SS=None, Qfun=None, CVX=None):
        """This method solves an FTOCP given:
			- x0: initial condition
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
		"""
        # Initialize Variables

        self.len_conx = 0
        self.len_conu = 0

        for i in range(self.N+1):
            self.len_conx += self.constr_x[time+i].A.shape[0]

        for i in range(self.N):
            self.len_conu += self.constr_u[time+i].A.shape[0]

        x = MX.sym('x', self.n*(self.N+1))
        u = MX.sym('u', self.d*self.N)
        # State Constraints
        constraints = []
        for i in range(self.N+1):
            constraints = vertcat(constraints,
                                  mtimes(self.constr_x[time+i].A, x[self.n * i:self.n * (i + 1)])-self.constr_x[time+i].b)
        for i in range(self.N):
            constraints = vertcat(constraints, mtimes(self.constr_u[time+i].A,
                                                      u[self.d * i:self.d * (i + 1)])-self.constr_u[time+i].b)
        # constraints = vertcat(constraints, x[:self.n]-np.array(x0))
        cost = 0
        for i in range(self.N):
            constraints = vertcat(constraints,
                                  x[self.n * (i + 1):self.n * (i + 2)] -
                                  mtimes(self.A, x[self.n * i:self.n * (i + 1)]) -
                                  mtimes(self.B, u[self.d * i:self.d * (i + 1)]))
            cost = cost + mtimes(mtimes(x[self.n * i:self.n * (i + 1)].T, self.Q),
                                 x[self.n * i:self.n * (i + 1)]) + \
                    mtimes(mtimes(u[self.d * i:self.d * (i + 1)].T, self.R), u[self.d * i:self.d * (i + 1)])
            if i == 0:
                cost = cost + mtimes(mtimes(u[self.d * i:self.d * (i + 1)].T, self.R_delta),
                                     u[self.d * i:self.d * (i + 1)])
            else:
                cost = cost + mtimes(mtimes((u[self.d * i:self.d * (i + 1)] - u[self.d * (i-1):self.d * i]).T,
                                            self.R_delta),
                                     (u[self.d * i:self.d * (i + 1)] - u[self.d * (i-1):self.d * i]))

        # SS的shape应该是 n × SS中点的个数
        # 如果SS是None，末项就是简单的二次型，否则是SS中各个点value的加权和
        if SS is not None:

            # K改变之后,mrpi随之改变,所以SS里面可能存在不属于X-Z的元素
            SS_ = []
            Qfun_ = []
            if self.N+1 <= self.s:
                cons = self.X.minkowskiDiff(self.F_list[self.N])
            else:
                cons = self.X.minkowskiDiff(self.mrpi)
            cons.minVrep()
            if not cons.hasHrep:
                try:
                    cons.vertices = np.round(cons.vertices, 3)
                    cons.compute_Hrep()
                except RuntimeError:
                    cons.vertices = np.round(cons.vertices)
                    cons.compute_Hrep()
            for i in range(SS.shape[1]):
                if cons.contains(SS[:, i]):
                    SS_.append(SS[:, i].tolist())
                    Qfun_.append(Qfun[0, i])
            SS_ = np.array(SS_).T
            Qfun_ = np.array(Qfun_).reshape(1, -1)
            # SS_ = SS.copy()
            # Qfun_ = Qfun.copy()
            lambVar = MX.sym('lam', SS_.shape[1])
            # Terminal Constraint if SS not empty --> enforce the terminal constraint
            for i in range(self.n):
                constraints = vertcat(constraints, x[self.n * self.N + i] - dot(SS_[i], lambVar))
            constraints = vertcat(constraints, dot(np.ones(SS_.shape[1]), lambVar) - 1)
            # constraints = vertcat(constraints, lambVar - 0)
            # Terminal cost if SS not empty
            cost = cost + dot(Qfun_[0], lambVar)  # Its terminal cost is given by interpolation using \lambda
            options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                       "ipopt.mu_strategy": "adaptive", "ipopt.dual_inf_tol":1e-10,
                       "ipopt.acceptable_dual_inf_tol": 1, "ipopt.theta_max_fact": 0.9,
                        "ipopt.constr_viol_tol": 1e-10, "ipopt.acceptable_constr_viol_tol": 1e-5}
            nlp = {'x': vertcat(x, u, lambVar), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            lbg = [-np.inf]*(self.len_conx + self.len_conu) + [0.] * (self.n * (self.N)) + [-0.] * self.n + [0] * 1
            ubg = [0]*(self.len_conx + self.len_conu) + [0.] * (self.n * (self.N)) + [0.] * self.n + [0] * 1
            lbx = [x0[0], x0[1]] + [-10., -10.] * int(self.n/2 * (self.N)) + [-1.] * (self.d * self.N) + [0.] * SS_.shape[1]
            ubx = [x0[0], x0[1]] + [10., 10.] * int(self.n/2 * (self.N)) + [1.] * (self.d * self.N) + [1.] * SS_.shape[1]
            sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            g = np.array(sol['g'][self.len_conx+self.len_conu:self.len_conx+self.len_conu+self.n*(self.N)])
            if np.sum(g) > 0.001:
                a = 1
                # return 0
        else:
            cost = cost + mtimes(mtimes(x[self.n * self.N:self.n * (self.N + 1)].T, self.Q),
                                 x[self.n * self.N:self.n * (self.N + 1)])  # If SS is not given terminal cost is quadratic
            options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                       "ipopt.mu_strategy": "adaptive",
                       "ipopt.mu_init": 1e-5, "ipopt.mu_min": 1e-15,
                       "ipopt.barrier_tol_factor": 1, "ipopt.gamma_theta": 0.01}
            nlp = {'x': vertcat(x, u), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            lbg = [-np.inf]*(self.len_conx + self.len_conu) + [0] * (self.n * (self.N))
            ubg = [0]*(self.len_conx + self.len_conu) + [0] * (self.n * (self.N))
            lbx = [x0[0], x0[1]] + [-10., -10.] * int(self.n/2 * (self.N)) + [-1.] * (self.d * self.N)
            ubx = [x0[0], x0[1]] + [10., 10.] * int(self.n/2 * (self.N)) + [1.] * (self.d * self.N)
            sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            g = np.array(sol['g'][-self.n * (self.N + 1):])
            if np.sum(g) > 0.001:
                a = 1
                # return 0
        res = np.array(sol['x'])

        xSol = res[0:(self.N + 1) * self.n].reshape((self.N + 1, self.n)).T
        uSol = res[(self.N + 1) * self.n:((self.N + 1) * self.n + self.d * self.N)].reshape(
            (self.N, self.d)).T
        self.xPred = xSol
        self.uPred = uSol
        return 1
        # Store the open-loop predicted trajectory

    def model(self, x, u):
        #     # Compute state evolution
        return (np.dot(self.A, x) + np.squeeze(np.dot(self.B, u))).tolist()
