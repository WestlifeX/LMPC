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
        # x = ca.SX.sym('x', self.n, self.N + 1, boolean=True)
        x = Variable((self.n, self.N + 1))
        u = Variable((self.d, self.N))
        constr = []
        for i in range(self.N+1):
            constr += [self.constr_x[time+i].A @ x[:, i] <= self.constr_x[time+i].b.reshape(-1, ), ]
        for i in range(self.N):
            constr += [self.constr_u[time+i].A @ u[:, i] <= self.constr_u[i].b.reshape(-1,), ]
        # State Constraints
        constr += [x[:, 0] == x0[:]]  # initializing condition
        for i in range(self.N):
            constr += [x[:, i + 1] == self.A @ x[:, i] + self.B @ u[:, i], ]
            # constr += [x[:, i + 1] == odeint(inv_pendulum, x[:, i], [0, 0.1], args=(u[:, i], params))[1]]
        # Cost Function
        cost = 0
        for i in range(self.N):
            # Running cost h(x,u) = x^TQx + u^TRu
            cost += quad_form(x[:, i], self.Q) + quad_form(u[:, i], self.R)
            if i == 0:
                cost += quad_form(u[:, i], self.R_delta)
            else:
                cost += quad_form(u[:, i] - u[:, i - 1], self.R_delta)
        # cost += quad_form(x[:, self.N], self.Q)
        # cost += norm(self.Q**0.5*x[:,i])**2 + norm(self.R**0.5*u[:,i])**2

        # If SS is given initialize lambdaVar multipliers used to enforce terminal constraint
        # SS的shape应该是 n × SS中点的个数
        # 如果SS是None，末项就是简单的二次型，否则是SS中各个点value的加权和
        if SS is not None:
            # K改变之后,mrpi随之改变,所以SS里面可能存在不属于X-Z的元素
            SS_ = []
            Qfun_ = []
            if self.N + 1 <= self.s:
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
            if CVX:
                lambVar = Variable((SS_.shape[1], 1), boolean=False)  # Initialize vector of variables
            else:
                lambVar = Variable((SS_.shape[1], 1), boolean=True)  # Initialize vector of variables
            # cost += quad_form(x[:, self.N], self.Q)
            # Terminal Constraint if SS not empty --> enforce the terminal constraint
            # 改变Q的话，xN就会不一样，xN不一样约束这里就会不一样
            constr += [SS_ @ lambVar[:, 0] == x[:, self.N],  # Terminal state \in ConvHull(SS)
                       np.ones((1, SS_.shape[1])) @ lambVar[:, 0] == 1,  # Multiplies \lambda sum to 1
                       lambVar >= 0]  # Multiplier are positive definite

            # Terminal cost if SS not empty
            cost += Qfun_[0, :] @ lambVar[:, 0]  # Its terminal cost is given by interpolation using \lambda

            self.lamb = lambVar.value
        else:
            cost += quad_form(x[:, self.N], self.Q)  # If SS is not given terminal cost is quadratic

        # Solve the Finite Time Optimal Control Problem
        problem = Problem(Minimize(cost), constr)
        if CVX:
            # problem.solve(verbose=verbose)
            problem.solve(verbose=True, solver=MOSEK)  # I find that ECOS is better please use it when solving QPs 3q
        else:
            problem.solve(verbose=True, solver=MOSEK)
        self.xPred = x.value
        self.uPred = u.value
        # Store the open-loop predicted trajectory

    def model(self, x, u):
        #     # Compute state evolution
        return (np.dot(self.A, x) + np.squeeze(np.dot(self.B, u))).tolist()
