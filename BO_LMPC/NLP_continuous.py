import types

import numpy as np
import pdb
import scipy
from cvxpy import *
from scipy.integrate import odeint
from objective_functions_lqr import inv_pendulum, get_params
from casadi import *


class FTOCP(object):
    """ Finite Time Optimal Control Problem (FTOCP)
	Methods:
		- solve: solves the FTOCP given the initial condition x0, terminal contraints (optinal) and terminal cost (optional)
		- model: given x_t and u_t computes x_{t+1} = Ax_t + Bu_t

	"""

    def __init__(self, N, A, B, Q, R, params, args):
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

        # Initialize Predicted Trajectory
        self.xPred = []
        self.uPred = []

        self.params = params
        T1 = params['T1']
        K = params['K']
        m = params['mass_pole']
        M = params['mass_cart']
        l = params['length_pole']
        bc = params['friction_coef_bc']
        bp = params['friction_coef_bp']
        mu_friction = bc
        g = 9.81
        d = 0.005
        Jd = m * (l / 2) ** 2 + 1 / 12 * m * l ** 2 + 0.25 * m * (d / 2) ** 2  # 转动惯量

        x1 = MX.sym('x')
        x2 = MX.sym('dx')
        x3 = MX.sym('theta')
        x4 = MX.sym('dtheta')
        x = vertcat(x1, x2, x3, x4)
        u = MX.sym('u')
        ode = [x2,
               (1 / T1 * (K * u - x2)),
               x4,
               0.5 * m * g * l / Jd * np.sin(x3)
               - 0.5 * m * l / Jd * np.cos(x3) * 1 / T1 * (K * u - x2)
               - mu_friction / Jd * x4
               ]
        # ode = [x2,
        #        1 / (M + m - (m ** 2 * l ** 2 * np.cos(x3) ** 2) / (Jd + m * l ** 2)) * \
        #        (u - bc * x2 - m * l * x4 ** 2 * np.sin(x3) + m ** 2 * l ** 2 * g * np.sin(x3) * np.cos(x3) / (
        #                Jd + m * l ** 2) -
        #         m * l * bp * x4 * np.cos(x3) / (Jd + m * l ** 2)),
        #        x4,
        #        1 / (Jd + m * l ** 2 - (m ** 2 * l ** 2 * np.cos(x3) ** 2) / (M + m)) * \
        #        (m * l * np.cos(x3) / (M + m) * u - bp * x4 + m * g * l * np.sin(x3) -
        #         m ** 2 * l ** 2 * x4 ** 2 * np.sin(x3) * np.cos(x3) / (M + m) - m * l * bc * x2 * np.cos(x3) / (
        #                 M + m))
        #        ]

        f = Function('f', [x, u], [vcat(ode)], ['state', 'input'], ['ode'])
        intg_options = {'tf': 0.1}
        dae = {'x': x, 'p': u, 'ode': f(x, u)}
        intg = integrator('intg', 'rk', dae, intg_options)
        res = intg(x0=x, p=u)
        x_next = res['xf']
        self.F = Function('F', [x, u], [x_next], ['state', 'input'], ['x_next'])
        self.args = args
    def solve(self, x0, verbose=False, SS=None, Qfun=None, CVX=None):
        """This method solves an FTOCP given:
			- x0: initial condition
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
		"""
        # Initialize Variables
        if SS is not None:
            lambVar = MX.sym('lam', SS.shape[1])
            # cost_final = np.inf
            # length_ = SS.shape[1]
            # idx2try = range(length_-1, 0, -1)
            # for j in idx2try:
            #     if Qfun[0][j] > cost_final:
            #         continue
            #     x_terminal = SS[:, j]
            X = MX.sym('X', self.n * (self.N + 1))
            U = MX.sym('U', self.d * self.N)
            cost = 0

            constraints = []
            constraints = vertcat(constraints, X[:self.n] - np.array(x0))
            for i in range(self.N):
                constraints = vertcat(constraints,
                                      X[self.n * (i + 1):self.n * (i + 2)] - self.F(
                                          X[self.n * i:self.n * (i + 1)],
                                          U[self.d * i:self.d * (i + 1)]))
                cost = cost + mtimes(mtimes(X[self.n * i:self.n * (i + 1)].T,
                       self.Q), X[self.n * i:self.n * (i + 1)]) + \
                    mtimes(mtimes(U[self.d * i:self.d * (i + 1)].T, self.R), U[self.d * i:self.d * (i + 1)])

            # constraints = vertcat(constraints, X[self.n*(self.N):self.n*(self.N+1)]-mtimes(SS, lambVar))
            for i in range(self.n):
                constraints = vertcat(constraints,
                                  X[self.n * self.N + i] - dot(SS[i], lambVar))
            constraints = vertcat(constraints, dot(np.ones(SS.shape[1]), lambVar) - 1)
            # constraints = vertcat(constraints, dot(lambVar, lambVar) - 1)
            # constraints = vertcat(constraints, lambVar - 0)
            # cost = cost + Qfun[0][j]
            # for idx in range(SS.shape[1]):
            cost = cost + dot(Qfun[0], lambVar)
            options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                       "ipopt.mu_strategy": "adaptive",
                       "ipopt.mu_init": 1e-5, "ipopt.mu_min": 1e-15,
                       "ipopt.barrier_tol_factor": 1}
            nlp = {'x': vertcat(X, U, lambVar), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            lbg = [0] * (self.n * (self.N + 1)) + [0] * self.n + [0] * 1
            ubg = [0] * (self.n * (self.N + 1)) + [0] * self.n + [0] * 1
            lbx = [-np.inf] * (self.n * (self.N + 1)) + [-self.args['u_limit']] * (self.d * self.N) + [0] * SS.shape[1]
            ubx = [np.inf] * (self.n * (self.N + 1)) + [self.args['u_limit']] * (self.d * self.N) + [1] * SS.shape[1]
            sol = solver(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            res = np.array(sol['x'])
            xSol = res[0:(self.N + 1) * self.n].reshape((self.N + 1, self.n)).T
            uSol = res[(self.N + 1) * self.n:((self.N + 1) * self.n + self.d * self.N)].reshape(
                (self.N, self.d)).T
            # cost = sol['f']
            # if cost < cost_final:
            #     cost_final = cost
            self.xPred = xSol
            self.uPred = uSol
        else:
            X = MX.sym('X', self.n * (self.N + 1))
            U = MX.sym('U', self.d * self.N)
            cost = 0

            constraints = []
            constraints = vertcat(constraints, X[:self.n] - np.array(x0))
            for i in range(self.N):
                constraints = vertcat(constraints,
                                      X[self.n * (i + 1):self.n * (i + 2)] - self.F(X[self.n * i:self.n * (i + 1)],
                                                                            U[self.d * i:self.d * (i + 1)]))
                cost = cost + mtimes(mtimes(X[self.n * i:self.n * (i + 1)].T,
                                            self.Q), X[self.n * i:self.n * (i + 1)]) + \
                       mtimes(mtimes(U[self.d * i:self.d * (i + 1)].T, self.R), U[self.d * i:self.d * (i + 1)])

            cost = cost + mtimes(mtimes(X[self.n * self.N:self.n * (self.N + 1)].T,
                                            self.Q), X[self.n * self.N:self.n * (self.N + 1)])

            options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                       "ipopt.mu_strategy": "adaptive",
                       "ipopt.mu_init": 1e-5, "ipopt.mu_min": 1e-15,
                       "ipopt.barrier_tol_factor": 1}
            nlp = {'x': vertcat(X, U), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            lbx = [-np.inf] * (self.n * (self.N + 1)) + [-self.args['u_limit']] * (self.d * self.N)
            ubx = [np.inf] * (self.n * (self.N + 1)) + [self.args['u_limit']] * (self.d * self.N)
            # xGuess = self.xGuessTot = np.concatenate((np.array(x0), np.zeros((self.n+self.d) * self.N)), axis=0)
            sol = solver(lbx=lbx, ubx=ubx, lbg=0, ubg=0)
            res = np.array(sol['x'])
            xSol = res[0:(self.N + 1) * self.n].reshape((self.N + 1, self.n)).T
            uSol = res[(self.N + 1) * self.n:((self.N + 1) * self.n + self.d * self.N)].reshape(
                (self.N, self.d)).T
            self.xPred = xSol
            self.uPred = uSol
                # x = Variable((self.n, self.N + 1))
        # Store the open-loop predicted trajectory

    # def model(self, x, u):
    #     # Compute state evolution
    #     return (np.dot(self.A, x) + np.squeeze(np.dot(self.B, u))).tolist()
