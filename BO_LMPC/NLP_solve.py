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

    def __init__(self, N, A, B, Q, R, params):
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
        self.T1 = self.params['T1']
        self.K = self.params['K']
        self.mp = self.params['mass_pole']
        self.l = self.params['length_pole']
        self.mu_friction = self.params['friction_coef']
        self.g = 9.81
        self.d_ = 0.005
        self.Jd = self.mp * (self.l / 2) ** 2 + 1 / 12 * self.mp * self.l ** 2 + 0.25 * self.mp * (self.d_ / 2) ** 2

        x1 = MX.sym('x')
        x2 = MX.sym('dx')
        x3 = MX.sym('theta')
        x4 = MX.sym('dtheta')
        x = vertcat(x1, x2, x3, x4)
        u = MX.sym('u')
        ode = [x2,
               (1 / self.T1 * (self.K * u - x2)),
               x4,
               0.5 * self.mp * self.g * self.l / self.Jd * np.sin(x3)
               - 0.5 * self.mp * self.l / self.Jd * np.cos(x3) * 1 / self.T1 * (self.K * u - x3)
               - self.mu_friction / self.Jd * x4
               ]
        f = Function('f', [x, u], [vcat(ode)], ['state', 'input'], ['ode'])
        intg_options = {'tf': 0.1}
        dae = {'x': x, 'p': u, 'ode': f(x, u)}
        intg = integrator('intg', 'rk', dae, intg_options)
        res = intg(x0=x, p=u)
        x_next = res['xf']
        self.F = Function('F', [x, u], [x_next], ['state', 'input'], ['x_next'])
        a = 1
    def solve(self, x0, verbose=False, SS=None, Qfun=None, CVX=None):
        """This method solves an FTOCP given:
			- x0: initial condition
			- SS: (optional) contains a set of state and the terminal constraint is ConvHull(SS)
			- Qfun: (optional) cost associtated with the state stored in SS. Terminal cost is BarycentrcInterpolation(SS, Qfun)
		"""
        # Initialize Variables
        if SS is not None:
            cost_final = np.inf
            for id in range(len(SS)):
                length_ = len(SS[id])
                idx2try = range(5, length_)
                for j in idx2try:
                    x_terminal = SS[id][:, j]
                    X = MX.sym('X', self.n * (self.N + 1))
                    U = MX.sym('U', self.d * self.N)
                    cost = 0
                    constraints = types.SimpleNamespace()

                    constraints = []
                    constraints = vertcat(constraints, X[:self.n] - x0[:])
                    for i in range(self.N):
                        constraints = vertcat(constraints,
                                              X[self.n * (i + 1):self.n * (i + 2)] - self.F(
                                                  X[self.n * i:self.n * (i + 1)],
                                                  U[self.d * i:self.d * (i + 1)]))
                        cost = cost + self.Q[0, 0] * X[self.n * i] ** 2 + \
                               self.Q[1, 1] * X[self.n * i + 1] ** 2 + \
                               self.Q[2, 2] * X[self.n * i + 2] ** 2 + \
                               self.Q[3, 3] * X[self.n * i + 3] ** 2 + \
                               self.R[0, 0] * U[self.d * i] ** 2

                    constraints = vertcat(constraints, X[self.n * self.N:self.n * (self.N + 1)] - x_terminal)
                    cost = cost + Qfun[id][j]

                    options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                               "ipopt.mu_strategy": "adaptive",
                               "ipopt.mu_init": 1e-5, "ipopt.mu_min": 1e-15,
                               "ipopt.barrier_tol_factor": 1}
                    nlp = {'x': vertcat(X, U), 'f': cost, 'g': constraints}
                    solver = nlpsol('solver', 'ipopt', nlp, options)
                    xGuess = np.zeros((self.n + self.d) * self.N + self.n)
                    sol = solver(lbg=0, ubg=0)
                    res = np.array(sol['x'])
                    xSol = res[0:(self.N + 1) * self.n].reshape((self.N + 1, self.n)).T
                    uSol = res[(self.N + 1) * self.n:((self.N + 1) * self.n + self.d * self.N)].reshape(
                        (self.N, self.d)).T
                    cost = sol['f']
                    if cost < cost_final:
                        cost_final = cost
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
                cost = cost + self.Q[0, 0] * X[self.n * i] ** 2 + \
                       self.Q[1, 1] * X[self.n * i + 1] ** 2 + \
                       self.Q[2, 2] * X[self.n * i + 2] ** 2 + \
                       self.Q[3, 3] * X[self.n * i + 3] ** 2 + \
                       self.R[0, 0] * U[self.d * i] ** 2
            cost = cost + self.Q[0, 0] * X[self.n * self.N] ** 2 + \
                   self.Q[1, 1] * X[self.n * self.N + 1] ** 2 + \
                   self.Q[2, 2] * X[self.n * self.N + 2] ** 2 + \
                   self.Q[3, 3] * X[self.n * self.N + 3] ** 2

            # options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
            #            "ipopt.mu_strategy": "adaptive",
            #            "ipopt.mu_init": 1e-5, "ipopt.mu_min": 1e-15,
            #            "ipopt.barrier_tol_factor": 1}
            options = {}
            nlp = {'x': vertcat(X, U), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            xGuess = self.xGuessTot = np.concatenate((np.array(x0), np.zeros((self.n+self.d) * self.N)), axis=0)
            sol = solver(x0=xGuess)
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
