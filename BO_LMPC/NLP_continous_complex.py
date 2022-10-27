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
        self.C = C
        self.n = A.shape[1]
        self.d = B.shape[1]

        # Cost (h(x,u) = x^TQx +u^TRu)
        self.Q = Q
        self.R = R

        # Initialize Predicted Trajectory
        self.xPred = []
        self.uPred = []

        m = 9375
        Iyy = 7e6
        S = 3603
        c_ = 80
        ce = 0.0292
        RE = 20902230.972
        mu = 6.6743e-11 * 3.28083989501 ** 3 * 14.59


        V = MX.sym('V')
        gamma = MX.sym('gamma')
        h = MX.sym('h')
        alpha = MX.sym('alpha')
        q = MX.sym('q')
        x = vertcat(V, gamma, h, alpha, q)
        beta = MX.sym('beta')
        delta_e = MX.sym('delta_e')
        u = vertcat(beta, delta_e)

        CL = 0.6203 * alpha
        CD = 0.6450 * alpha ** 2 + 0.0043378 * alpha + 0.003772
        CT = 0.02576 * beta if beta < 1 else 0.0224 + 0.00336 * beta
        CM_alpha = -0.035 * alpha ** 2 + 0.036617 * alpha + 5.3261e-6
        CM_delta_e = ce * (delta_e - alpha)
        CM_q = c_ / (2 * V) * q * (-6.796 * alpha ** 2 + 0.3015 * alpha - 0.2289)

        rho = 0.0023769 * np.exp(-h / (10.4 * 3280.83989501))
        L = 0.5 * rho * V ** 2 * S * CL
        D = 0.5 * rho * V ** 2 * S * CD
        T = 0.5 * rho * V ** 2 * S * CT
        Myy = 0.5 * rho * V ** 2 * S * c_ * (CM_alpha + CM_delta_e + CM_q)
        r = h + RE

        ode = [(T * np.cos(alpha) - D) / m - mu * np.sin(gamma) / (r ** 2),
               (L + T * np.sin(alpha)) / (m * V) - (mu - V**2 * r) * np.cos(gamma) / (V * r**2),
               V * np.sin(gamma),
               q - dgamma,
               Myy - Iyy
               ]
        f = Function('f', [x, u], [vcat(ode)], ['state', 'input'], ['ode'])
        intg_options = {'tf': 0.1}  # 记得跟着改
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
                cost = cost + mtimes(mtimes(mtimes(X[self.n * i:self.n * (i + 1)].T, self.C.T),
                       self.Q), mtimes(self.C, X[self.n * i:self.n * (i + 1)])) + \
                    mtimes(mtimes(U[self.d * i:self.d * (i + 1)].T, self.R), U[self.d * i:self.d * (i + 1)])

            for i in range(self.n):
                constraints = vertcat(constraints,
                                  X[self.n * self.N + i] - dot(SS[i], lambVar))
            constraints = vertcat(constraints, dot(np.ones(SS.shape[1]), lambVar) - 1)
            # constraints = vertcat(constraints, dot(lambVar, lambVar) - 1)
            constraints = vertcat(constraints, lambVar - 0)
            # cost = cost + Qfun[0][j]
            # for idx in range(SS.shape[1]):
            cost = cost + dot(Qfun[0], lambVar)
            options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                       "ipopt.mu_strategy": "adaptive",
                       "ipopt.mu_init": 1e-5, "ipopt.mu_min": 1e-15,
                       "ipopt.barrier_tol_factor": 1}
            nlp = {'x': vertcat(X, U, lambVar), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            lbg = [0] * (self.n * (self.N + 1)) + [0] * self.n + [0] * 1 + [0] * SS.shape[1]
            ubg = [0] * (self.n * (self.N + 1)) + [0] * self.n + [0] * 1 + [1] * SS.shape[1]
            sol = solver(lbg=lbg, ubg=ubg)
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
                                            self.Q), X[self.n * self.N:self.n * (self.N + 1)]) + \
            mtimes(mtimes(U[self.d * self.N:self.d * (self.N + 1)].T, self.R), U[self.d * self.N:self.d * (self.N + 1)])

            options = {"verbose": False, "ipopt.print_level": 0, "print_time": 0,
                       "ipopt.mu_strategy": "adaptive",
                       "ipopt.mu_init": 1e-5, "ipopt.mu_min": 1e-15,
                       "ipopt.barrier_tol_factor": 1}
            nlp = {'x': vertcat(X, U), 'f': cost, 'g': constraints}
            solver = nlpsol('solver', 'ipopt', nlp, options)
            # xGuess = self.xGuessTot = np.concatenate((np.array(x0), np.zeros((self.n+self.d) * self.N)), axis=0)
            sol = solver(lbg=0, ubg=0)
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
