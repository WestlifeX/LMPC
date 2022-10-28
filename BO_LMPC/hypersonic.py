import numpy as np
from scipy.linalg import expm
from scipy import integrate
# V = 15060
# gamma = 0
# h = 110000
# alpha = 0.0312
# q = 0
# beta = 0.1762
# delta_e = -0.0069
m = 9375
Iyy = 7e6
S = 3603
c_ = 80
ce = 0.0292
RE = 20902230.972
mu = 32.15  # 万有引力常数
def hypersonic(state, t, u):
    V, gamma, h, alpha, q = state
    beta, delta_e = u

    CL = 0.6203 * alpha
    CD = 0.6450 * alpha ** 2 + 0.0043378 * alpha + 0.003772
    CT = 0.02576 * beta if beta < 1 else 0.0224 + 0.00336 * beta
    CM_alpha = -0.035 * alpha ** 2 + 0.036617 * alpha + 5.3261e-6
    CM_delta_e = ce * (delta_e - alpha)
    CM_q = c_ / (2 * V) * q * (-6.796*alpha**2 + 0.3015*alpha - 0.2289)

    rho = 0.0023769 * np.exp(-h/24000)
    L = 0.5 * rho * V**2 * S * CL
    D = 0.5 * rho * V**2 * S * CD
    T = 0.5 * rho * V**2 * S * CT
    Myy = 0.5 * rho * V**2 * S * c_ * (CM_alpha + CM_delta_e + CM_q)
    r = h + RE
    dV = (T * np.cos(alpha) - D) / m - mu * np.sin(gamma)
    dgamma = (L + T * np.sin(alpha)) / (m * V) - mu * np.cos(gamma) / V
    dh = V * np.sin(gamma)
    dalpha = q - dgamma
    dq = Myy / Iyy
    return [dV, dgamma, dh, dalpha, dq]

def linear_model(Ts):
    V = 15060
    gamma = 0
    h = 110000
    alpha = 0.0312
    q = 0
    beta = 0.1762
    delta_e = -0.0069
    rho = 0.0023769 * np.exp(-h / (10.4 * 3280.83989501))
    Q = 0.5 * rho * V**2
    r = h + RE
    Ax = np.array([[-0.003772 * Q * S / (m * V), -mu/(r**2), 0., -(0.645*alpha+0.0043378)*Q*S/m, 0.],
                   [1/r, 0., -mu/(V*(r**2)*h), 0.6203*Q*S/(m*V), 0.],
                   [0., V, 0, 0, 0],
                   [-1/r, 0., mu/(V*(r**2)*h), -0.6203*Q*S/(m*V), 1.],
                   [5.3261e-6*Q*S*c_/(Iyy * V), 0, 0, (-0.0035*alpha + 0.036617 - ce) * Q * S * c_ / Iyy,
                    (-6.796*alpha**2+0.3015*alpha-0.2289)*Q*S*c_**2/(2*Iyy*V)]])

    Bx = np.array([[0.02576*Q*S*np.cos(alpha)/m, 0],
                   [0.02576*Q*S*np.sin(alpha)/(m*V), 0],
                   [0, 0],
                   [-0.02576*Q*S*np.sin(alpha)/(m*V), 0],
                   [0, Q*S*c_*ce/Iyy]])

    A = expm(Ax * Ts)
    B = np.zeros_like(Ax)
    for t in np.linspace(0, 1, 1000):
        B = B + f(t, Ax)
    B = B / 1000
    B = np.dot(B, Bx)
    return A, B

def f(t, A):
    return expm(A * t)