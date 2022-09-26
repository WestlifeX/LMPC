from scipy.stats import norm
import numpy as np
import torch
from scipy.optimize import minimize
def opt_acquision(X, model, bounds, beta):
    """
    The acqusition function tries to figure out
    which point is most effective/economic to evaluate with the object function
    """
    # random sampling, instead of BFGS algorithm
    n_theta = 4
    n_data = 500
    Xsamples = np.zeros((n_data, n_theta))
    for i in range(n_theta):
        Xsamples[:, i] = np.linspace(0.2, 5, n_data)
    # Xsamples = np.random.random((100, n_theta))
    # for i in range(n_theta):
    #     Xsamples[:, i] = Xsamples[:, i] * (bounds[1][i] - bounds[0][i]) + bounds[0][i]
    Xsamples = Xsamples.reshape(-1, n_theta)
    # calculate the acquisition function for each sample
    # res = minimize(ucb, x0=np.array([1, 1, 1, 1]), args=(model, beta), bounds=bounds)
    scores = acquisition(X, Xsamples, model, beta)
    # pick the one with the highest 'acquisition score', or most poorly constrained point.
    ix = np.argmin(scores)
    return Xsamples[ix, :]
    # return res.x

def acquisition(X, Xsamples, model, beta):
    """
    We use PI, the simplest acqusition function.
    """
    # yhat, _ = model.predict(X, return_var=True)
    # X will be updated every iteration.
    # Don't worry too much.
    # We won't go through too many iterations.
    # current_best = torch.max(yhat).view(-1, 1)

    Xsamples = torch.tensor(Xsamples, dtype=torch.float32)
    mu, std = model.predict(Xsamples, return_var=True)
    # mu = mu[:, 0]
    # std = torch.sqrt(std)
    # Probability of Improvement
    # probs = (mu - beta * std.reshape(-1, 1))
    with torch.no_grad():
        probs = (mu - beta * std.reshape(-1, 1)).detach().numpy()
    return probs

def ucb(x, model, beta):
    x = x.reshape(1, -1)
    mu, std = model.predict(x, return_std=True)
    # return -(mu + beta * std(-1, 1))
    return model.predict(x, return_std=True)[0] - beta * model.predict(x, return_std=True)[1]
