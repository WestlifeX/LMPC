from scipy.stats import norm
import numpy as np
import torch
from scipy.optimize import minimize
def opt_acquision(model, bounds, beta, ts=True, dt=None, prior=None):
    """
    The acqusition function tries to figure out
    which point is most effective/economic to evaluate with the object function
    """
    # random sampling, instead of BFGS algorithm
    n_theta = len(bounds)
    n_data = 10000
    # Xsamples = np.zeros((n_data, n_theta))
    # for i in range(n_theta):
    #     Xsamples[:, i] = np.linspace(0.2, 5, n_data)  # 这是非常蠢的做法！完全没有遍历到整个空间，所以效果差也是显然的
    Xsamples = np.random.random((n_data, n_theta))
    for i in range(n_theta):
        Xsamples[:, i] = Xsamples[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    Xsamples = Xsamples.reshape(-1, n_theta)
    if prior is not None:
        Xsamples[:10, :] = prior
    # calculate the acquisition function for each sample
    # res = minimize(ucb, x0=np.array([1, 1]), args=(model, beta), bounds=bounds)
    scores = acquisition(Xsamples, model, beta, ts, dt)
    # pick the one with the highest 'acquisition score', or most poorly constrained point.
    ix = np.argmin(scores)
    scores_sorted = np.argsort(scores, 0)[:10]
    return np.squeeze(Xsamples[scores_sorted, :], axis=1)
    # return res.x

def acquisition(Xsamples, model, beta, ts=True, dt=None):
    """
    We use PI, the simplest acqusition function.
    """
    # yhat, _ = model.predict(X, return_var=True)
    # X will be updated every iteration.
    # Don't worry too much.
    # We won't go through too many iterations.
    # current_best = torch.max(yhat).view(-1, 1)
    if ts:
        Xsamples = torch.tensor(Xsamples, dtype=torch.float32)
    if dt is not None:
        mu, std = model.predict(Xsamples, dt, return_std=True)
    else:
        mu, std = model.predict(Xsamples, return_std=True)
    # mu = mu[:, 0]
    # std = torch.sqrt(std)
    # Probability of Improvement
    # probs = (mu - beta * std.reshape(-1, 1))
    if ts:
        with torch.no_grad():
            probs = (mu - beta * std.reshape(-1, 1)).detach().numpy()
    else:
        probs = (mu - beta * std.reshape(-1, 1))
    return probs

def ucb(x, model, beta):
    # with torch.no_grad():
    x = x.reshape(1, -1)
        # x = torch.tensor(x, dtype=torch.float32)
    # return -(mu + beta * std(-1, 1))
    return model.predict(x, return_std=True)[0] - beta * model.predict(x, return_std=True)[1]
