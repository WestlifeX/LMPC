from scipy.stats import norm
import numpy as np
import torch
from scipy.optimize import minimize

def opt_acquision(model, bounds, beta,
                  ts=True, dt=None, prior=None, random_search=False, n_restarts=1):
    """
    The acqusition function tries to figure out
    which point is most effective/economic to evaluate with the object function
    """
    # random sampling, instead of BFGS algorithm
    n_theta = bounds.shape[0]
    n_data = 10000
    bounds = np.array(bounds)
    best_x = None
    if random_search:
        Xsamples = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_data, n_theta))
        scores = acquisition(Xsamples, model, beta, ts, dt)
        # pick the one with the highest 'acquisition score', or most poorly constrained point.
        ix = np.argmin(scores)
        best_x = Xsamples[ix, :]
    # if prior is not None:
    #     Xsamples[:1, :] = prior
    # calculate the acquisition function for each sample
    else:
        best_value = np.inf
        point_list = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_theta))
        if prior is not None:
            point_list = np.vstack((point_list, prior))
        # else:
        #     point_list = np.vstack((point_list, np.array([1]*n_theta).reshape(1, -1)))
        for starting_point in point_list:

            res = minimize(ucb,
                           x0=starting_point.reshape(1, -1),
                           method='L-BFGS-B',
                           args=(model, beta, ts),
                           bounds=bounds)

            if res.fun < best_value:
                best_value = res.fun
                best_x = res.x

    return best_x

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
            probs = - (mu + beta * std.reshape(-1, 1)).detach().numpy()
    else:
        probs = - (mu + beta * std.reshape(-1, 1))
    return probs

def ucb(x, model, beta, ts):
    # with torch.no_grad():
    x = x.reshape(1, -1)
    if ts:
        x = torch.tensor(x, dtype=torch.float32)
        # return -(mu + beta * std(-1, 1))
        return (model.predict(x, return_std=True)[0] - beta * model.predict(x, return_std=True)[1]).detach().numpy()
    else:
        return model.predict(x, return_std=True)[0] - beta * model.predict(x, return_std=True)[1]
