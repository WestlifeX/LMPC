import torch
import numpy as np

from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.acquisition import ExpectedImprovement, UpperConfidenceBound

import matplotlib.pyplot as plt

# %% md
# Fit GP model to dataset
# The following function (get_model) fits a Gaussian Process to a set of observations.

## initialize a GP with data
## use likelihood to find GP params
def get_model(train_x, train_y, state_dict=None, debug=False):
    gp = SingleTaskGP(train_x, train_y)
    if debug:
        print("Prior hyperparams lengthscale & noise: {}, {}".format(gp.covar_module.base_kernel.lengthscale.item(),
                                                                     gp.likelihood.noise.item()))
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    if state_dict is not None:
        gp.load_state_dict(state_dict)  # speeds up fit
    fit_gpytorch_model(mll)  # performs the hyperparam fit
    if debug:
        print("Post hyperparams lengthscale & noise:  {}, {}".format(gp.covar_module.base_kernel.lengthscale.item(),
                                                                     gp.likelihood.noise.item()))
    return gp, mll

### Target Function
# The function we would like to maximize.
def target(x):
    return torch.exp(-(x - 2) ** 2) + torch.exp(-(x - 6) ** 2 / 10) + 1 / (x ** 2 + 1)


### Random Init
# Randomly sample search space to warm up Gaussian Process
train_x = torch.FloatTensor(4, 1).uniform_(-2., 10.)

# train_x   = torch.tensor([[3.109], [7.775]])
train_obj = target(train_x)
model, mll = get_model(train_x, train_obj, debug=True)

train_x_explore = torch.clone(train_x)
train_obj_explore = torch.clone(train_obj)
model_explore, mll_explore = get_model(train_x, train_obj, debug=True)
# %% md
### Plot Target
# We can see that the max is at x~2 .
# %%
x = torch.linspace(-2, 10, 10000).reshape(-1, 1)
y = target(x)
plt.plot(x, y)


# The following function finds the maximum of the Acquisition Function. (only used for plotting purposes)
def get_max(model, beta):
    UCB = UpperConfidenceBound(model=model, beta=beta)
    new_point_analytic, acq_value_list = optimize_acqf(
        acq_function=UCB,
        bounds=torch.tensor([[-2.], [10.]]),
        q=1,  # number of candidates
        num_restarts=20,  # The number of starting points for multistart acquisition function optimization.
        raw_samples=100,  # The number of samples for initialization.
        options={},
        return_best_only=True,
        sequential=False  # If False, uses joint optimization, otherwise uses sequential optimization.
    )
    return new_point_analytic, acq_value_list


# %% md
### One Step of Bayes Opt
# step probes a single query point (chosen by the acquisition function) and updates the GP.
def step(model, mll, train_x, train_obj, beta=5.):
    # optimize acquisition function
    UCB = UpperConfidenceBound(model=model, beta=beta)
    new_point_analytic, acq_value_list = optimize_acqf(
        acq_function=UCB,
        bounds=torch.tensor([[-2.], [10.]]),
        q=1,
        num_restarts=20,
        raw_samples=100,
        options={},
        return_best_only=True,
        sequential=False
    )
    smth = target(new_point_analytic)
    train_obj = torch.cat([smth, train_obj])
    train_x = torch.cat([new_point_analytic, train_x])

    model, mll = get_model(train_x, train_obj, model.state_dict())
    return model, mll, train_x, train_obj


# %% md
### Plot mean prediction and uncertainty
# Plot mean prediction and uncertainty
# Plots to compare the decision making of two different models.
def plot():
    with torch.no_grad():
        mean_preds = model(x).mean
        std_preds = model(x).stddev
        lower, upper = model(x).confidence_region()
        mean_preds_explore = model_explore(x).mean
        lower_explore, upper_explore = model_explore(x).confidence_region()
        std_preds_explore = model_explore(x).stddev

    _argmax, _max = get_max(model, 1.)  # max of acq function
    _argmax_explore, _max_explore = get_max(model_explore, 25.)  # max of acq function
    acq = mean_preds + std_preds
    acq_explore = mean_preds_explore + 5. * std_preds_explore
    fix, ax = plt.subplots(2, 2, sharey=True, figsize=(16, 8))
    ax[0, 0].set_title("Exploitation")
    ax[0, 0].plot(x, mean_preds, label='mean')
    ax[0, 0].plot(x, y, label='target')
    ax[0, 0].fill_between(x.flatten(), lower, upper, alpha=0.5, label='uncertainty')
    ax[0, 0].scatter(train_x, train_obj, label='observations')
    ax[0, 0].legend()
    ax[1, 0].set_title("Acquisition Function: $beta$=1")
    ax[1, 0].plot(x, acq, label='UCB')
    ax[1, 0].scatter(_argmax, _max, marker='d', color='r', label='Max')
    ax[0, 1].set_title("Exploration")
    ax[0, 1].plot(x, mean_preds_explore, label='mean')
    ax[0, 1].plot(x, y, label='target')
    ax[0, 1].fill_between(x.flatten(), lower_explore, upper_explore, alpha=0.5, label='uncertainty')
    ax[0, 1].scatter(train_x_explore, train_obj_explore, label='observations')
    ax[1, 1].set_title("Acquisition Function: $beta$=25")
    ax[1, 1].plot(x, acq_explore, label='UCB')
    ax[1, 1].scatter(_argmax_explore, _max_explore, marker='d', color='r', label='Max')
    ax[1, 1].legend()


# %% md
#### Initial Plot with 4 random observations
# %%
plot()
# %%
for i in range(10):
    model, mll, train_x, train_obj = step(model, mll, train_x, train_obj, beta=1.)
    model_explore, mll_explore, train_x_explore, train_obj_explore = step(model_explore, mll_explore, train_x_explore,
                                                                      train_obj_explore, beta=25.)
    plot()
# The exploitation model fixates on the maximum of the function,
# while the exploration model builds a more complete surrogate model.
# They both are able to learn the maximum. But, with poor random initialization,
# the exploitation model will not be able to learn the global maximum.