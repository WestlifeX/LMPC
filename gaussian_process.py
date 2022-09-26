import torch
from copy import deepcopy


class GaussianProcess(object):

    def __init__(self, kernel, alpha=1e-5):
        self.kernel = kernel
        self.alpha = alpha
        # Assuming the training data or objective function is noisy,
        # alpha should be set to the standard deviation of the noise

    # nll: negative log likelihood， 为了学超参
    def nll(self, x1, x2, y, det_tol=1e-12):
        b = y.size(0)  # n
        m = y.size(1)  # 1
        k = self.kernel(x1, x2) + self.alpha * torch.eye(b)
        nll = 0.5 * torch.log(torch.det(k) + torch.tensor(det_tol)) + \
              0.5 * y.view(m, b) @ torch.inverse(k) @ y.view(b, m)
        return nll

    # x: nxp, y: nx1
    def fit(self, x, y, lr=0.01, iters=100, restarts=0):
        b = x.size(0)  # n
        n = x.size(1)
        x1 = x.unsqueeze(1).expand(b, b, n)
        x2 = x.unsqueeze(0).expand(b, b, n)
        optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
        assert iters > 0
        for i in range(iters):
            optimiser.zero_grad()
            nll = self.nll(x1, x2, y)
            nll.backward()
            optimiser.step()
        best_nll = nll.item()
        params = self.kernel.get_params()
        for i in range(restarts):
            self.kernel.randomise_params()
            optimiser = torch.optim.Adam(self.kernel.parameters(), lr)
            for j in range(iters):
                optimiser.zero_grad()
                nll = self.nll(x1, x2, y)
                nll.backward()
                optimiser.step()
            if nll.item() < best_nll:
                best_nll = nll
                params = self.kernel.get_params()

        self.kernel.set_params(params)
        self.x = x
        self.y = y
        k = self.kernel(x1, x2).view(b, b) + self.alpha * torch.eye(b)
        self.kinv = torch.inverse(k)

    def predict(self, x, return_var=False):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        k = self.kernel(x1, x2).view(b_prime, b)
        mu = k @ self.kinv @ self.y
        x = x.unsqueeze(1).expand(b_prime, 1, n)
        sigma = self.kernel(x, x) - (k @ self.kinv @ k.t()).diag().view(mu.size())
        if return_var:
            return mu, sigma
        else:
            return mu

    def mu_grad(self, x):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        dk = self.kernel.grad(x1, x2)
        grad = (dk * (self.kinv @ self.y)).sum(1)
        return grad

    def sigma_grad(self, x):
        b_prime = x.size(0)
        b = self.x.size(0)
        n = self.x.size(1)
        x1 = x.unsqueeze(1).expand(b_prime, b, n)
        x2 = self.x.unsqueeze(0).expand(b_prime, b, n)
        x = x.unsqueeze(1)
        k = self.kernel(x1, x2)
        dk = self.kernel.grad(x1, x2)
        k_grad = (k @ self.kinv + (self.kinv @ k.t()).t()).unsqueeze(-1)
        grad = (self.kernel.grad(x, x) - k_grad * dk).sum(1)
        return grad

