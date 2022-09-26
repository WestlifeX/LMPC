import torch
import torch.nn as nn
from math import pi


class Kernel(nn.Module):

    def __init__(self):
        super(Kernel, self).__init__()

    def __add__(self, other):
        return SumKernel(self, other)

    def __mul__(self, other):
        return ProductKernel(self, other)


class SumKernel(Kernel):

    def __init__(self, k1, k2):
        super(SumKernel, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.num_params = k1.num_params + k2.num_params

    def __call__(self, x1, x2):
        out = self.k1(x1, x2) + self.k2(x1, x2)
        return out

    def grad(self, x1, x2):
        out = self.k1.grad(x1, x2) + self.k2.grad(x1, x2)
        return out

    def reset_params(self):
        self.k1.reset_params()
        self.k2.reset_params()

    def randomise_params(self):
        self.k1.randomise_params()
        self.k2.randomise_params()

    def get_params(self):
        k1_params = self.k1.get_params()
        k2_params = self.k2.get_params()
        params = k1_params + k2_params
        return params

    def set_params(self, p):
        self.k1.set_params = p[:self.k1.num_params]
        self.k2.set_params = p[self.k1.num_params:]


class ProductKernel(Kernel):

    def __init__(self, k1, k2):
        super(ProductKernel, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.num_params = k1.num_params + k2.num_params

    def __call__(self, x1, x2):
        out = self.k1(x1, x2) * self.k2(x1, x2)
        return out

    def grad(self, x1, x2):
        out = self.k1(x1, x2).unsqueeze(-1) * self.k2.grad(x1, x2) + \
              self.k1.grad(x1, x2) * self.k2(x1, x2).unsqueeze(-1)
        return out

    def reset_params(self):
        self.k1.reset_params()
        self.k2.reset_params()

    def randomise_params(self):
        self.k1.randomise_params()
        self.k2.randomise_params()

    def get_params(self):
        k1_params = self.k1.get_params()
        k2_params = self.k2.get_params()
        params = k1_params + k2_params
        return params

    def set_params(self, p):
        self.k1.set_params = p[:self.k1.num_params]
        self.k2.set_params = p[self.k1.num_params:]


class LinearKernel(Kernel):

    def __init__(self, sigma, rho, c):
        super(LinearKernel, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.c = nn.Parameter(torch.tensor(float(c)))
        self.init_sigma = torch.tensor(float(sigma))
        self.init_rho = torch.tensor(float(rho))
        self.init_c = torch.tensor(float(c))
        self.num_params = 3

    def __call__(self, x1, x2):
        z = ((x1 - self.c) * (x2 - self.c)).sum(-1)
        out = self.sigma.pow(2) + self.rho.pow(2) * z
        return out

    def grad(self, x1, x2):
        out = self.rho.pow(2) * (x2 - self.c)
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()
        self.c.data = self.init_c.clone()

    def randomise_params(self):
        self.sigma.data = torch.rand(1) - 0.5
        self.rho.data = torch.rand(1) - 0.5
        self.c.data = torch.rand(1) - 0.5

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone(), self.c.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]
        self.c.data = p[2]


class Matern32Kernel(Kernel):

    def __init__(self, sigma, rho):
        super(Matern32Kernel, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.init_sigma = torch.tensor(float(sigma))
        self.init_rho = torch.tensor(float(rho))
        self.num_params = 2

    def __call__(self, x1, x2):
        d = (x1 - x2).pow(2).sum(-1).pow(0.5)
        out = self.sigma.pow(2) * (1 + 3 ** 0.5 * d / self.rho) * \
              torch.exp(-3 ** 0.5 * d / self.rho)
        return out

    def grad(self, x1, x2):
        diff = x1 - x2
        d = diff.pow(2).sum(-1).pow(0.5).unsqueeze(-1)
        out = -self.sigma.pow(2) * ((3 * diff) / self.rho.pow(2)) * \
              torch.exp(-3 ** 0.5 * d / self.rho)
        out[out != out] = 0.0
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(1)
        self.rho.data = 1 + 4 * torch.rand(1)

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class Matern52Kernel(Kernel):

    def __init__(self, sigma, rho):
        super(Matern52Kernel, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.init_sigma = torch.tensor(float(sigma))
        self.init_rho = torch.tensor(float(rho))
        self.num_params = 2

    def __call__(self, x1, x2):
        d = (x1 - x2).pow(2).sum(-1).pow(0.5)
        out = self.sigma.pow(2) * \
              (1 + 5 ** 0.5 * d / self.rho + (5 * d.pow(2)) / (3 * self.rho.pow(2))) * \
              torch.exp(-5 ** 0.5 * d / self.rho)
        return out

    def grad(self, x1, x2):
        diff = x1 - x2
        d = diff.pow(2).sum(-1).pow(0.5).unsqueeze(-1)
        out = -self.sigma.pow(2) * \
              ((5 * diff) / (3 * self.rho.pow(2)) + (5 * 5 ** 0.5 * diff * d) / (3 * self.rho.pow(3))) * \
              torch.exp(-5 ** 0.5 * d / self.rho)
        out[out != out] = 0.0
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(1)
        self.rho.data = 1 + 4 * torch.rand(1)

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class SquaredExponentialKernel(Kernel):

    def __init__(self, sigma, rho):
        super(SquaredExponentialKernel, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.init_sigma = torch.tensor(float(sigma))
        self.init_rho = torch.tensor(float(rho))
        self.num_params = 2

    def __call__(self, x1, x2):
        d = (x1 - x2).pow(2).sum(-1).pow(0.5)
        out = self.sigma.pow(2) * torch.exp(-d.pow(2) / (2 * self.rho.pow(2)))
        return out

    def grad(self, x1, x2):
        diff = x1 - x2
        d = diff.pow(2).sum(-1).pow(0.5).unsqueeze(-1)
        out = -self.sigma.pow(2) * (diff / (self.rho.pow(2))) * \
              torch.exp(-d.pow(2) / (2 * self.rho.pow(2)))
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(1)
        self.rho.data = 1 + 4 * torch.rand(1)

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class PeriodicKernel(Kernel):

    def __init__(self, sigma, rho, period):
        super(PeriodicKernel, self).__init__()
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.period = nn.Parameter(torch.tensor(float(period)))
        self.init_sigma = torch.tensor(float(sigma))
        self.init_rho = torch.tensor(float(rho))
        self.init_period = torch.tensor(float(period))
        self.num_params = 3

    def __call__(self, x1, x2):
        d = (x1 - x2).pow(2).sum(-1).pow(0.5)
        out = self.sigma.pow(2) * \
              torch.exp(-2 * torch.sin(pi * d / self.period).pow(2) / self.rho.pow(2))
        return out

    def grad(self, x1, x2):
        diff = x1 - x2
        d = diff.pow(2).sum(-1).pow(0.5).unsqueeze(-1)
        inner = pi * d / self.period
        out = -((2 * self.sigma.pow(2) * pi * diff) / (self.period * d * self.rho.pow(2))) * \
              torch.sin(2 * inner) * \
              torch.exp(-2 * torch.sin(inner).pow(2) / self.rho.pow(2))
        out[out != out] = 0.0
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()
        self.period.data = self.init_period.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(1)
        self.rho.data = 1 + 4 * torch.rand(1)
        self.period.data = 0.5 + 4.5 * torch.rand(1)

    def get_params(self):
        params = [self.sigma.data.clone(), self.rho.data.clone(),
                  self.period.data.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]
        self.period.data = p[2]


class AdditiveLinearKernel(Kernel):

    def __init__(self, sigma, rho, c):
        super(AdditiveLinearKernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.c = nn.Parameter(c.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.init_c = c.float().clone()
        self.num_params = 3

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        c = self.c.unsqueeze(0).unsqueeze(0)
        z = ((x1 - c) * (x2 - c))
        out = (sigma.pow(2) + rho.pow(2) * z).sum(-1)
        return out

    def grad(self, x1, x2):
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        c = self.c.unsqueeze(0).unsqueeze(0)
        out = rho.pow(2) * (x2 - c)
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()
        self.c.data = self.init_c.clone()

    def randomise_params(self):
        self.sigma.data = torch.rand(self.sigma.size()) - 0.5
        self.rho.data = torch.rand(self.rho.size()) - 0.5
        self.c.data = torch.rand(self.c.size()) - 0.5

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone(), self.c.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]
        self.c.data = p[2]


class AdditiveMatern32Kernel(Kernel):

    def __init__(self, sigma, rho):
        super(AdditiveMatern32Kernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.num_params = 2

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        diff = torch.abs(x1 - x2)
        out = (sigma.pow(2) * (1 + 3 ** 0.5 * diff / rho) * \
               torch.exp(-3 ** 0.5 * diff * rho)).sum(-1)
        return out

    def grad(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        diff = x1 - x2
        out = -sigma.pow(2) * \
              (3 * torch.sign(diff) * torch.abs(diff) / rho.pow(2)) * \
              torch.exp(-3 * diff / rho)
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(self.sigma.size())
        self.rho.data = 1 + 4 * torch.rand(self.rho.size())

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class AdditiveMatern52Kernel(Kernel):

    def __init__(self, sigma, rho):
        super(AdditiveMatern52Kernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.num_params = 2

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        diff = torch.abs(x1 - x2)
        out = (sigma.pow(2) * \
               (1 + 5 ** 0.5 * diff / rho + (5 * diff.pow(2)) / (3 * rho.pow(2))) * \
               torch.exp(-5 ** 0.5 * diff / rho)).sum(-1)
        return out

    def grad(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        diff = x1 - x2
        out = -sigma.pow(2) * \
              ((5 * diff) / (3 * rho.pow(2)) + \
               (5 * 5 ** 0.5 * torch.sign(diff) * diff.pow(2)) / (3 * rho.pow(3))) * \
              torch.exp(-5 ** 0.5 * torch.abs(diff) / rho)
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(self.sigma.size())
        self.rho.data = 1 + 4 * torch.rand(self.rho.size())

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class AdditiveSquaredExponentialKernel(Kernel):

    def __init__(self, sigma, rho):
        super(AdditiveSquaredExponentialKernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.num_params = 2

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        out = (sigma.pow(2) * \
               torch.exp(-(x1 - x2).pow(2) / (2 * rho.pow(2)))).sum(-1)
        return out

    def grad(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        diff = x1 - x2
        out = -sigma.pow(2) * (diff / rho.pow(2)) * \
              torch.exp(-diff.pow(2) / (2 * rho.pow(2)))
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(self.sigma.size())
        self.rho.data = 1 + 4 * torch.rand(self.rho.size())

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class AdditivePeriodicKernel(Kernel):

    def __init__(self, sigma, rho, period):
        super(AdditivePeriodicKernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.period = nn.Parameter(period.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.init_period = period.float().clone()
        self.num_params = 3

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        period = self.period.unsqueeze(0).unsqueeze(0)
        out = (sigma.pow(2) * \
               torch.exp(-2 * torch.sin(pi * torch.abs(x1 - x2) / period).pow(2) / \
                         rho.pow(2))).sum(-1)
        return out

    def grad(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        period = self.period.unsqueeze(0).unsqueeze(0)
        diff = x1 - x2
        inner = pi * torch.abs(diff) / period
        out = -((2 * sigma.pow(2) * pi * torch.sign(diff)) / (period * rho.pow(2))) * \
              torch.sin(2 * inner) * \
              torch.exp(-2 * torch.sin(inner).pow(2) / rho.pow(2))
        return out

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()
        self.period.data = self.init_period.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(self.sigma.size())
        self.rho.data = 1 + 4 * torch.rand(self.rho.size())
        self.period.data = 0.9 + 0.2 * torch.rand(self.period.size())

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone(), self.period.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]
        self.period.data = p[2]


class MultiplicativeLinearKernel(Kernel):

    def __init__(self, sigma, rho, c):
        super(MultiplicativeLinearKernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.c = nn.Parameter(c.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.init_c = c.float().clone()
        self.num_params = 3

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        c = self.c.unsqueeze(0).unsqueeze(0)
        z = ((x1 - c) * (x2 - c))
        out = (sigma.pow(2) + rho.pow(2) * z).prod(-1)
        return out

    def grad(self, x1, x2):
        raise NotImplementedError()

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()
        self.c.data = self.init_c.clone()

    def randomise_params(self):
        self.sigma.data = torch.rand(self.sigma.size()) - 0.5
        self.rho.data = torch.rand(self.rho.size()) - 0.5
        self.c.data = torch.rand(self.c.size()) - 0.5

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone(), self.c.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]
        self.c.data = p[2]


class MultiplicativeMatern32Kernel(Kernel):

    def __init__(self, sigma, rho):
        super(MultiplicativeMatern32Kernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.num_params = 2

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        diff = torch.abs(x1 - x2)
        out = (sigma.pow(2) * (1 + 3 ** 0.5 * diff / rho) * \
               torch.exp(-3 ** 0.5 * diff * rho)).prod(-1)
        return out

    def grad(self, x1, x2):
        raise NotImplementedError()

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(self.sigma.size())
        self.rho.data = 1 + 4 * torch.rand(self.rho.size())

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class MultiplicativeMatern52Kernel(Kernel):

    def __init__(self, sigma, rho):
        super(MultiplicativeMatern52Kernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.num_params = 2

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        diff = torch.abs(x1 - x2)
        out = (sigma.pow(2) * \
               (1 + 5 ** 0.5 * diff / rho + (5 * diff.pow(2)) / (3 * rho.pow(2))) * \
               torch.exp(-5 ** 0.5 * diff / rho)).prod(-1)
        return out

    def grad(self, x1, x2):
        raise NotImplementedError()

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(self.sigma.size())
        self.rho.data = 1 + 4 * torch.rand(self.rho.size())

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class MultiplicativeSquaredExponentialKernel(Kernel):

    def __init__(self, sigma, rho):
        super(MultiplicativeSquaredExponentialKernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.num_params = 2

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        out = (sigma.pow(2) * \
               torch.exp(-(x1 - x2).pow(2) / (2 * rho.pow(2)))).prod(-1)
        return out

    def grad(self, x1, x2):
        raise NotImplementedError()

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(self.sigma.size())
        self.rho.data = 1 + 4 * torch.rand(self.rho.size())

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]


class MultiplicativePeriodicKernel(Kernel):

    def __init__(self, sigma, rho, period):
        super(MultiplicativePeriodicKernel, self).__init__()
        self.sigma = nn.Parameter(sigma.float())
        self.rho = nn.Parameter(rho.float())
        self.period = nn.Parameter(period.float())
        self.init_sigma = sigma.float().clone()
        self.init_rho = rho.float().clone()
        self.init_period = period.float().clone()
        self.num_params = 3

    def __call__(self, x1, x2):
        sigma = self.sigma.unsqueeze(0).unsqueeze(0)
        rho = self.rho.unsqueeze(0).unsqueeze(0)
        period = self.period.unsqueeze(0).unsqueeze(0)
        out = (sigma.pow(2) * \
               torch.exp(-2 * torch.sin(pi * torch.abs(x1 - x2) / period).pow(2) / \
                         rho.pow(2))).prod(-1)
        return out

    def grad(self, x1, x2):
        raise NotImplementedError()

    def reset_params(self):
        self.sigma.data = self.init_sigma.clone()
        self.rho.data = self.init_rho.clone()
        self.period.data = self.init_period.clone()

    def randomise_params(self):
        self.sigma.data = 1 + 4 * torch.rand(self.sigma.size())
        self.rho.data = 1 + 4 * torch.rand(self.rho.size())
        self.period.data = 0.9 + 0.2 * torch.rand(self.period.size())

    def get_params(self):
        params = [self.sigma.clone(), self.rho.clone(), self.period.clone()]
        return params

    def set_params(self, p):
        self.sigma.data = p[0]
        self.rho.data = p[1]
        self.period.data = p[2]