import argparse

def get_args():
    parser = argparse.ArgumentParser("bayesian optimization for learning mpc")
    parser.add_argument('--seed', type=int, default=3, help='random seed')
    parser.add_argument('--totalIterations', type=int, default=50, help='total number of iterations')
    args = parser.parse_args()
    return args

