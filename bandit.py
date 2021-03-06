import numpy as np
import math
from abc import ABC, abstractmethod
import os, datetime

_base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_base, "data")
detail_dir = os.path.join(_base, "data/detail")

# number of independent run
run = 100
# total time of bandit problem
T = 10000
# features
features = np.mat([[math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)]
                          for i in range(8)])
# known parameter
sigma = 1
action_dimension = 8
feature_dimension = 2

# hyper parameter
lambda_ = 1.0
# alpha_T = sqrt(2logt)
simga0 = 1.0

# bandit environment
class BanditEnv:
    def __init__(self):
        self.theta_star = np.mat([3, 1]).T

    def sample(self, action):
        assert 0 <= action < action_dimension
        feature = features[action]
        reward = np.random.normal(loc=feature*self.theta_star, scale=sigma)
        return reward

# method basic class
class BanditMethod(ABC):
    @abstractmethod
    def take_action(cls):
        raise NotImplementedError

    @abstractmethod
    def update(cls, at, rt):
        raise NotImplementedError


# greedy method class
class Greedy(BanditMethod):
    def __init__(self):
        self.A_inv = np.mat(lambda_ * np.identity(feature_dimension))
        self.b = np.zeros((feature_dimension, 1))
        self.theta_hat = np.zeros((feature_dimension, 1))

    def take_action(self):
        expect = features*self.theta_hat
        return np.random.choice(np.where(expect == expect.max())[0])

    def update(self, action, rt):
        at = features[action].T
        self.A_inv -= (self.A_inv * at * at.T * self.A_inv) / (1 + at.T * self.A_inv * at)
        self.b += at * rt
        self.theta_hat = self.A_inv * self.b


# LinUCB method class
class LinUCB(BanditMethod):
    def __init__(self):
        self.A_inv = np.mat(lambda_ * np.identity(feature_dimension))
        self.b = np.zeros((feature_dimension, 1))
        self.theta_hat = np.zeros((feature_dimension, 1))
        self.time_step = 0

    def take_action(self):
        expect = features*self.theta_hat
        if self.time_step > 0:
            # print(features)
            # print(self.A_inv)
            M = features*self.A_inv
            M2 = np.mat([(M[i]*features[i].T)[0, 0] for i in range(action_dimension)]).T
            # print(M2)
            # print(np.sqrt(M2))
            expect += sigma * math.sqrt(2 * math.log(self.time_step)) * np.sqrt(M2)

        self.time_step += 1
        return np.random.choice(np.where(expect == expect.max())[0])

    def update(self, action, rt):
        at = features[action].T
        self.A_inv -= (self.A_inv * at * at.T * self.A_inv) / (1 + at.T * self.A_inv * at)
        self.b += at * rt
        self.theta_hat = self.A_inv * self.b


# ThompsonSampling method class
class ThompsonSampling(BanditMethod):
    def __init__(self):
        self.sigma0 = simga0
        self.A_inv = np.mat((sigma/self.sigma0)**2 * np.identity(feature_dimension))
        self.b = np.zeros((feature_dimension, 1))
        self.theta_hat = np.mat(np.random.multivariate_normal(mean=[0, 0],
                                                              cov=[[self.sigma0, 0], [0, self.sigma0]])).T

    def take_action(self):
        expect = features*self.theta_hat
        return np.random.choice(np.where(expect == expect.max())[0])

    def update(self, action, rt):
        at = features[action].T
        self.A_inv -= (self.A_inv * at * at.T * self.A_inv) / (1 + at.T * self.A_inv * at)
        self.b += at * rt
        self.theta_hat = np.mat(np.random.multivariate_normal(mean=np.array(self.A_inv*self.b).ravel(),
                                                              cov=sigma*self.A_inv)).T


# solve bandit problem by selected method
def play_bandit(method_name: str):
    env = BanditEnv()
    opt_action = np.argmax(features*env.theta_star)
    regrets_sum = np.zeros(T+1)
    for _ in range(run):
        method = None
        if method_name == "greedy":
            method = Greedy()
        elif method_name == "linucb":
            method = LinUCB()
        elif method_name == "thompson":
            method = ThompsonSampling()
        assert method is not None
        regret = 0.0
        regrets = [0.0]
        for t in range(T):
            action = method.take_action()
            assert 0 <= action < action_dimension
            rt = env.sample(action)
            method.update(action=action, rt=rt)
            regret += ((features[opt_action]-features[action])*env.theta_star)[0, 0]
            regrets.append(regret)
        regrets_sum += regrets

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(detail_dir, exist_ok=True)
    now = datetime.datetime.now()
    np.save(os.path.join(data_dir, f"{method_name}.npy"), regrets_sum / run)
    np.save(os.path.join(detail_dir, f"{method_name}_{now.strftime('%Y%m%d_%H%M%S')}.npy"), regrets_sum / run)


if __name__ == '__main__':
    play_bandit("greedy")
    play_bandit("linucb")
    play_bandit("thompson")
