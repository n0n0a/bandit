from matplotlib import pyplot as plt
import os
import numpy as np

_base = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(_base, "data")

greedy_dir = os.path.join(data_dir, "greedy.npy")
linucb_dir = os.path.join(data_dir, "linucb.npy")
thompson_dir = os.path.join(data_dir, "thompson.npy")

normal_plot_dir = os.path.join(data_dir, "regret_normal.jpg")
log_plot_dir = os.path.join(data_dir, "regret_log.jpg")


def plot_regrets():
    paths = [greedy_dir, linucb_dir, thompson_dir]
    names = ["Greedy", "LinUCB", "ThompsonSampling"]
    fig = plt.figure()
    for idx, path in enumerate(paths):
        if os.path.exists(path):
            y = np.load(path)
            x = list(range(len(y)))
            plt.plot(x, y, label=names[idx])
    plt.xlabel("timestep")
    plt.ylabel("regret")
    fig.legend()
    plt.yscale("log")
    plt.show()
    fig.savefig(log_plot_dir)


if __name__ == '__main__':
    plot_regrets()