import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def plot_q_value_max_heatmap(data_path):
    q_value_max = np.load('scripts/visualization/' + str(data_path) + '/q_value_max.npy')
    q_value_max[q_value_max==-100] = 0

    sns.set()
    sns.heatmap(q_value_max.T, cmap="rainbow").invert_yaxis()
    plt.show()

def plot_action():
    action1 = np.load('scripts/visualization/goal1/action1.npy')
    action2 = np.load('scripts/visualization/goal1/action2.npy')
    q_values = np.load('scripts/visualization/goal1/q_values.npy')

    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        sns.set()
        sns.heatmap(action1[:, :, i].T, cmap="rainbow").invert_yaxis()

    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        sns.set()
        sns.heatmap(action2[:, :, i].T, cmap="rainbow").invert_yaxis()

    plt.show()

def plot_q_value_max_heatmap_multi():
    """ plot q_value_max heatmap for all models of end point -1 9 """

    plt.figure()
    # load data
    for i in range(8):
        if i == 0:
            data_path = 'scripts/visualization/model-{:d}'.format(800)
        else:
            data_path = 'scripts/visualization/model-{:d}'.format(i * 100)
        q_value_max = np.load(data_path + '/q_value_max.npy')
        plt.subplot(2, 4, i+1)
        sns.set()
        sns.heatmap(q_value_max.T, vmin=0, vmax=12,cmap="rainbow").invert_yaxis()

    plt.show()

if __name__ == "__main__":
    plot_q_value_max_heatmap('model-700_goal_-1_9')
    # plot_action()
    # plot_q_value_max_heatmap_multi()