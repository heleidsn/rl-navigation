import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

def plot_q_value_max_heatmap(data_path):
    q_value_max = np.load('scripts/visualization/' + str(data_path) + '/q_value_max.npy')
    # q_value_max[q_value_max==-100] = 0
    np.savetxt('scripts/visualization/' + str(data_path) + '/q_value_max.csv', q_value_max, delimiter=',', fmt='%2.1f')
    sns.set()
    sns.heatmap(q_value_max.T, vmin=0, cmap="rainbow", xticklabels=10, yticklabels=10).invert_yaxis()
    plt.show()

def plot_action(data_path):
    data_path_str = 'scripts/visualization/' + str(data_path)
    action1 = np.load(data_path_str + '/action1.npy')
    action2 = np.load(data_path_str + '/action2.npy')

    fig_speed = plt.figure(figsize=(18, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1, title='yaw={:d}(degrees)'.format(36 * i))
        sns.set()
        if i < 5:
            if i == 0:
                sns.heatmap(action1[:, :, i].T, cmap="rainbow", xticklabels=False, yticklabels=10).invert_yaxis()
            else:
                sns.heatmap(action1[:, :, i].T, cmap="rainbow", xticklabels=False, yticklabels=False).invert_yaxis()
        else:
            if i == 5:
                sns.heatmap(action1[:, :, i].T, cmap="rainbow", xticklabels=10, yticklabels=10).invert_yaxis()
            else:
                sns.heatmap(action1[:, :, i].T, cmap="rainbow", xticklabels=10, yticklabels=False).invert_yaxis()
    fig_speed.suptitle('Speed command for each state')

    fig_rotate = plt.figure(figsize=(18, 6))
    for i in range(10):
        plt.subplot(2, 5, i+1, title='yaw={:d}(degrees)'.format(36 * i))
        sns.set()
        if i < 5:
            if i == 0:
                sns.heatmap(action2[:, :, i].T, cmap="rainbow", xticklabels=False, yticklabels=10).invert_yaxis()
            else:
                sns.heatmap(action2[:, :, i].T, cmap="rainbow", xticklabels=False, yticklabels=False).invert_yaxis()
        else:
            if i == 5:
                sns.heatmap(action2[:, :, i].T, cmap="rainbow", xticklabels=10, yticklabels=10).invert_yaxis()
            else:
                sns.heatmap(action2[:, :, i].T, cmap="rainbow", xticklabels=10, yticklabels=False).invert_yaxis()

    fig_rotate.suptitle('Angular velocity command for each state')
    plt.show()

def plot_q_value_max_heatmap_multi():
    """ plot q_value_max heatmap for all models of end point -1 9 """

    plt.figure(figsize=(18, 7))
    # load data
    for i in range(8):
        if i == 0:
            data_path = 'scripts/visualization/model-{:d}'.format(i)
        else:
            data_path = 'scripts/visualization/model-{:d}'.format(i * 100)
        q_value_max = np.load(data_path + '/q_value_max.npy')

        plt.subplot(2, 4, i+1, title='model-{:d}'.format(i*100))
        sns.set()
        sns.heatmap(q_value_max.T, vmin=0, cmap="rainbow", xticklabels=5, yticklabels=5).invert_yaxis()

    plt.show()

if __name__ == "__main__":
    # plot_q_value_max_heatmap('model-700-detail')
    plot_action('model-700_-5_5')
    # plot_q_value_max_heatmap_multi()