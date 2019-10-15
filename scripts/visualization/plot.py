import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

q_values = np.load('scripts/visualization/q_values.npy')

q_value_max = np.zeros((9, 9))

for i in range(9):
    for j in range(9):
        # print(q_values[i, j])
        q_value_max[i, j] = q_values[i, j].max()

numpy.savetxt("foo.csv", q_value_max, delimiter=",")
x = np.arange(0, 9 ,1)
y = np.arange(0, 9 ,1)
x, y = np.meshgrid(x, y)

fig = plt.figure()
ax = Axes3D(fig)

ax = plt.subplot(111, projection='3d')
# ax.scatter(x, y, q_value_max[x, y], c = 'r')
ax.plot_surface(x, y, q_value_max, rstride=1, cstride=1, cmap='rainbow')
plt.show()