import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

q_value_max = np.load('scripts/visualization/goal1/q_value_max.npy')
q_value_max[q_value_max==-100] = 0

# q_value_max = np.zeros((101, 101))

""" for i in range(9):
    for j in range(9):
        # print(q_values[i, j])
        q_value_max[i, j] = q_values[i, j].max()


numpy.savetxt("foo.csv", q_value_max, delimiter=",") """
""" x = np.arange(0, 10.1 ,0.1)
y = np.arange(0, 10.1, 0.1)
x, y = np.meshgrid(x, y) """

fig = plt.figure()
ax = Axes3D(fig)

ax = plt.subplot(111, projection='3d')
x = np.arange(0, 100, 1)
y = np.arange(0, 100, 1)
x, y = np.meshgrid(x, y)
# ax.scatter(x, y, q_value_max[x, y], cmap='rainbow')
ax.plot_surface(x, y, q_value_max[x, y], rstride=1, cstride=1, cmap='rainbow')
plt.show()