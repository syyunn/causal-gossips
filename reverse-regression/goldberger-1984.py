import numpy as np
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Setting the random seed for reproducibility
np.random.seed(0)

# Generate X, EY, EZ from Uniform(0, 1)
n_samples = 10000
X = np.random.uniform(0, 1, n_samples)
EY = np.random.uniform(0, 1, n_samples)
EZ = np.random.uniform(0, 1, n_samples)

# Compute Y and Z based on the given equations
Y = 0.5 * X + EY
Z = Y + EZ

# Reshape X, Y, Z for the sklearn LinearRegression model
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
Z = Z.reshape(-1, 1)

# Perform regression to estimate Y_hat
reg = LinearRegression().fit(np.hstack([X, Z]), Y)

# Output the coefficients
print(f'Coefficients: {reg.coef_}')


# Generate values for X, Z and Y_hat for the regression plane
x_plane, z_plane = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 2, 10))
y_hat_plane = reg.coef_[0][0] * x_plane + reg.coef_[0][1] * z_plane

# Create a 3D plot
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Plot the data points
ax.scatter(X, Z, Y, alpha=0.2)

# Plot the regression plane
ax.plot_surface(x_plane, z_plane, y_hat_plane, color='r', alpha=0.5)

ax.set_xlabel('X')
ax.set_ylabel('Z')
ax.set_zlabel('Y')

plt.show()