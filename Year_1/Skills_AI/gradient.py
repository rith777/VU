import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Define the fixed personality vectors
s = np.array([95, 31])
r = np.array([35, 39])
m = np.array([3, 11])


# Step 2: Define the distance function between two vectors
def dist(v, w):
    return np.log(3 + 3 * (v[0] - w[0]) ** 2 + 1.5 * (v[1] - w[1]) ** 2)


# Step 3: Define the overall difference function for a new input vector x
def diff(x):
    return dist(r, x) + dist(s, x) + dist(m, x)


# Step 4: Test that the distance between a vector and itself is log(3)
distance_r_r = dist(r, r)
distance_s_s = dist(s, s)
distance_m_m = dist(m, m)

print("Distance between r and r:", distance_r_r)
print("Distance between s and s:", distance_s_s)
print("Distance between m and m:", distance_m_m)

# Step 5: Test the overall difference function for a new input vector
x = np.array([2.0, 3.0])  # Example new input vector
overall_difference = diff(x)

print("Overall difference for x:", overall_difference)

# Step 6: Compute Diff(x) for 20 random points and compute statistics
random_points = np.random.rand(20, 2) * 100  # Generate 20 random points in the range [0, 100]
diff_values = [diff(point) for point in random_points]

average_diff = np.mean(diff_values)
min_diff = np.min(diff_values)
max_diff = np.max(diff_values)

print("\nAverage Diff(x) for 20 random points:", average_diff)
print("Lowest Diff(x):", min_diff)
print("Highest Diff(x):", max_diff)

# Step 7: Compute Diff(50, 50)
xzero = np.array([50.0, 50.0])
diff_xzero = diff(xzero)
print("\nDiff(50, 50):", diff_xzero)


# Step 8: Create a function to compute the gradient of Diff
def gradient(f, x, delta=0.001):
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_delta = np.copy(x)
        x_delta[i] += delta
        grad[i] = (f(x_delta) - f(x)) / delta
    return grad


# Step 9: Compute the gradient at (0, 0), (100, 0), (0, 100), and (100, 100)
points_to_check = [np.array([0.0, 0.0]), np.array([100.0, 0.0]),
                   np.array([0.0, 100.0]), np.array([100.0, 100.0])]

print("\nGradients at different points:")
for point in points_to_check:
    grad = gradient(diff, point)
    print(f"Gradient at {point}: {grad}")


# Step 10: Create a function to perform gradient descent
def gradient_descent(f, x_init, stepsize=1.0, num_steps=100):
    x = np.copy(x_init)
    for i in range(num_steps):
        grad = gradient(f, x)
        x = x - stepsize * grad
        if (i + 1) % 5 == 0:
            print(f"Step {i + 1}: x = {x}, Diff(x) = {f(x)}")
        if np.linalg.norm(grad) < 1e-6:
            break
    return x


# Step 11: Perform gradient descent from (50, 50)
print("\nGradient descent from (50, 50):")
gradient_descent(diff, np.array([50.0, 50.0]))

# Step 12: Repeat gradient descent from (0, 0) and (100, 100)
print("\nGradient descent from (0, 0):")
gradient_descent(diff, np.array([0.0, 0.0]))

print("\nGradient descent from (100, 100):")
gradient_descent(diff, np.array([100.0, 100.0]))

# Step 13: Create a plot of the Diff function
x_vals = np.linspace(0, 100, 100)
y_vals = np.linspace(0, 100, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = np.array([[diff(np.array([x, y])) for x in x_vals] for y in y_vals])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis
ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')  # Surface plot
ax.set_title('3D Surface Plot of Diff Function')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Diff(x)')
plt.show()

# Step 14: Show the gradient field on top of the contour plot
grad_x = np.zeros_like(X)
grad_y = np.zeros_like(Y)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        grad = gradient(diff, np.array([X[i, j], Y[i, j]]))
        grad_x[i, j] = grad[0]
        grad_y[i, j] = grad[1]

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=50, cmap='viridis')
plt.colorbar(label='Diff(x)')
plt.quiver(X, Y, -grad_x, -grad_y, color='white')
plt.title('Diff Function with Gradient Field')
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()
