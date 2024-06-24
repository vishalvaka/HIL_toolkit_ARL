import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors

# Load the CSV file
data = np.loadtxt('models/iter_15/data.csv')

# Extract columns
x1_values = data[:, 0]
x2_values = data[:, 1]
neg_y1_values = data[:, 2]
neg_y2_values = data[:, 3]

# Convert negative values back to positive for y1 and y2
y1_values = -neg_y1_values
y2_values = -neg_y2_values

# Determine the most optimal explored point based on a simple sum of y1 and y2 (assuming equal importance)
weighted_y1 = 0.5 * y1_values
weighted_y2 = 0.5 * y2_values
optimal_index = np.argmin(np.minimum(weighted_y1, weighted_y2) + 0.05 * np.add(weighted_y1, weighted_y2))
optimal_x1 = x1_values[optimal_index]
optimal_x2 = x2_values[optimal_index]
optimal_y1 = y1_values[optimal_index]
optimal_y2 = y2_values[optimal_index]

# Fonesca and Fleming function for plotting
def f(x):
    n = len(x)
    f1 = 1 - np.exp(-np.sum((x - 1 / np.sqrt(n)) ** 2))
    f2 = 1 - np.exp(-np.sum((x + 1 / np.sqrt(n)) ** 2))
    return np.array([f1, f2])

# Generate a grid of points for plotting the Fonesca and Fleming function
num_points = 50
x1 = np.linspace(0, 1, num_points)
x2 = np.linspace(0, 1, num_points)
X1, X2 = np.meshgrid(x1, x2)
print(x1.size)
# Evaluate the Fonesca and Fleming function on the grid
F1 = np.zeros_like(X1)
F2 = np.zeros_like(X2)

for i in range(num_points):
    for j in range(num_points):
        f1, f2 = f([X1[i, j], X2[i, j]])
        F1[i, j] = f1
        F2[i, j] = f2

weighted_y1 = 0.5 * F1
weighted_y2 = 0.5 * F2
optimal_index = np.argmin(np.add(np.minimum(weighted_y1, weighted_y2), 0.05 * np.add(weighted_y1, weighted_y2)))
true_optimal_x1 = X1[optimal_index]
true_optimal_x2 = X2[optimal_index]
true_optimal_y1 = F1[optimal_index]
true_optimal_y2 = F2[optimal_index]

# True optimal points for the Fonesca and Fleming function in the objective space
# true_optimal_x = np.linspace(0.0, 1.0, num_points)
# true_optimal_y1 = 1 - np.exp(-np.sum((true_optimal_x - 1 / np.sqrt(2)) ** 2))
# true_optimal_y2 = 1 - np.exp(-np.sum((true_optimal_x + 1 / np.sqrt(2)) ** 2))

# Plotting
fig = plt.figure(figsize=(16, 8))

# 3D plot of the Fonesca and Fleming function
ax1 = fig.add_subplot(121, projection='3d')

# Plot the Fonesca and Fleming function surfaces
surface1 = ax1.plot_surface(X1, X2, F1, cmap='viridis', edgecolor='k', alpha=0.5)
surface2 = ax1.plot_surface(X1, X2, F2, cmap='plasma', edgecolor='k', alpha=0.5)

# Scatter the explored points with color-coding by row number in the 3D plot
num_rows = len(x1_values)
colors = plt.cm.jet(np.linspace(0, 1, num_rows))

for idx in range(num_rows):
    ax1.scatter(x1_values[idx], x2_values[idx], y1_values[idx], color=colors[idx], s=20)
    ax1.scatter(x1_values[idx], x2_values[idx], y2_values[idx], color=colors[idx], s=20)

# Highlight the most optimal explored point
ax1.scatter(optimal_x1, optimal_x2, optimal_y1, color='g', s=100, marker='*', label='Optimal Explored Point (f1)')
ax1.scatter(optimal_x1, optimal_x2, optimal_y2, color='y', s=100, marker='*', label='Optimal Explored Point (f2)')

# Highlight the true optimal point
ax1.scatter(true_optimal_x1, true_optimal_x2, true_optimal_y1, color='r', s=100, marker='^', label='True Optimal Point (f1)')
ax1.scatter(true_optimal_x1, true_optimal_x2, true_optimal_y2, color='r', s=100, marker='^', label='True Optimal Point (f2)')

# Annotate the most optimal point with its row number
ax1.text(optimal_x1, optimal_x2, optimal_y1, f'{optimal_index}', color='g')
ax1.text(optimal_x1, optimal_x2, optimal_y2, f'{optimal_index}', color='y')

# Customizing the 3D plot
ax1.set_title('Fonesca and Fleming Function with Explored Points')
ax1.set_xlabel('x1')
ax1.set_ylabel('x2')
ax1.set_zlabel('Objective Values')

# Adding a legend to the 3D plot
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='g', marker='*', linestyle='', markersize=15, label=f'Optimal Explored Point (f1) at iter {optimal_index}'),
    Line2D([0], [0], color='y', marker='*', linestyle='', markersize=15, label=f'Optimal Explored Point (f2) at iter {optimal_index}'),
    Line2D([0], [0], color='r', marker='^', linestyle='', markersize=15, label='True Optimal Point (f1, f2)'),
    Line2D([0], [0], color='b', marker='o', linestyle='', markersize=10, label='Explored Points')
]

# Add row number color mapping to legend
for idx in range(num_rows):
    legend_elements.append(Line2D([0], [0], color=colors[idx], marker='o', linestyle='', markersize=10, label=f'Row {idx}'))

ax1.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

# 2D scatter plot of the explored points on the parameter space
ax2 = fig.add_subplot(122)
ax2.scatter(x1_values, x2_values, c=colors, s=50, edgecolor='k')
ax2.scatter(optimal_x1, optimal_x2, color='r', s=100, marker='*', label=f'Optimal Explored Point at iter {optimal_index}')
ax2.set_title('Explored Points in Parameter Space')
ax2.set_xlabel('x1')
ax2.set_ylabel('x2')

# Adding a legend to the 2D plot
ax2.legend(loc='upper left')

plt.show()

print("Optimal Explored Point (x1, x2):", (optimal_x1, optimal_x2))
print("Objective y1 at Optimal Explored Point:", optimal_y1)
print("Objective y2 at Optimal Explored Point:", optimal_y2)
