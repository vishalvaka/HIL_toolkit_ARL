import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import yaml

args = yaml.safe_load(open('configs/test_function.yml','r'))
filename = 'models/iter_' + str(args['Optimization']['n_steps']) + '/data.csv'

# Load the data from the CSV file
data = np.loadtxt(filename, delimiter=' ')
# Assuming 'outputs' and 'batch_number' are already defined as:
outputs = data[:, 1:] * -1  # With 'data' being your input dataset, converted to a numpy array
batch_number = np.arange(outputs.shape[0])

def is_pareto_efficient(costs):
    """
    Identify the Pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :return: A boolean array of Pareto-efficient points
    """
    is_efficient = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(costs[is_efficient] < c, axis=1)
            is_efficient[i] = True
    return is_efficient

# Finding Pareto front
pareto_mask = is_pareto_efficient(outputs)
pareto_front = outputs[pareto_mask]

# Sorting Pareto front points for a coherent line plot
pareto_front_sorted = pareto_front[np.argsort(pareto_front[:, 0])]

# Creating the plot
fig, axes = plt.subplots(1, 1, figsize=(10, 6))
cm = plt.cm.get_cmap('viridis')

# Scatter plot for all points with iteration-based color coding
sc = axes.scatter(outputs[:, 0], outputs[:, 1], c=batch_number, alpha=0.8, cmap=cm)

# Plotting the Pareto front with a line and semi-transparent markers
axes.plot(pareto_front_sorted[:, 0], pareto_front_sorted[:, 1], color='black', label='Pareto Front', linestyle='-')
axes.scatter(pareto_front_sorted[:, 0], pareto_front_sorted[:, 1], edgecolor='black', facecolor='none', alpha=0.5, s=100, label='Pareto Points')

# Adding labels and title
axes.set_xlabel("Function 1 Output")
axes.set_ylabel("Function 2 Output")
plt.title('Data Points with Iteration Colorbar and Pareto Front')

# Setting the axis limits to start from zero
axes.set_xlim([outputs[:, 0].min(), outputs[:, 0].max() + 1])  # Adding a buffer for clarity
axes.set_ylim([outputs[:, 1].min(), outputs[:, 1].max() + 1])

# Setting up the colorbar
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
cbar.ax.set_title("Iteration")

# Adding the legend and displaying the plot
axes.legend()
plt.show()