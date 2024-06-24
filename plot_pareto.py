import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import yaml

args = yaml.safe_load(open('configs/test_function.yml','r'))
filename = 'models/iter_' + str(args['Optimization']['n_steps']) + '/data.csv'

# Load the data from the CSV file
data = np.loadtxt(filename, delimiter=' ')
# Assuming 'outputs' and 'batch_number' are already defined as:
outputs = data[:, args['Optimization']['n_parms']:] * -1  # With 'data' being your input dataset, converted to a numpy array
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

# def true_pareto_front(n_points=100): #ZDT1
#     # f1 = np.linspace(outputs[:, 0].min(), outputs[:, 0].max(), n_points)
#     f1 = np.linspace(0, 1, n_points)
#     f2 = 1 - np.sqrt(f1)
#     return f1, f2

def true_pareto_front(n_points=100): #ZDT2
    # f1 = np.linspace(outputs[:, 0].min(), outputs[:, 0].max(), n_points)
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - f1**2
    return f1, f2

# def true_pareto_front(n_points=100): #Schaffer
#     x = np.linspace(0, 2, n_points)
#     f1 = x**2
#     f2 = (x - 2)**2
#     return f1, f2

# def true_pareto_front(n_points=100): #ZDT2 1-D
#     x = np.linspace(0, 1, n_points)
#     f1 = x
#     f2 = 1 - x ** 2
#     return f1, f2

# def true_pareto_front(n_points=100): #ZDT1 1-D
#     x = np.linspace(0, 1, n_points)
#     f1 = x
#     f2 = 1 - np.sqrt(x)
#     return f1, f2

# def true_pareto_front(n_points=100):
#     x = np.linspace(-5, 5, n_points)
#     print(x)
#     f1 = 1 - np.exp(-(x - 1/np.sqrt(2))**2)
#     f2 = 1 - np.exp(-(x + 1/np.sqrt(2))**2)
#     return f1, f2


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
true_f1, true_f2 = true_pareto_front()

print(true_f1, true_f2)
axes.plot(true_f1, true_f2, label='True Pareto Front', color='blue')
# Adding labels and title
axes.set_xlabel("Objective 1")
axes.set_ylabel("Objective 2")
plt.title('Data Points with Iteration Colorbar and Pareto Front')

# Setting the axis limits to start from zero
# axes.set_xlim([outputs[:, 0].min() - 0.2, outputs[:, 0].max() + 1])  # Adding a buffer for clarity
# axes.set_ylim([outputs[:, 1].min() - 0.2, outputs[:, 1].max() + 1])

# Setting up the colorbar
norm = plt.Normalize(batch_number.min(), batch_number.max())
sm = ScalarMappable(norm=norm, cmap=cm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=axes, orientation='vertical', fraction=0.046, pad=0.04)
cbar.ax.set_title("Iteration")

# Adding the legend and displaying the plot
axes.legend()
plt.show()