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
# print(outputs.shape)

def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

def generational_distance(explored_front, true_front):
    distances = []
    for explored_point in explored_front:
        min_distance = float('inf')
        for true_point in true_front:
            distance = euclidean_distance(explored_point, true_point)
            if distance < min_distance:
                min_distance = distance
        distances.append(min_distance)
    gd = np.sqrt(np.sum(np.square(distances)) / len(distances))
    return gd

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

# def true_pareto_front(n_points=10000): #ZDT1
#     f1 = np.linspace(0, 1, n_points)
#     # f2 = 1 - 
#     f2 = 1 - np.sqrt(f1)

#     return np.array([f1, f2])
    # x = np.random.rand(args['Optimization']['n_parms'], n_points)
    # # x = np.array(x)
    
    # # return np.array([x * shift , 1 - np.sqrt(x * shift)])
    # shift = 0.0
    # g = 1 + ((9 - shift) / (len(x) - 1)) * x.T[:, -(args['Optimization']['n_parms']-1):].sum(axis = 1)
    # # print(g.shape)
    # f1 = x[0, :] * (1 - shift)
    # f2 = g * (1 - np.sqrt(x[0, :] / g))

    # return np.array([f1, f2])

def true_pareto_front(n_points=10000): #ZDT2
    # f1 = np.linspace(outputs[:, 0].min(), outputs[:, 0].max(), n_points)
    f1 = np.linspace(0, 1, n_points)
    f2 = 1 - f1**2
    return np.array([f1, f2])

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
#     return np.array([f1, f2])

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
true_values = true_pareto_front()
print(true_values.T.shape)
# pareto_front = []
# for i in range(len(true_f1)):
#     is_dominated = False
#     for j in range(len(true_f1)):
#         if (true_f1[j] <= true_f1[i] and true_f2[j] <= true_f2[i]) and (true_f1[j] < true_f1[i] or true_f2[j] < true_f2[i]):
#             is_dominated = True
#             break
#     if not is_dominated:
#         pareto_front.append([true_f1[i], true_f2[i]])

true_pareto_mask = is_pareto_efficient(true_values.T)
pareto_front = true_values.T[true_pareto_mask]
    
# print(pareto_front)
true_pareto = np.array(pareto_front)
# true_pareto[true_pareto[:, 1].argsort()]
# axes.plot(true_pareto[:, 0], true_pareto[:, 1], label='True Pareto Front', color='blue')
axes.scatter(true_pareto[:, 0], true_pareto[:, 1], label='True Pareto Front', color='blue', s=10, marker='.')
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

generational_distances = []
for i in range(args['Optimization']['n_steps']):
    generational_distances.append(generational_distance(
        pareto_front_sorted[:i+1, :], true_pareto))
    print(f'Generational Distance at iteration {i + 1} = {generational_distances[i]}')

print(f'minimum generational distance = {min(generational_distances)}')