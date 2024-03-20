import numpy as np
import matplotlib.pyplot as plt
import yaml
from botorch.utils.multi_objective.pareto import is_non_dominated
import numpy as np
import torch
from scripts.test_cost_function import f


# def is_non_dominated_simple(scores):
#     is_pareto = np.ones(scores.shape[0], dtype=bool)
#     for i in range(scores.shape[0]):
#         for j in range(scores.shape[0]):
#             if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
#                 is_pareto[i] = False
#                 break
#     return is_pareto


args = yaml.safe_load(open('configs/test_function.yml','r'))
filename = 'models/iter_' + str(args['Optimization']['n_steps']) + '/data.csv'

# Load the data from the CSV file
data = np.loadtxt(filename, delimiter=' ')
data[:, 1] = -data[:, 1]
data[:, 2] = -data[:, 2]
data = data[np.argsort(data[:, 0])]
# print(data)
# Splitting the data into X, Y1, and Y2
x = data[:, 0]
y1 = data[:, 1]
y2 = data[:, 2]

x = np.array(x)
# x_norm = (x - x.min()) / (x.max() - x.min(    ))

# The original function
bounds = torch.tensor(args["Optimization"]["range"])
xs = np.linspace(bounds[0][0], bounds[1][0], 1000)
xs = xs.reshape((1000, -1))
ys_true = f(xs, noise_level=0).numpy()

# # Plotting
# plt.figure(figsize=(12, 6))

plt.plot(xs, ys_true, label=["F1", "F2"])
# the fill method constructs a polygon of the specified color delimited by all the point
# in the xs and ys arrays.
# plt.fill(np.concatenate([xs, xs[::-1]]),
#          np.concatenate(([fx_i - 1 * noise_level for fx_i in ys_true],
#                          [fx_i + 1 * noise_level for fx_i in ys_true[::-1]])),
#          alpha=.2, ec="None")
# plt.legend()
# plt.xlabel('x')
# plt.ylabel('Objectives')
# plt.grid()
# plt.show()

# Plot y1 vs x
plt.plot(x, y1, 'o', label='y1', color = "blue")

# Plot y2 vs x on the same graph
plt.plot(x, y2, 's', label='y2', color='orange')

plt.title('F1 and F2 vs parameter (x)')
plt.xlabel('Parameter (x)')
plt.ylabel('Objective Values')
plt.grid(True)
plt.legend()

plt.show()

# Combine y1 and y2 into a single matrix for processing
# objectives = np.vstack((y1, y2)).T

# # Re-running the non-dominance check with the correct data
# is_pareto = is_non_dominated_simple(objectives)

# # Extracting non-dominated (Pareto front) points
# pareto_front = objectives[is_pareto]

# # Re-plotting the results
# plt.figure(figsize=(10, 6))

# # Plot all points
# plt.scatter(y1, y2, color='lightgray', label='Dominated Points')

# # Plot Pareto front
# plt.scatter(pareto_front[:, 0], pareto_front[:, 1], color='red', label='Pareto Front')

# plt.title('Pareto Front for Y1 vs Y2')
# plt.xlabel('Y1')
# plt.ylabel('Y2')
# plt.ylim(-5, 25)
# plt.xlim(-5, 25)
# plt.legend()
# plt.grid(True)
# plt.show()