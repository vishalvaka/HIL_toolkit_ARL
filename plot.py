import numpy as np
import matplotlib.pyplot as plt
import yaml
from botorch.utils.multi_objective.pareto import is_non_dominated
import numpy as np

# def is_non_dominated_simple(scores):
#     is_pareto = np.ones(scores.shape[0], dtype=bool)
#     for i in range(scores.shape[0]):
#         for j in range(scores.shape[0]):
#             if all(scores[j] <= scores[i]) and any(scores[j] < scores[i]):
#                 is_pareto[i] = False
#                 break
#     return is_pareto

# Assuming the file is named 'data.csv' and located in the current working directory
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

# Plotting
plt.figure(figsize=(12, 6))

# Plot y1 vs x
plt.plot(x, y1, 'o-', label='y1')

# Plot y2 vs x on the same graph
plt.plot(x, y2, 's-', label='y2', color='orange')

plt.title('Y1 and Y2 vs X')
plt.xlabel('X')
plt.ylabel('Values')
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