import numpy as np
import yaml
import pandas as pd
import torch
import matplotlib.pyplot as plt
import os
import re

args = yaml.safe_load(open('configs/test_function.yml','r'))

test_x = np.linspace(args['Optimization']['range'][0][0], args['Optimization']['range'][1][0], num=10000)

# output1 = values**2
# output2 = (values - 2)**2

# noise1 = np.random.normal(0, np.std(output1))
# noise2 = np.random.normal(0, np.std(output2))

# noise1 = 0.0
# noise2 = 0.0

# print(np.std(output1))

df_xy = pd.read_csv('models/iter_15/data.csv', delim_whitespace=True, header=None)

x = np.array(df_xy.iloc[:, :args['Optimization']['n_parms']].values, dtype = np.float64).squeeze(-1)

y = np.array(df_xy.iloc[:, args['Optimization']['n_parms']:].values, dtype = np.float64).squeeze() # y has the negated values


def f(x): # Alpine function
    x = np.array(x)

    shift  = (np.pi*0)/12.0

    #f1 = x * np.sin(x + np.pi + shift) + x / 10.0
    f1 = (x - 2)**2
    return f1

# Calculate function values for the given parameter range
function_values = np.array([-f(xi) for xi in test_x])
max_function_value = np.max(function_values)
print("Max value:", np.max(function_values))
print("Min value:", np.min(function_values))
print("Max function value:",max_function_value)

# Plot
fig, ax = plt.subplots(2, 1, figsize=(12, 24))

# Calculate and plot distances for each iteration
distances = []
distances = np.array(distances)
for i, xi in enumerate(x):
    function_value_at_x = -f(xi)
    distance = max_function_value - function_value_at_x
    print(function_value_at_x)
    if distances.size == 0 or distance < np.min(distances):
        distances = np.append(distances, distance)
    else:
        smallest_value = np.min(distances)
        distances = np.append(distances, smallest_value)
ax[0].plot(range(len(distances)), distances, label='Distance per Iteration', color='red')
ax[0].set_title('Distance Between Max Function Value and Current Iteration')
ax[0].set_xlabel('Iteration')
ax[0].set_ylabel('Distance')
ax[0].legend()

# Plot the function
Y4_clean = np.array([-f(xi) for xi in test_x])
ax[1].plot(test_x, Y4_clean, label='Function', color='blue')
ax[1].fill_between(test_x, Y4_clean - np.std(Y4_clean), Y4_clean + np.std(Y4_clean), color='blue', alpha=0.2)
scatter = ax[1].scatter(x, y, c=np.arange(len(df_xy)), cmap='viridis', label='Points', marker='o')
ax[1].set_title('Function with Points Explored')
ax[1].set_xlabel('x')
ax[1].set_ylabel('Function Value')
ax[1].legend()

plt.colorbar(scatter, ax=ax[1], label='Iteration Count')
plt.tight_layout()
plt.show()