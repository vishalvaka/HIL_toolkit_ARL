import numpy as np
import yaml
import pandas as pd
import torch
import matplotlib.pyplot as plt

args = yaml.safe_load(open('configs/test_function.yml','r'))

values = np.linspace(args['Optimization']['range'][0][0], args['Optimization']['range'][1][0], num=100000)

# output1 = values**2
# output2 = (values - 2)**2

# noise1 = np.random.normal(0, np.std(output1))
# noise2 = np.random.normal(0, np.std(output2))

# noise1 = 0.0
# noise2 = 0.0

# print(np.std(output1))

df_xy = pd.read_csv('models/iter_15/data.csv', delim_whitespace=True, header=None)

x = np.array(df_xy.iloc[:, :args['Optimization']['n_parms']].values, dtype = np.float64).squeeze(-1)

y = np.array(df_xy.iloc[:, args['Optimization']['n_parms']:].values, dtype = np.float64).squeeze()
y = -y

# def f(x): #ZDT2 1-D
#     x = np.array(x)

#     return np.array([x, 1 - x ** 2])

# def f(x): #Schaffer N1
#     # print('recieved x: ', x)
#     # x = x[0][0]
#     # print(x)
#     x = np.array(x)
#     # print(x.shape)

#     shift  = 1.0

#     f1 = (x * shift) ** 2
#     f2 = (x * shift - 2) ** 2
#     return np.array([f1, f2])

# def f(x): #Schaffer N2
#     # print('recieved x: ', x)
#     # x = x[0][0]
#     # print(x)
#     x = np.array(x)
#     # print(x.shape)

#     shift  = 1.0

#     f1 = (x * shift)
#     f2 = (x * shift - 2) ** 2
#     return np.array([f1, f2])

# def f(x): #ZDT1 1-D
#     x = np.array(x)
#     shift = 1.0
#     return np.array([x * shift , 1 - np.sqrt(x * shift)])

def f(x): #fronesca and fleming
    # n = len(x)
    n = 1
    shift = 0.0
    f1 = 1 - np.exp(-np.sum((x + shift - 1 / np.sqrt(n)) ** 2))
    f2 = 1 - np.exp(-np.sum((x + shift + 1 / np.sqrt(n)) ** 2))
    return np.array([f1, f2])

# def f(x): #levy and ackley 1d
#     # print('recieved x: ', x)
#     f1 = np.sin(3 * np.pi * x)**2 + (x - 1)**2 * (1 + np.sin(3 * np.pi * x)**2)
#     a = 20
#     b = 0.2
#     c = 2 * np.pi
    
#     # Calculate the Ackley function
#     term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2)))
#     term2 = -np.exp(0.5 * (np.cos(c * x)))
#     f2 = term1 + term2 + a + np.exp(1)
#     return np.array([f1, f2])

# def f(x, noise_level=0):
#     results = []
#     # for x_i in x:
#         # print('x_i', x_i)
#         # x_i[0] = (x_i[0] - args["Optimization"]["range"][0])/(args["Optimization"]["range"][1] - args["Optimization"]["range"][0])   
#     sub_results = [
#         (np.sin(6 *x)**3 * (1 - np.tanh(x ** 2))) + (-1 + torch.rand(1)[0] * 2) * noise_level,
#         .5 - (np.cos(5 * x + 0.7)**3 * (1 - np.tanh(x ** 2))) + (-1 + torch.rand(1)[0] * 2) * noise_level,
#     ]
#     results.append(sub_results)
#     return torch.tensor(results, dtype=torch.float32)

# outputs = f(values)
# print(outputs.shape)
# Y4_clean = np.array([f(xi) for xi in values]).T

# noise1 = np.random.normal(0, np.std(outputs[0]))
# noise2 = np.random.normal(0, np.std(outputs[1]))

# Y4_clean = -Y4_clean

# Define the weights and alpha for the custom scalarization
weights = torch.tensor([0.5, 0.5])
alpha = 0.05

# Define the custom scalarization function with min and sum components
def custom_scalarization(x, weights, alpha=0.05):
    values = -f(x)
    weighted_values = weights * values
    return weighted_values.min(dim=-1).values + alpha * weighted_values.sum(dim=-1)
    # return weighted_values.sum(dim = -1)
custom_scalarized_values = np.array([custom_scalarization(xi, weights, alpha) for xi in values])

fig, ax = plt.subplots(3, 1, figsize=(12, 24))

ax[0].plot(values, custom_scalarized_values, label='Custom Scalarized Function (min + alpha * sum)', color='purple')
ax[0].scatter(x, [custom_scalarization(xi, weights, alpha) for xi in x], c=np.arange(len(df_xy)), cmap='viridis', label='Explored Points', marker='o')
if args['Optimization']['GP'] == 'RGPE':    
    ax[0].set_title('Custom Scalarized Function and Points Explored by RGPE')
else:
    ax[0].set_title('Custom Scalarized Function and Points Explored by GP')
ax[0].set_xlabel('x')
ax[0].set_ylabel('Scalarized Function Value')
ax[0].legend()

# Calculate and plot distances for each iteration
distances = []
for i, xi in enumerate(x):
    max_custom_scalarized_value = np.max(custom_scalarized_values)
    max_custom_scalarization_at_x = custom_scalarization(xi, weights, alpha)
    distance = max_custom_scalarized_value - max_custom_scalarization_at_x
    distances.append(distance.item())

ax[1].plot(range(len(distances)), distances, label='Distance per Iteration', color='red')
ax[1].set_title('Distance Between Max Scalarized Value and Current Iteration')
ax[1].set_xlabel('Iteration')
ax[1].set_ylabel('Distance')
ax[1].legend()

# Plot the components of the function
# outputs = np.array([f(xi) for xi in values]).T
Y4_clean = np.array([f(xi) for xi in values]).T
# print(outputs.shape)
ax[2].plot(values, Y4_clean[0], label='Component 1', color='blue')
ax[2].plot(values, Y4_clean[1], label='Component 2', color='green')
ax[2].fill_between(values, Y4_clean[0] - np.std(Y4_clean[0]), Y4_clean[0] + np.std(Y4_clean[0]), color='blue', alpha=0.2)
ax[2].fill_between(values, Y4_clean[1] - np.std(Y4_clean[1]), Y4_clean[1] + np.std(Y4_clean[1]), color='green', alpha=0.2)
scatter = ax[2].scatter(x, y[:, 0], c=np.arange(len(df_xy)), cmap='viridis', label='Component 1 Points', marker='o')
ax[2].scatter(x, y[:, 1], c=np.arange(len(df_xy)), cmap='viridis', label='Component 2 Points', marker='o')
ax[2].set_title('Components with Points Explored')
ax[2].set_xlabel('x')
ax[2].set_ylabel('Function Value')
ax[2].legend()

plt.colorbar(scatter, ax=ax[2], label='Iteration Count')
plt.tight_layout()
# plt.show()

distances_df = pd.DataFrame(distances)
distances_df.to_csv('models/distances/distances.csv', sep=' ', header=False, index=False)