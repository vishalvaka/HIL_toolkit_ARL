import numpy as np
import yaml
import pandas as pd
import torch
import matplotlib.pyplot as plt

args = yaml.safe_load(open('configs/test_function.yml','r'))

values = np.linspace(args['Optimization']['range'][0][0], args['Optimization']['range'][1][0], num=100000)

output1 = values**2
output2 = (values - 4)**2

# noise1 = np.random.normal(0, np.std(output1))
# noise2 = np.random.normal(0, np.std(output2))

noise1 = 0.0
noise2 = 0.0

print(np.std(output1))

df_xy = pd.read_csv('models/iter_15/data.csv', delim_whitespace=True, header=None)

x = torch.tensor(df_xy.iloc[:, :args['Optimization']['n_parms']].values, dtype = torch.float64).squeeze(-1)

y = torch.tensor(df_xy.iloc[:, args['Optimization']['n_parms']:].values, dtype = torch.float64).squeeze()
y = -y

def f(x): #ZDT2 1-D
    x = np.array(x)

    return np.array([x, 1 - x ** 2])

# def f(x): #ZDT1 1-D
#     x = np.array(x)

#     return np.array([x, 1 - np.sqrt(x)])

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


Y4_clean = np.array([f(xi) for xi in values]).T

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

fig, ax = plt.subplots(2, 1, figsize=(12, 16))

ax[0].plot(values, Y4_clean[0], label='Component 1', color='blue')
ax[0].plot(values, Y4_clean[1], label='Component 2', color='green')
# ax[0].fill_between(values, Y4_clean[0] - np.std(output1), Y4_clean[0] + np.std(output1), color='blue', alpha=0.2)
# ax[0].fill_between(values, Y4_clean[1] - np.std(output2), Y4_clean[1] + np.std(output2), color='green', alpha=0.2)
scatter = ax[0].scatter(x, y[:, 0], c=np.arange(len(df_xy)), cmap='viridis', label='Component 1 Points', marker='o')
ax[0].scatter(x, y[:, 1], c=np.arange(len(df_xy)), cmap='viridis', label='Component 2 Points', marker='o')
ax[0].set_title('Day 2: Function 2 Components with Noise as Shaded Region and Points Explored')
ax[0].set_xlabel('x')
ax[0].set_ylabel('Function Value')
ax[0].legend()

# Plot the custom scalarized function with noise as shaded region
ax[1].plot(values, custom_scalarized_values, label='Custom Scalarized Function (min + alpha * sum)', color='purple')
# ax[1].fill_between(x, custom_scalarized_values - 0.1, custom_scalarized_values + 0.1, color='purple', alpha=0.2)
scatter = ax[1].scatter(x, [custom_scalarization(xi, weights, alpha) for xi in x], c=np.arange(len(df_xy)), cmap='viridis', label='Explored Points', marker='o')
ax[1].set_title('Custom Scalarized Function (min + alpha * sum, Weights 0.5, 0.5, Alpha 0.05) and Points Explored by GP')
ax[1].set_xlabel('x')
ax[1].set_ylabel('Scalarized Function Value')
ax[1].legend()

# plt.colorbar(scatter, ax=ax[1], label='Iteration Count')
plt.tight_layout()
plt.show()