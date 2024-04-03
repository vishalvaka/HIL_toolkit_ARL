from HIL.optimization.BO import BayesianOptimization
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

# 90, 130
BO = BayesianOptimization(range = np.array([90,130]), noise_range=np.array([0.005, 2]), plot=False) # Change the range depending on whether the parameter range is normalized or not
length_scale = []
variance = []
noise_data = []


# Load data

# data = pd.read_excel('models\test\Feb_29_ETC.xlsx') 
# Fit GP for entire dataset
df_xy = pd.read_csv("models/test/Feb_29_ETC.csv", delim_whitespace=True)
# df_xy = pd.read_csv("./function_data.csv")

# Extract the 'x' column and convert to a 1D Numpy array
# x = df_xy['x_normalized'].values

# Since the provided CSV file only contains 2 columns ('x' and 'y'),
# and based on your description, if you expect to extract 'y' as a separate column:
# y = df_xy['y_normalized'].values.reshape(-1, 1) 

# Extract the first column as x and convert to a 1D Numpy array
x = df_xy.iloc[:, 0].values

# # Extract the second and third columns as a vector of two objectives (y1 and y2), and convert to a 2D Numpy array
y = torch.tensor(df_xy.iloc[:, 1:].values, dtype = torch.float64)
#y = (y-np.mean(y))/(np.std(y))
print(y.shape)
x = x.reshape(-1, 1)

BO.COMMS = False
new_parameter = BO.run(x, y)

with torch.no_grad():
    x_length = np.linspace(BO.range[0],BO.range[1],100).reshape(-1, 1)    
    observed = BO.likelihood(BO.model(torch.tensor(x_length))) #type: ignore
    observed_mean = observed.mean.cpu().numpy() #type: ignore
    upper, lower = observed.confidence_region() #type: ignore

    std_dev = observed.stddev.cpu().numpy()
    scaling_factor = 1.96 
    upper = observed_mean + scaling_factor * std_dev
    lower = observed_mean - scaling_factor * std_dev

# # To fit GP iteratively 
# for i in range(3, 16):
#     data = np.loadtxt(r'models\iter_15\data.csv')
#     # split the data to x and y
#     data = data.reshape(-1, 2)
#     x = data[:, 0].reshape(-1, 1)
#     y = data[:, 1].reshape(-1, 1)
#     x_new = x[:i].reshape(-1,1)
#     y_new = y[:i].reshape(-1,1)
#     BO.COMMS = False
#     new_parameter = BO.run(x_new, y_new)

#     with torch.no_grad():
#         x_length = np.linspace(BO.range[0],BO.range[1],100).reshape(-1, 1)
#         observed = BO.likelihood(BO.model(torch.tensor(x_length))) #type: ignore
#         observed_mean = observed.mean.cpu().numpy() #type: ignore
#         upper, lower = observed.confidence_region() #type: ignore

#         std_dev = observed.stddev.cpu().numpy()
#         scaling_factor = 1.96 
#         upper = observed_mean + scaling_factor * std_dev
#         lower = observed_mean - scaling_factor * std_dev

x_new=x
y_new=y

# max_val = max(observed_mean)
# index = np.where(observed_mean == max_val)[0]
# opt_param = x_length[index]

# print("The optimal parameter is", opt_param)

# Plot the GP
# plt.figure(figsize=(8, 6))
# plt.plot(x_length, observed_mean, label='mean', linewidth=3, color='b')
# plt.scatter(x_new, y_new, label='Points', color='r', marker='o', s=50)
# plt.fill_between(x_length.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.3)
# plt.xlim(BO.range[0], BO.range[1])
# # plt.xticks(np.arange(BO.range[0], BO.range[1], 0.2), fontsize=13)
# plt.yticks(fontsize=14)
# plt.ylim(y_new.min()-1, y_new.max()+1)
# plt.xlabel('Step frequency', fontsize=14)
# plt.ylabel('ETC', fontsize=14)
# # plt.axvline(x = 97, color = 'g', label = 'preferred step frequency')
# plt.legend()
# plt.show()

# index_max = np.argmax(observed_mean)

# # Use the index to find the corresponding x value
# x_max = x_length[index_max]

# print("The x value at which the observed mean is maximized:", x_max)

# # # Plot the stiffness param vs. iteration
# plt.figure(figsize=(8, 6))
# plt.plot(np.arange(3, 8), x_new, linewidth=3, color='b')
# plt.scatter(np.arange(3, 8), x_new, linewidth=3, color='b')
# plt.xticks(np.arange(3, 8), fontsize=13)
# plt.yticks(fontsize=14)
# plt.xlabel('Iteration', fontsize=14)
# plt.ylabel('Stiffness parameter', fontsize=14)

# plt.show()

# def _plot(self) -> None:
#     plt.cla()
#     x = self.x.detach().numpy()
#     y = self.y.detach().numpy()
#     plt.plot(x, y, 'r.', ms = 10)
#     x_torch = torch.tensor(x).to(self.device)
#     y_torch = torch.tensor(y).to(self.device)
#     self.model.eval()  #type: ignore
#     self.likelihood.eval()
#     with torch.no_grad():
#         x_length = np.linspace(self.range[0,0],self.range[1,0],100).reshape(-1, 1)
#         # print(x_length,self.range)
#         observed = self.likelihood(self.model(torch.tensor(x_length))) #type: ignore
#         observed_mean = observed.mean.cpu().numpy() #type: ignore
#         upper, lower = observed.confidence_region() #type: ignore
#         # x_length = x_length.cpu().numpy()
        
#     plt.plot(x_length.flatten(), observed_mean)
#     plt.fill_between(x_length.flatten(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.2)
#     plt.legend(['Observed Data', 'mean', 'Confidence'])
#     plt.pause(0.01)

plt.figure(figsize=(8, 6))
plt.plot(x_length, observed_mean, label='mean', linewidth=3, color='b')
plt.scatter(x_new, y_new, label='Points', color='r', marker='o', s=50)
plt.fill_between(x_length.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.3)

# Find the index of the maximum value in observed_mean
index_max = np.argmax(observed_mean)
# Use the index to find the corresponding x value
x_max = x_length[index_max]
# Plot the maximum observed mean point as a star
plt.plot(x_max, observed_mean[index_max], 'y*', markersize=15, label='Max Mean')

plt.xlim(BO.range[0], BO.range[1])
plt.yticks(fontsize=14)
plt.ylim(y_new.min()-1, y_new.max()+1)
plt.xlabel('Step frequency', fontsize=14)
plt.ylabel('ETC', fontsize=14)

# Annotate the plot with the x value at the maximum observed mean
plt.annotate(f'{x_max[0]:.2f}', # text
             (x_max, observed_mean[index_max]), # point to annotate
             textcoords="offset points", # how to position the text
             xytext=(0,10), # distance from text to points (x,y)
             ha='center', # horizontal alignment can be left, right or center
             fontsize=12, 
             color='blue')

plt.legend()
plt.show()
