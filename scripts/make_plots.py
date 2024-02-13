from HIL.optimization.BO import BayesianOptimization
import numpy as np
import torch
import matplotlib.pyplot as plt

BO = BayesianOptimization(range = np.array([0, 1]), noise_range=np.array([0.005, 10]), plot=False)
length_scale = []
variance = []
noise_data = []

for i in range(3, 16):
    data = np.loadtxt(r'C:\Users\sruth\OneDrive - University of Illinois at Chicago\Documents\Sruthi\HIL_toolkit-main\HIL_toolkit-main\models\iter_15\data.csv')
    # split the data to x and y
    data = data.reshape(-1, 2)
    x = data[:, 0].reshape(-1, 1)
    y = data[:, 1].reshape(-1, 1)
    x_new = x[:i].reshape(-1,1)
    y_new = y[:i].reshape(-1,1)
    BO.COMMS = False
    new_parameter = BO.run(x_new, y_new)

    with torch.no_grad():
        x_length = np.linspace(BO.range[0],BO.range[1],100).reshape(-1, 1)
        observed = BO.likelihood(BO.model(torch.tensor(x_length))) #type: ignore
        observed_mean = observed.mean.cpu().numpy() #type: ignore
        upper, lower = observed.confidence_region() #type: ignore

        std_dev = observed.stddev.cpu().numpy()
        scaling_factor = 1.96 
        upper = observed_mean + scaling_factor * std_dev
        lower = observed_mean - scaling_factor * std_dev

# Plot the GP
plt.figure(figsize=(8, 6))
plt.plot(x_length, observed_mean, label='mean', linewidth=3, color='b')
plt.scatter(x_new, y_new, label='Points', color='r', marker='o', s=50)
plt.fill_between(x_length.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.3)
plt.xlim(BO.range[0], BO.range[1])
plt.xticks(np.arange(BO.range[0], BO.range[1], 0.2), fontsize=13)
plt.yticks(fontsize=14)
plt.ylim(y_new.min()-1, y_new.max()+1)
plt.xlabel('Normalized parameter', fontsize=14)
plt.ylabel('Normalized ECG-RMSSD cost', fontsize=14)
plt.show()

# # Plot the stiffness param vs. iteration
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