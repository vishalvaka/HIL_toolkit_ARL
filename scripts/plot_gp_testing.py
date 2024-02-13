import math
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ProbabilityOfImprovement
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from botorch.sampling import IIDNormalSampler
from botorch.optim import optimize_acqf
from sklearn.preprocessing import StandardScaler
from HIL.optimization.kernel import SE, Matern 
from botorch.models.converter import batched_to_model_list, model_list_to_batched


# # Multi-objective optimization - plotting the Gaussian Process models (final models)

# Load x and y
# os.chdir(r"./models/iter_15")
# os.chdir(r"./models/test")
# Import the x and y data
df_xy = pd.read_csv("testing_data/step_freq_data.csv", delim_whitespace=True, header=None)

# Extract the first column as x and convert to a 1D Numpy array
x = df_xy.iloc[:, 0].values

# Extract the second and third columns as a vector of two objectives (y1 and y2), and convert to a 2D Numpy array
y = df_xy.iloc[:, 1:].values

# Convert x and y numpy arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
x_tensor = x_tensor.view(x_tensor.shape[0], 1)
y_tensor = torch.tensor(y, dtype=torch.float32)
print("Shape of x tensor", x_tensor.shape)
print("Shape of y tensor", y_tensor.shape)


# function to initialize GP models
def initialize_model(train_x, train_y): 
    models = []
    for i in range(train_y.shape[-1]):
        train_objective = train_y[:, i]
        models.append(
            SingleTaskGP(train_x, train_objective.unsqueeze(-1))
        )
    model_list = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
    return mll, model_list

# # Train the models and tune hyperparameters (default method in BoTorch library)
# fit_gpytorch_model(mll) is a method provided by BoTorch (a library built on top of PyTorch) that 
# performs hyperparameter optimization for Gaussian Process (GP) models. It is an alternative way to optimize the hyperparameters 
# of the GP model compared to using a separate optimizer.

mll, trained_model_list = initialize_model(x_tensor, y_tensor)
fit_gpytorch_model(mll)




# # # Train the models and tune the hyperparameters using the Adam optimizer.
# # Define the kernel and Likelihood

# # The kernel specifies the covariance (or similarity) between different points in the input space. 
# # It determines how the output values at different input points are correlated.
# # In other words, the kernel captures the smoothness and correlations in the underlying function 

# # The likelihood models the noise or uncertainty in the observed data. 
# # It represents the probability distribution of the observed output given the true underlying function values 
# # predicted by the GP.
# kernel = SE(1) # argument n_parms = 1
# covar_module = kernel.get_covr_module()
# noise_range = np.array([0.01, 0.05])
# _noise_constraints = noise_range 
# likelihood = GaussianLikelihood(noise_constraint = Interval(_noise_constraints[0], _noise_constraints[1]))

# # Define a function to train the models and tune the hyperparameters (using Adam optimizer)
# def _training_multi_objective(train_x, train_y):
#     """
#     Train the multi-objective model using Adam Optimizer and gradient descent.
#     Log Marginal Likelihood is used as the cost function.
#     """
#     num_objectives = train_y.shape[-1]

#     # Initialize a list to store individual GP models for each objective
#     models = []
#     for i in range(num_objectives):
#         # Create a SingleTaskGP for each objective
#         single_objective_model = SingleTaskGP(train_x, train_y[:, i].unsqueeze(-1), likelihood = likelihood, covar_module = covar_module) 
#         models.append(single_objective_model)

#     # Create a ModelListGP to manage multiple objectives
#     model_list = ModelListGP(*models)
    
#     # Set up the optimizer
#     parameters = list(model_list.parameters()) + list(likelihood.parameters())
#     optimizer = torch.optim.Adam(parameters, lr=0.01)
    
#     # Training loop
#     for epoch in range(1000):
        
#         optimizer.zero_grad()
        
#         # Initialize the sum of marginal log likelihoods for all models
#         sum_mll = 0.0

#         # Loop over individual models in the ModelListGP
#         for i, model in enumerate(model_list.models):
#             # Forward pass to get the output from the model
#             # During the forward pass, operations are performed on tensors, and a computation graph is built to represent the sequence of operations.

#             # Loss Calculation: At the end of the forward pass, a scalar tensor is usually obtained, representing the loss or objective function. 
#             # This scalar tensor is what you want to minimize during training.

#             output = model(train_x)
            
#             # Calculate Exact Marginal Log Likelihood (loss function) for the current model
#             mll = ExactMarginalLogLikelihood(model.likelihood, model)
#             loss = -mll(output, train_y[:, i]) 
            
#             # Add the EMLL for the current model to the sum
#             sum_mll = sum_mll + loss
            
#         # Backward pass 
#         # The backward() method is called on the scalar tensor. It computes the gradients of the loss function with 
#         # respect to each parameter of the model. 
#         sum_mll.backward()
        
#         # Optimization step (Parameter update)
#         # Adjusts the model parameters in the opposite direction of the gradients to minimize the loss.
#         optimizer.step()
        
#         # Print the loss after every 50 epochs
#         # if epoch % 50 == 0:
#         #     print(f"Epoch {epoch}, Loss: {sum_mll.item()}")
    
#     # Extract optimized hyperparameters
  
#     # Objective 1
#     # Hyperparameters of the covariance module (kernel)
#     output_scale_1 = model_list.models[0].covar_module.outputscale.item()
#     length_scale_1 = model_list.models[0].covar_module.base_kernel.lengthscale.item()

#     # Hyperparameters of the likelihood
#     noise_variance_1 = model_list.models[0].likelihood.noise_covar.noise.item()

#     # Objective 2
#     # Hyperparameters of the covariance module (kernel)
#     output_scale_2 = model_list.models[1].covar_module.outputscale.item()
#     length_scale_2 = model_list.models[1].covar_module.base_kernel.lengthscale.item()

#     # Hyperparameters of the likelihood
#     noise_variance_2 = model_list.models[1].likelihood.noise_covar.noise.item()

#     return model_list, output_scale_1, length_scale_1, noise_variance_1, output_scale_2, length_scale_2, noise_variance_2


# # # Visualize the hyperparameter convergence with iterations
# output_scale_1_list = []
# length_scale_1_list = []
# noise_variance_1_list = []

# output_scale_2_list = []
# length_scale_2_list = []
# noise_variance_2_list = []


# for n in range(3, 11):
#     trained_model_list, os_1, ls_1, nv_1, os_2, ls_2, nv_2 = _training_multi_objective(x_tensor[:n], y_tensor[:n])
#     output_scale_1_list.append(os_1)
#     length_scale_1_list.append(ls_1)
#     noise_variance_1_list.append(nv_1)
#     output_scale_2_list.append(os_2)
#     length_scale_2_list.append(ls_2)
#     noise_variance_2_list.append(nv_2)


# # Objective 1 (RMSSD)
# # plotting noise var
# # plt.figure(figsize=(8, 6))

# iterations = range(3, 11, 1)  
# plt.subplot(2, 3, 1)
# plt.plot(iterations, noise_variance_1_list, marker='o', linestyle='-', color='blue')
# plt.title(f'Noise variance versus iterations \n Objective 1 (RMSSD)')
# plt.xlabel('Iteration')
# plt.ylabel('Noise variance')
# plt.grid(True)


# # plotting length scale

# #plt.figure(figsize=(8, 6))
# plt.subplot(2, 3, 2)
# plt.plot(iterations, length_scale_1_list, marker='o', linestyle='-', color='red')
# plt.title(f'Length scale versus iterations \n Objective 1 (RMSSD)')
# plt.xlabel('Iteration')
# plt.ylabel('Length scale')
# plt.grid(True)


# # plotting output scale variance

# #plt.figure(figsize=(8, 6))
# plt.subplot(2, 3, 3)
# plt.plot(iterations, output_scale_1_list, marker='o', linestyle='-', color='green')
# plt.title(f'Output scale versus iterations \n Objective 1 (RMSSD)')
# plt.xlabel('Iteration')
# plt.ylabel('Output scale')
# plt.grid(True)


# # Objective 2 (ETC)
# # plotting noise var
# #plt.figure(figsize=(8, 6))
    
# plt.subplot(2, 3, 4)  
# plt.plot(iterations, noise_variance_2_list, marker='o', linestyle='-', color='blue')
# plt.title(f'Noise variance versus iterations \n Objective 2 (ETC)')
# plt.xlabel('Iteration')
# plt.ylabel('Noise variance')
# plt.grid(True)


# # plotting length scale

# #plt.figure(figsize=(8, 6))
# plt.subplot(2, 3, 5)  
# plt.plot(iterations, length_scale_2_list, marker='o', linestyle='-', color='red')
# plt.title(f'Length scale versus iterations \n Objective 2 (ETC)')
# plt.xlabel('Iteration')
# plt.ylabel('Length scale')
# plt.grid(True)


# # plotting output scale variance

# #plt.figure(figsize=(8, 6))
# plt.subplot(2, 3, 6)
# plt.plot(iterations, output_scale_2_list, marker='o', linestyle='-', color='green')
# plt.title(f'Output scale versus iterations \n Objective 2 (ETC)')
# plt.xlabel('Iteration')
# plt.ylabel('Output scale')
# plt.grid(True)
# plt.subplots_adjust(wspace=0.8, hspace=0.4)
# plt.show()



# # # Plot the final models (GPs for Objective 1 (RMSSD) and Objective 2 (ETC))

# # Call the _training_multi_objective method,  also optimizes the hyperparameters and fits the model, returns the trained model list and optimized hyperparameters
# trained_model_list, os_1, ls_1, nv_1, os_2, ls_2, nv_2 = _training_multi_objective(x_tensor, y_tensor)

# Generate test points for plotting in the range [0, 1]
test_x = torch.linspace(0, 1, 100).unsqueeze(-1)

# Make predictions for the test points
# Iterate over models and get the posterior predictive distribution for the test points

trained_model_list.eval()
scaling_factor = 1.96 # Z-value associated with 95% is z = 1.96 # Z-value associated witb 99% is 2.57
observed = []
observed_mean = []
std_dev = []
lower = []
upper = []

with torch.no_grad():
    # Loop through each model in the list
    for i, model in enumerate(trained_model_list.models):
        # Get the posterior distribution for the i-th model
        observed_i = model.likelihood(model(test_x))
        
        # Extract mean and variance
        observed_mean_i = observed_i.mean.cpu().numpy()
        
        # print(observed_mean_i)
        std_dev_i = observed_i.stddev.cpu().numpy()
        
        # Obtain the confidence intervals
        upper_i = observed_mean_i + scaling_factor * std_dev_i
        lower_i = observed_mean_i - scaling_factor * std_dev_i

        observed.append(observed_i)
        observed_mean.append(observed_mean_i)
        std_dev.append(std_dev_i)
        lower.append(lower_i)
        upper.append(upper_i)


# # Alternate way to generate predictions (y values)
# pred_y = trained_model_list.posterior(x_tensor).mean
# print(pred_y)
# print(pred_y.size())

# Plot the GP for the first objective
plt.figure(figsize=(8, 6))
plt.plot(test_x, observed_mean[0], label='mean', linewidth=3, color='b')
plt.scatter(x, y[:,0], label='Points', color='r', marker='o', s=50)
plt.fill_between(test_x.squeeze(), lower[0].squeeze(), upper[0].squeeze(), alpha=0.3)
plt.xlim(0, 1) # Normalized parameter range
plt.xticks(np.arange(0, 1, 0.2), fontsize=13)
plt.yticks(fontsize=14)
plt.ylim(y[:,0].min()-1, y[:,0].max()+1)
plt.xlabel('Normalized parameter', fontsize=14)
plt.ylabel('Normalized ECG-RMSSD cost', fontsize=14)
plt.axvline(x = 0.5789, color = 'r', label = 'axvline - full height')
plt.show()

# Plot the GP for the second objective 
plt.figure(figsize=(8, 6))
plt.plot(test_x, observed_mean[1], label='mean', linewidth=3, color='g')
plt.scatter(x, y[:,1], label='Points', color='r', marker='o', s=50)
plt.fill_between(test_x.squeeze(), lower[1].squeeze(), upper[1].squeeze(), alpha=0.3)
plt.xlim(0, 1) # Normalized parameter range
plt.xticks(np.arange(0, 1, 0.2), fontsize=13)
plt.yticks(fontsize=14)
plt.ylim(y[:,1].min()-1, y[:,1].max()+1)
plt.xlabel('Normalized parameter', fontsize=14)
plt.ylabel('Normalized ETC cost', fontsize=14)
plt.axvline(x = 0.5789, color = 'r', label = 'axvline - full height')
plt.show()


# Function for plotting GP results
# def plot_gp_result(test_x, observed_mean, lower, upper, x, y, color, label):
#     plt.figure(figsize=(8, 6))
#     plt.plot(test_x, observed_mean, label='mean', linewidth=3, color=color)
#     plt.scatter(x, y, label='Points', color='r', marker='o', s=50)
#     plt.fill_between(test_x.squeeze(), lower.squeeze(), upper.squeeze(), alpha=0.3)
#     plt.xlim(0, 1)  # Normalized parameter range
#     plt.xticks(np.arange(0, 1, 0.2), fontsize=13)
#     plt.yticks(fontsize=14)
#     plt.ylim(y.min() - 1, y.max() + 1)
#     plt.xlabel('Normalized parameter', fontsize=14)
#     plt.ylabel(label, fontsize=14)
#     plt.legend()
#     plt.show()


# # Plot the GP for each objective
# for i in range(len(trained_model_list.models)):
#     plot_gp_result(test_x, observed_mean[i], lower[i], upper[i], x, y[:, i], color=['b', 'g'][i], label=['ECG-RMSSD', 'ETC'][i])