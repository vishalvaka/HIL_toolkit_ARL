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

# # Single objective optimization
# # Load x and y
# os.chdir(r"C:\Users\sruth\OneDrive - University of Illinois at Chicago\Documents\Sruthi\HIL_toolkit-main\HIL_toolkit-main\models\iter_15")
# # Import the x and y data
# df_xy = pd.read_csv("data.csv", delim_whitespace=True, header=None)
# df_xy.columns = ["x", "y"]
# print(df_xy["x"])
# print(df_xy["y"])

# # Convert x and y columns to PyTorch tensors
# x_tensor = torch.tensor(df_xy["x"].values, dtype=torch.float32)
# y_tensor = torch.tensor(df_xy["y"].values, dtype=torch.float32)

# # Instantiate SingleTaskGP with training data
# Kernel = SE(1) # 1 represents number of parameters
# Covar_module = Kernel.get_covr_module()
# noise_range = np.array([0.005, 10])
# Likelihood = GaussianLikelihood(noise_constraint = Interval(noise_range[0], noise_range[1]))
# loaded_model = SingleTaskGP(train_X=x_tensor.unsqueeze(-1),train_Y=y_tensor.unsqueeze(-1), likelihood = Likelihood, covar_module = Covar_module )

# # Train the model
# mll = ExactMarginalLogLikelihood(likelihood=Likelihood, model=loaded_model)
# fit_gpytorch_model(mll)


# model_path = r"C:\Users\sruth\OneDrive - University of Illinois at Chicago\Documents\Sruthi\HIL_toolkit-main\HIL_toolkit-main\models\iter_15\model.pth"
# loaded_model.load_state_dict(torch.load(model_path))

# # load_state_dict is a method in PyTorch that is used to load the state dictionary into a PyTorch model. 
# # The state dictionary contains the parameters of the model, and this method is commonly used to load pre-trained models 
# # or to resume training from a saved checkpoint.


# # Assuming your model has a predict method
# def predict(model, x):
#    with torch.no_grad():
#        y_pred = model(x)
#    return y_pred.mean, y_pred.variance

# # Generate test points for plotting in the range [0, 1]
# test_x = torch.linspace(0, 1, 100).unsqueeze(-1)

# # Make predictions for the test points
# mean, variance = predict(loaded_model, test_x)

# # Convert tensor to numpy array
# test_x_np = test_x.cpu().numpy().squeeze()  # Squeeze to make it 1-dimensional
# mean_np = mean.cpu().numpy().squeeze()
# std_dev_np = torch.sqrt(variance).cpu().numpy().squeeze()

# # Plot the Gaussian Process with uncertainties
# plt.figure(figsize=(8, 6))
# plt.plot(test_x_np, mean_np, label='Mean Prediction')
# plt.fill_between(test_x_np, mean_np - std_dev_np, mean_np + std_dev_np, alpha=0.2, label='Uncertainties')
# plt.scatter(x_tensor.cpu().numpy(), y_tensor.cpu().numpy(), color='red', marker='x', label='Observations')
# plt.title('Gaussian Process with Uncertainties')
# plt.xlabel('Input')
# plt.ylabel('Output')
# plt.legend()
# plt.show()

# # Multi-objective optimization - plotting the final models

# Load x and y
# os.chdir(r"./models/iter_15")
# Import the x and y data
df_xy = pd.read_csv("./models/iter_15/data.csv", delim_whitespace=True)
# df_xy = pd.read_csv("./function_data.csv")

# Extract the 'x' column and convert to a 1D Numpy array
# x = df_xy['x_normalized'].values

# Since the provided CSV file only contains 2 columns ('x' and 'y'),
# and based on your description, if you expect to extract 'y' as a separate column:
# y = df_xy['y_normalized'].values.reshape(-1, 1) 

# Extract the first column as x and convert to a 1D Numpy array
x = df_xy.iloc[:, 0].values

# # Extract the second and third columns as a vector of two objectives (y1 and y2), and convert to a 2D Numpy array
y = df_xy.iloc[:, 1:].values

# # Print the extracted data
print("x:")
print(x)
print("\ny:")
print(y)
print("Length of x:", len(x))
print("Length of y:", len(y))
print("Shape of x", x.shape)
print("Shape of y", y.shape)

# Convert x and y numpy arrays to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32)
x_tensor = x_tensor.view(x_tensor.shape[0], 1)
y_tensor = torch.tensor(y, dtype=torch.float32)
print(x_tensor)
print(y_tensor)
print("Shape of x tensor", x_tensor.shape)
print("Shape of y tensor", y_tensor.shape)
# Following the same procedure as in the MOBO.py code

# # function to initialize GP models
# def initialize_model(train_x, train_y): 
#     models = []
#     for i in range(train_y.shape[-1]):
#         train_objective = train_y[:, i]
#         models.append(
#             SingleTaskGP(train_x, train_objective.unsqueeze(-1))
#         )
#     model_list = ModelListGP(*models)
#     mll = SumMarginalLogLikelihood(model_list.likelihood, model_list)
#     return mll, model_list

# # # Train the models and tune hyperparameters (default method in BoTorch library)
# # fit_gpytorch_model(mll) is a method provided by BoTorch (a library built on top of PyTorch) that 
# # performs hyperparameter optimization for Gaussian Process (GP) models. It is an alternative way to optimize the hyperparameters 
# # of the GP model compared to using a separate optimizer.

# mll, trained_model_list = initialize_model(x_tensor, y_tensor)
# fit_gpytorch_model(mll)
# pred_y = trained_model_list.posterior(x_tensor).mean
# print(pred_y)
# print(pred_y.size())

# Define the kernel and Likelihood
kernel = SE(1) # argument n_parms = 1
covar_module = kernel.get_covr_module()
noise_range = np.array([0, 0.4])
_noise_constraints = noise_range 
likelihood = GaussianLikelihood(noise_constraint = Interval(_noise_constraints[0], _noise_constraints[1]))


# Train the models and tune the hyperparameters (using Adam optimizer)
def _training_multi_objective(train_x, train_y):
    """
    Train the multi-objective model using Adam Optimizer and gradient descent.
    Log Marginal Likelihood is used as the cost function.
    """
    num_objectives = train_y.shape[-1]
    print('@@'+str(num_objectives))

    # Initialize a list to store individual GP models for each objective
    models = []
    for i in range(num_objectives):
        # Create a SingleTaskGP for each objective
        single_objective_model = SingleTaskGP(train_x, train_y[:, i].unsqueeze(-1), likelihood = likelihood, covar_module = covar_module) 
        models.append(single_objective_model)

    # Create a ModelListGP to manage multiple objectives
    model_list = ModelListGP(*models)
   
    # Set up the optimizer
    parameters = list(model_list.parameters()) + list(likelihood.parameters())
    optimizer = torch.optim.Adam(parameters, lr=0.01)

    # Training loop
    for epoch in range(1000):
        
        optimizer.zero_grad()
        
        # Initialize the sum of marginal log likelihoods for all models
        sum_mll = 0.0

        # Loop over individual models in the ModelListGP
        for i, model in enumerate(model_list.models):
            # Forward pass to get the output from the model
            # During the forward pass, operations are performed on tensors, and a computation graph is built to represent the sequence of operations.

            # Loss Calculation: At the end of the forward pass, a scalar tensor is usually obtained, representing the loss or objective function. 
            # This scalar tensor is what you want to minimize during training.

            output = model(train_x)
            
            # Calculate Exact Marginal Log Likelihood (loss function) for the current model
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            loss = -mll(output, train_y[:, i]) 
            
            # Add the EMLL for the current model to the sum
            sum_mll = sum_mll + loss
            
        # Backward pass 
        # The backward() method is called on the scalar tensor. It computes the gradients of the loss function with 
        # respect to each parameter of the model. 
        sum_mll.backward()
        
        # Optimization step (Parameter update)
        # Adjusts the model parameters in the opposite direction of the gradients to minimize the loss.
        optimizer.step()

        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Loss: {sum_mll.item()}")

    return model_list

# Call the _training_multi_objective method,  also optimizes the hyperparameters and fits the model
trained_model_list = _training_multi_objective(x_tensor, y_tensor)

# Generate test points for plotting in the range [0, 1]
test_x = torch.linspace(0, 1, 100).unsqueeze(-1)

# Make predictions for the test points
# Iterate over models and get the posterior predictive distribution for the test points

trained_model_list.eval()
scaling_factor = 1.96 
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
        print(observed_mean_i)
        std_dev_i = observed_i.stddev.cpu().numpy()
        
        # Obtain the confidence intervals
        upper_i = observed_mean_i + scaling_factor * std_dev_i
        lower_i = observed_mean_i - scaling_factor * std_dev_i

        observed.append(observed_i)
        observed_mean.append(observed_mean_i)
        std_dev.append(std_dev_i)
        lower.append(lower_i)
        upper.append(upper_i)

# Plot the GP for the first objective
for i in range(len(observed_mean)):
    plt.figure(figsize=(8, 6))
    plt.plot(test_x, observed_mean[i], label='mean', linewidth=3, color='b')
    plt.scatter(x, y[:,i], label='Points', color='r', marker='o', s=50)
    plt.fill_between(test_x.squeeze(), lower[i].squeeze(), upper[i].squeeze(), alpha=0.3)
    plt.xlim(0, 1) # Normalized parameter range
    plt.xticks(np.arange(0, 1, 0.2), fontsize=13)
    plt.yticks(fontsize=14)
    plt.ylim(y[:,i].min()-1, y[:,i].max()+1)
    plt.xlabel('Normalized parameter', fontsize=14)
    # plt.ylabel('Normalized ECG-RMSSD cost', fontsize=14)
    plt.ylabel('Normalized ' + df_xy.columns.to_list()[i+1] + ' cost', fontsize=14)
    # plt.show()
    plt.savefig('GP ' + str(i) + '.png')

# # Plot the GP for the second objective 
# plt.figure(figsize=(8, 6))
# plt.plot(test_x, observed_mean[1], label='mean', linewidth=3, color='g')
# plt.scatter(x, y[:,1], label='Points', color='r', marker='o', s=50)
# plt.fill_between(test_x.squeeze(), lower[1].squeeze(), upper[1].squeeze(), alpha=0.3)
# plt.xlim(0, 1) # Normalized parameter range
# plt.xticks(np.arange(0, 1, 0.2), fontsize=13)
# plt.yticks(fontsize=14)
# plt.ylim(y[:,1].min()-1, y[:,1].max()+1)
# plt.xlabel('Normalized parameter', fontsize=14)
# plt.ylabel('Normalized ETC cost', fontsize=14)
# plt.show()

# # Function for plotting GP results
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