import numpy as np
import matplotlib.pyplot as plt
import torch
import os
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch import fit_gpytorch_model
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling import IIDNormalSampler
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from HIL.optimization.kernel import SE, Matern 


class MultiObjectiveBayesianOptimization(object):

    def __init__(self, bounds) -> None:
        
        """Multi-objective Bayesian optimization for HIL"""
        
        self.NUM_RESTARTS =  10
        self.RAW_SAMPLES = 1024
    
        self.standard_bounds = torch.tensor(bounds)
        self.MC_SAMPLES = 256
        #bounds = torch.tensor([[-1.2], [1.2]])
    

    # Use this code if using the traditional method of fitting the model using fit_gpytorch_model
    def initialize_model(self, train_x, train_y): 

        #train_x = normalize(train_x, bounds)
        
        models = []
        for i in range(train_y.shape[-1]):
            train_objective = train_y[:, i]
            models.append(
                SingleTaskGP(train_x, train_objective.unsqueeze(-1))
            )
        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        return mll, model

    def generate_next_candidate(self, x, y):
            # Convert 2D NumPy arrays to 2D Pytorch tensors
            self.x = torch.tensor(x)
            self.y = torch.tensor(y)
            n_candidates = 1
            
            self.mll, self.model = self.initialize_model(self.x, self.y)
            fit_gpytorch_model(self.mll)
            self._save_model()
            
            sampler=IIDNormalSampler(200, seed = 1234)
        
            #train_x = normalize(x, bounds)
            with torch.no_grad():
                #pred = model.posterior(normalize(train_x, bounds)).mean
                pred = self.model.posterior(self.x).mean
            acq_fun_list = []
            for _ in range(n_candidates):
                
                weights = sample_simplex(2).squeeze()
                objective = GenericMCObjective(
                    get_chebyshev_scalarization(
                        weights,
                        pred
                    )
                )
                acq_fun = qNoisyExpectedImprovement(
                    model=self.model,
                    objective=objective,
                    sampler=sampler,
                    #X_baseline=train_x,
                    X_baseline=self.x,
                    prune_baseline=True,
                )
                acq_fun_list.append(acq_fun)
            
        
            candidates, _ = optimize_acqf_list(
                acq_function_list=acq_fun_list,
                bounds=self.standard_bounds,
                num_restarts=self.NUM_RESTARTS,
                raw_samples=self.RAW_SAMPLES,
                options={
                    "batch_limit": 5,
                    "maxiter": 200,
                }
            )
        
            #return unnormalize(candidates, bounds)
            return candidates     
    
    # # Use this code if fitting the model and tuning the hyperparameters using Adam optimizer.
    # def _training_multi_objective(train_x, train_y):
    #     """
    #     Train the multi-objective model using Adam Optimizer and gradient descent.
    #     Log Marginal Likelihood is used as the cost function.
    #     """
        
    #     # Define the kernel and Likelihood
    #     kernel = SE(1) # argument n_parms = 1
    #     covar_module = kernel.get_covr_module()
    #     noise_range = np.array([0, 0.1])
    #     _noise_constraints = noise_range 
    #     likelihood = GaussianLikelihood(noise_constraint = Interval(_noise_constraints[0], _noise_constraints[1]))

        
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
    #     for epoch in range(500):
            
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

    #         if epoch % 50 == 0:
    #             print(f"Epoch {epoch}, Loss: {sum_mll.item()}")

    #     return model_list
    
    # def generate_next_candidate(self, x, y):
    #     # Convert 2D NumPy arrays to 2D Pytorch tensors
    #     self.x = torch.tensor(x)
    #     self.y = torch.tensor(y)
    #     n_candidates = 1
        
    #     self.model = self._training_multi_objective(self.x, self.y)
    #     self._save_model()
        
    #     sampler=IIDNormalSampler(200, seed = 1234)
      
    #     #train_x = normalize(x, bounds)
    #     with torch.no_grad():
    #         #pred = model.posterior(normalize(train_x, bounds)).mean
    #         pred = self.model.posterior(self.x).mean
    #     acq_fun_list = []
    #     for _ in range(n_candidates):
            
    #         weights = sample_simplex(2).squeeze()
    #         objective = GenericMCObjective(
    #             get_chebyshev_scalarization(
    #                 weights,
    #                 pred
    #             )
    #         )
    #         acq_fun = qNoisyExpectedImprovement(
    #             model=self.model,
    #             objective=objective,
    #             sampler=sampler,
    #             #X_baseline=train_x,
    #             X_baseline=self.x,
    #             prune_baseline=True,
    #         )
    #         acq_fun_list.append(acq_fun)
        
    
    #     candidates, _ = optimize_acqf_list(
    #         acq_function_list=acq_fun_list,
    #         bounds=self.standard_bounds,
    #         num_restarts=self.NUM_RESTARTS,
    #         raw_samples=self.RAW_SAMPLES,
    #         options={
    #             "batch_limit": 5,
    #             "maxiter": 200,
    #         }
    #     )
    
    #     #return unnormalize(candidates, bounds)
    #     return candidates
    

    def _save_model(self) -> None:
        """Save the model and data in the given path
        """
        save_iter_path = "models/" + f'iter_{len(self.x)}'
        os.makedirs(save_iter_path, exist_ok=True)
        model_path = save_iter_path +'/model.pth'
        torch.save(self.model.state_dict(), model_path) #type: ignore
        data_save = save_iter_path + '/data.csv'
        x = self.x.detach().cpu().numpy()
        y = self.y.detach().cpu().numpy()
        full_data = np.hstack((x,y))
        np.savetxt(data_save, full_data)
        print(f"model saved successfully at {save_iter_path}")
    
   
    

