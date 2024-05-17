import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch import fit_gpytorch_model
from botorch.optim.optimize import optimize_acqf_list
# from botorch.utils.transforms import unnormalize, normalize
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.sampling import sample_simplex
from botorch.acquisition.objective import GenericMCObjective
from botorch.sampling import IIDNormalSampler
# from gpytorch.likelihoods import GaussianLikelihood
# from gpytorch.constraints import GreaterThan, Interval
from botorch.sampling.samplers import SobolQMCNormalSampler
from HIL.optimization.kernel import SE, Matern
from gpytorch.kernels import RBFKernel,ScaleKernel
# from gpytorch.priors import NormalPrior
# from HIL.optimization.rgpe_functons import create_rgpe
from HIL.optimization.RGPE_model import RGPE
from gpytorch.likelihoods import GaussianLikelihood





class MultiObjectiveBayesianOptimization(object):

    def __init__(self, bounds, is_rgpe:bool, base_model_path = 'base_models', x_dim = 1) -> None:
        
        """Multi-objective Bayesian optimization for HIL"""
        
        self.NUM_RESTARTS =  10
        self.RAW_SAMPLES = 1024
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.standard_bounds = torch.tensor(bounds)
        self.MC_SAMPLES = 256
        self.is_rgpe = is_rgpe
        self.base_model_path = base_model_path
        self.x_dim = x_dim
        if self.is_rgpe and self.base_model_path is None:
            raise Exception("please give the path for the base models")
        elif self.is_rgpe and self.base_model_path is not None:
            self.create_base_models()
            print('\n\n length of base model list: ', len(self.base_model_list[1]))
        #bounds = torch.tensor([[-1.2], [1.2]])
    

    # Use this code if using the traditional method of fitting the model using fit_gpytorch_model
    def initialize_model(self, train_x, train_y): 
        models = []
        #train_x = normalize(train_x, bounds)
        if not self.is_rgpe:
            
            for i in range(train_y.shape[-1]):
                train_objective = train_y[:, i]
                models.append(
                    SingleTaskGP(train_x, train_objective.unsqueeze(-1))
                )
            model = ModelListGP(*models)
            mll = SumMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_model(mll)
        
        else:
            for i in range(train_y.shape[-1]):
                target_model = SingleTaskGP(train_x, train_y[:, i].unsqueeze(-1), covar_module=ScaleKernel(
                    base_kernel=RBFKernel(ard_num_dims=self.x_dim)))
                target_model.likelihood = GaussianLikelihood()
                target_model.to(self.device)
                mll = ExactMarginalLogLikelihood(target_model.likelihood, target_model)
                target_model.Y_mean = train_y.mean(dim=-2, keepdim=True)
                target_model.Y_std = train_y.std(dim=-2, keepdim=True)
                fit_gpytorch_model(mll)

                model_list = self.base_model_list[i] + [target_model]
                # print('\n\n length of model_list', len(model_list))
                weights = self.compute_weights(train_x, train_y[:, i].unsqueeze(-1), self.base_model_list[i], target_model, 256, self.device)
                print(f'\n\n\n the final weights are: {weights}')
                models.append(RGPE(model_list, weights))
            model = ModelListGP(*models)

        return model

    def generate_next_candidate(self, x, y):
            # Convert 2D NumPy arrays to 2D Pytorch tensors
            self.x = torch.tensor(x)
            self.y = torch.tensor(y)
            n_candidates = 1
            
            self.model = self.initialize_model(self.x, self.y)
            
            self._save_model()
            
            sampler=IIDNormalSampler(200, seed = 1234)
        
            #train_x = normalize(x, bounds)
            with torch.no_grad():
                #pred = model.posterior(normalize(train_x, bounds)).mean
                pred = self.model.posterior(self.x).mean
            print(f'\n\npred: {pred}')
            acq_fun_list = []
            for _ in range(n_candidates):
                
                # weights = sample_simplex(2).squeeze()
                weights = torch.tensor([0.5, 0.5]) #use this if you want equal importance for both objectives
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

    def create_base_models(self, all_positive:bool = False):
        self.base_model_list = []
        for i in os.listdir(self.base_model_path):
            models = []
            #csv_files = self.find_csv_filenames("base_models/"+i)
            with open(self.base_model_path+"/"+i+"/data.csv") as f:
                data = pd.read_csv(f, delim_whitespace=True)
            x_df = data.iloc[:, :self.x_dim]
            y_df = data.iloc[:, self.x_dim:]
            x = torch.tensor(x_df.values, dtype=torch.float64)
            y = torch.tensor(y_df.values, dtype=torch.float64)
            print(f'x: {x.shape}, y: {y.shape}')
            if all_positive:
                if any(y < 0):
                    y = -y
            print(x.shape,y.shape)
            print(y)
            #model_weights = torch.load("base_models/"+i+"/model.pth")
            #model=SingleTaskGP(x,y)
            #model.load_state_dict(torch.load("base_models/"+i+"/model.pth" ))
            # model = self.get_fitted_model_for_rgpe(x,y,state_dict =None,dimension=x.shape[0])
            for i in range(y.shape[-1]):
                model = SingleTaskGP(x, y[:, i].unsqueeze(-1), covar_module=ScaleKernel(
                    base_kernel=RBFKernel(ard_num_dims=self.x_dim)))
                model.likelihood = GaussianLikelihood()
                model.to(self.device)
                model.Y_mean = y.mean(dim=-2, keepdim=True)
                model.Y_std = y.std(dim=-2, keepdim=True)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)
                models.append(model)
            self.base_model_list.append(models)

    def compute_weights(self,train_x,train_y, base_models, target_model, num_samples, device):  
        """
        Compute ranking weights for each base model and the target model (using 
            LOOCV for the target model). Note: This implementation does not currently 
            address weight dilution, since we only have a small number of base models.

        Args:
            train_x: `n x d` tensor of training points (for target task)
            train_y: `n` tensor of training targets (for target task)
            base_models: list of base models
            target_model: target model
            num_samples: number of mc samples

        Returns:
            Tensor: `n_t`-dim tensor with the ranking weight for each model
        """
        ranking_losses = []
        # compute ranking loss for each base model
        for task in range(len(base_models)):

            model = base_models[task]
            # compute posterior over training points for target task
            posterior = model.posterior(train_x)
            sampler = SobolQMCNormalSampler(num_samples=num_samples)
            base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)

            #base_f_samps is the other models prediction at train_X
            # compute and save ranking loss
            ranking_losses.append(self.compute_ranking_loss(base_f_samps, train_y))
        # compute ranking loss for target model using LOOCV
        # f_samps

        target_f_samps = self.get_target_model_loocv_sample_preds(
            train_x, train_y, target_model, num_samples,device)
        ranking_losses.append(self.compute_ranking_loss(target_f_samps, train_y))
        ranking_loss_tensor = torch.stack(ranking_losses)
        # compute best model (minimum ranking loss) for each sample
        best_models = torch.argmin(ranking_loss_tensor, dim=0)
        # compute proportion of samples for which each model is best
        rank_weights = best_models.bincount(minlength=len(ranking_losses)).type_as(train_x) / num_samples

        return rank_weights
    
    def compute_ranking_loss(self,f_samps, target_y):
        """
        Compute ranking loss for each sample from the posterior over target points.
        
        Args:
            f_samps: `n_samples x (n) x n`-dim tensor of samples
            target_y: `n x 1`-dim tensor of targets
        Returns:
            Tensor: `n_samples`-dim tensor containing the ranking loss across each sample
        """
        n = target_y.shape[0]

        #print(f_samps.shape,"fsamps shape")
        if f_samps.ndim == 3:
            # Compute ranking loss for target model
            # take cartesian product of target_y
            cartesian_y = torch.cartesian_prod(
                target_y.squeeze(-1), 
                target_y.squeeze(-1),
            ).view(n, n, 2)
            #print(cartesian_y)
            # the diagonal of f_samps are the out-of-sample predictions
            # for each LOO model, compare the out of sample predictions to each in-sample prediction
            rank_loss = (
                (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps) ^
                (cartesian_y[..., 0] < cartesian_y[..., 1])
            ).sum(dim=-1).sum(dim=-1)

        else:
            rank_loss = torch.zeros(f_samps.shape[0], dtype=torch.long, device=target_y.device)
            y_stack = target_y.squeeze(-1).expand(f_samps.shape)
            for i in range(1,target_y.shape[0]):
                rank_loss += (
                    (self.roll_col(f_samps, i) > f_samps) ^ (self.roll_col(y_stack, i) > y_stack)
                ).sum(dim=-1)

        return rank_loss
    
    def get_target_model_loocv_sample_preds(self,train_x, train_y, target_model, num_samples,device):
        """
        Create a batch-mode LOOCV GP and draw a joint sample across all points from the target task.
        
        Args:
            train_x: `n x d` tensor of training points
            train_y: `n x 1` tensor of training targets
            target_model: fitted target model
            num_samples: number of mc samples to draw
        
        Return: `num_samples x n x n`-dim tensor of samples, where dim=1 represents the `n` LOO models,
            and dim=2 represents the `n` training points.
        """
        #print(train_x.shape,train_y.shape,yvar.shape,"fsamps")
        batch_size = len(train_x)
        masks = torch.eye(len(train_x), dtype=torch.uint8, device=device).bool()
        train_x_cv = torch.stack([train_x[~m] for m in masks])
        train_y_cv = torch.stack([train_y[~m] for m in masks])
        state_dict = target_model.state_dict()
        # expand to batch size of batch_mode LOOCV model
        state_dict_expanded = {
            name: t.expand(batch_size, *[-1 for _ in range(t.ndim)])
            for name, t in state_dict.items()
        }
        model = SingleTaskGP(train_x_cv, train_y_cv, covar_module=ScaleKernel(batch_shape = [train_x_cv.shape[0]],
                    base_kernel=RBFKernel(ard_num_dims=self.x_dim, batch_shape = [train_x_cv.shape[0]])))
        model.likelihood = GaussianLikelihood(batch_shape=[train_x_cv.shape[0]])
        model.Y_mean = train_y_cv.mean(dim=-2, keepdim=True)
        model.Y_std = train_y_cv.std(dim=-2, keepdim=True)
        model.load_state_dict(state_dict_expanded)
        with torch.no_grad():
            posterior = model.posterior(train_x)
            # Since we have a batch mode gp and model.posterior always returns an output dimension,
            # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
            # the last dimension.
            sampler = SobolQMCNormalSampler(num_samples=num_samples)
            return sampler(posterior).squeeze(-1)
    
    def roll_col(self,X, shift):  
        """
        Rotate columns to right by shift.
        """
        return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)

        # self.base_model_list = base_model_list   
    
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
        model_path = save_iter_path +'/multi_model.pth'
        torch.save(self.model.state_dict(), model_path) #type: ignore
        data_save = save_iter_path + '/data.csv'
        x = self.x.detach().cpu().numpy()
        y = self.y.detach().cpu().numpy()
        full_data = np.hstack((x,y))
        np.savetxt(data_save, full_data)
        print(f"model saved successfully at {save_iter_path}")
    
   
    

