import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
from botorch.models import SingleTaskGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood, ExactMarginalLogLikelihood
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.exceptions.errors import BotorchTensorDimensionError
from botorch import fit_gpytorch_model
from botorch.optim.optimize import optimize_acqf
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
import gpytorch
from torch import Tensor
from typing import Callable, Optional
from botorch.utils.transforms import normalize

def get_chebyshev_scalarization_unnormalized(
    weights: Tensor, Y: Tensor, alpha: float = 0.05
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    r"""Construct an augmented Chebyshev scalarization.

    Augmented Chebyshev scalarization:
        objective(y) = min(w * y) + alpha * sum(w * y)

    Outcomes are first normalized to [0,1] for maximization (or [-1,0] for minimization)
    and then an augmented Chebyshev scalarization is applied.

    Note: this assumes maximization of the augmented Chebyshev scalarization.
    Minimizing/Maximizing an objective is supported by passing a negative/positive
    weight for that objective. To make all w * y's have positive sign
    such that they are comparable when computing min(w * y), outcomes of minimization
    objectives are shifted from [0,1] to [-1,0].

    See [Knowles2005]_ for details.

    This scalarization can be used with qExpectedImprovement to implement q-ParEGO
    as proposed in [Daulton2020qehvi]_.

    Args:
        weights: A `m`-dim tensor of weights.
            Positive for maximization and negative for minimization.
        Y: A `n x m`-dim tensor of observed outcomes, which are used for
            scaling the outcomes to [0,1] or [-1,0].
        alpha: Parameter governing the influence of the weighted sum term. The
            default value comes from [Knowles2005]_.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = torch.tensor([0.75, -0.25])
        >>> transform = get_aug_chebyshev_scalarization(weights, Y)
    """
    if weights.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "weights must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {weights.shape} and {Y.shape}."
        )
    elif Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")

    def chebyshev_obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        product = weights * Y
        return product.min(dim=-1).values + alpha * product.sum(dim=-1)

    if Y.shape[-2] == 0:
        # If there are no observations, we do not need to normalize the objectives
        return chebyshev_obj
    if Y.shape[-2] == 1:
        # If there is only one observation, set the bounds to be
        # [min(Y_m), min(Y_m) + 1] for each objective m. This ensures we do not
        # divide by zero
        Y_bounds = torch.cat([Y, Y + 1], dim=0)
    else:
        # Set the bounds to be [min(Y_m), max(Y_m)], for each objective m
        Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])

    # A boolean mask indicating if minimizing an objective
    minimize = weights < 0

    def obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # scale to [0,1]
        # Y_normalized = normalize(Y, bounds=Y_bounds)
        # # If minimizing an objective, convert Y_normalized values to [-1,0],
        # # such that min(w*y) makes sense, we want all w*y's to be positive
        # Y_normalized[..., minimize] = Y_normalized[..., minimize] - 1
        return chebyshev_obj(Y=Y)

    return obj

def get_weighted_sum(
    weights: Tensor, Y: Tensor
) -> Callable[[Tensor, Optional[Tensor]], Tensor]:
    r"""Construct an augmented Chebyshev scalarization.

    Augmented Chebyshev scalarization:
        objective(y) = min(w * y) + alpha * sum(w * y)

    Outcomes are first normalized to [0,1] for maximization (or [-1,0] for minimization)
    and then an augmented Chebyshev scalarization is applied.

    Note: this assumes maximization of the augmented Chebyshev scalarization.
    Minimizing/Maximizing an objective is supported by passing a negative/positive
    weight for that objective. To make all w * y's have positive sign
    such that they are comparable when computing min(w * y), outcomes of minimization
    objectives are shifted from [0,1] to [-1,0].

    See [Knowles2005]_ for details.

    This scalarization can be used with qExpectedImprovement to implement q-ParEGO
    as proposed in [Daulton2020qehvi]_.

    Args:
        weights: A `m`-dim tensor of weights.
            Positive for maximization and negative for minimization.
        Y: A `n x m`-dim tensor of observed outcomes, which are used for
            scaling the outcomes to [0,1] or [-1,0].
        alpha: Parameter governing the influence of the weighted sum term. The
            default value comes from [Knowles2005]_.

    Returns:
        Transform function using the objective weights.

    Example:
        >>> weights = torch.tensor([0.75, -0.25])
        >>> transform = get_aug_chebyshev_scalarization(weights, Y)
    """
    if weights.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "weights must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {weights.shape} and {Y.shape}."
        )
    elif Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")

    def chebyshev_obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        product = weights * Y
        # return product.min(dim=-1).values + alpha * product.sum(dim=-1)
        return product.sum(dim=-1)
    if Y.shape[-2] == 0:
        # If there are no observations, we do not need to normalize the objectives
        return chebyshev_obj
    if Y.shape[-2] == 1:
        # If there is only one observation, set the bounds to be
        # [min(Y_m), min(Y_m) + 1] for each objective m. This ensures we do not
        # divide by zero
        Y_bounds = torch.cat([Y, Y + 1], dim=0)
    else:
        # Set the bounds to be [min(Y_m), max(Y_m)], for each objective m
        Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])

    # A boolean mask indicating if minimizing an objective
    minimize = weights < 0

    def obj(Y: Tensor, X: Optional[Tensor] = None) -> Tensor:
        # scale to [0,1]
        Y_normalized = normalize(Y, bounds=Y_bounds)
        # If minimizing an objective, convert Y_normalized values to [-1,0],
        # such that min(w*y) makes sense, we want all w*y's to be positive
        Y_normalized[..., minimize] = Y_normalized[..., minimize] - 1
        return chebyshev_obj(Y=Y_normalized)

    return obj

class MultiObjectiveBayesianOptimization(object):

    def __init__(self, bounds, is_rgpe:bool, base_model_path = 'base_models', x_dim = 1) -> None:
        
        """Multi-objective Bayesian optimization for HIL"""
        
        self.NUM_RESTARTS =  10
        self.RAW_SAMPLES = 1024
        self.device = 'cpu'
        self.standard_bounds = torch.tensor(bounds)
        self.MC_SAMPLES = 256
        self.is_rgpe = is_rgpe
        self.base_model_path = base_model_path
        self.x_dim = x_dim
        if self.is_rgpe and self.base_model_path is None:
            raise Exception("please give the path for the base models")
        elif self.is_rgpe and self.base_model_path is not None:
            self.create_base_models()
            print('\n\n length of base model list: ', len(self.base_model_list[0]))
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
                # models.append(SingleTaskGP(train_x, train_objective.unsqueeze(-1), covar_module=ScaleKernel(
                #     base_kernel=RBFKernel(ard_num_dims=self.x_dim)), likelihood=GaussianLikelihood()))
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
                target_model.Y_mean = train_y[:, i].mean(dim=-2, keepdim=True)
                target_model.Y_std = train_y[:, i].std(dim=-2, keepdim=True)
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
            # print(f'\n\npred: {pred}')
            # acq_fun_list = []
            # for _ in range(n_candidates):
                
            # weights = sample_simplex(2).squeeze()
            weights = torch.tensor([0.5, 0.5]) #use this if you want equal importance for both objectives
            objective = GenericMCObjective(
                # get_weighted_sum(
                # get_chebyshev_scalarization(
                get_chebyshev_scalarization_unnormalized(
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
                # acq_fun_list.append(acq_fun)
            
            

            candidates, _ = optimize_acqf(
                acq_function=acq_fun,
                bounds=self.standard_bounds,
                q=1,
                num_restarts=self.NUM_RESTARTS,
                raw_samples=self.RAW_SAMPLES,
                options={
                    "batch_limit": 5,
                    "maxiter": 200,
                }
            )

            # test_x = np.linspace(self.standard_bounds[0].squeeze(), self.standard_bounds[1].squeeze(), 1000)
            # acquisition_values = acq_fun(test_x.unsqueeze(-2))
            # max_acq_value, max_index = torch.max(acquisition_values, dim=0)
            # max_acq_x = test_x[max_index]

            # max_acq_x_scalar = max_acq_x.item()  # This ensures it's a scalar if it's a single-element tensor
            # max_acq_value_scalar = max_acq_value.item()
            # # Plotting
            # plt.figure(figsize=(10, 5))
            # # plt.plot(train_x.numpy(), train_y.numpy(), 'ro', label='Observations')
            # plt.plot(test_x.numpy(), acquisition_values.detach().numpy(), label='Acquisition Value')
            # plt.scatter(max_acq_x_scalar, max_acq_value_scalar, color='g', s=100, zorder=5, label=f'Max Acq at x={max_acq_x_scalar:.2f}')
            # plt.annotate(f'x={max_acq_x_scalar:.2f}', (max_acq_x_scalar, max_acq_value_scalar), textcoords="offset points", xytext=(0,10), ha='center')
            # plt.title('qNoisyExpected Improvement over Parameter Space')
            # plt.xlabel('Parameter')
            # plt.ylabel('Acquisition Value')
            # plt.legend()
            # plt.show()

            #return unnormalize(candidates, bounds)
            return candidates

    def create_base_models(self, all_positive:bool = False):
        self.base_model_list = []
        num_objectives = None

        for i in os.listdir(self.base_model_path):
            models = []
            #csv_files = self.find_csv_filenames("base_models/"+i)
            with open(self.base_model_path+"/"+i+"/data.csv") as f:
                data = pd.read_csv(f, delim_whitespace=True, header=None)
            x_df = data.iloc[:, :self.x_dim]
            y_df = data.iloc[:, self.x_dim:]
            print(f'x_df: {x_df}')
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
            if num_objectives is None:
                num_objectives = y.shape[-1]
                self.base_model_list = [[] for _ in range(num_objectives)]
            
            for obj_index in range(num_objectives):
                model = SingleTaskGP(x, y[:, obj_index].unsqueeze(-1), covar_module=ScaleKernel(
                    base_kernel=RBFKernel(ard_num_dims=self.x_dim)))

                model.likelihood = GaussianLikelihood()

                # model = SingleTaskGP(x, y[: obj_index].unsqueeze(-1))
                model.to(self.device)
                model.Y_mean = y[:, obj_index].mean(dim=-2, keepdim=True)
                model.Y_std = y[:, obj_index].std(dim=-2, keepdim=True)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_model(mll)
                self.base_model_list[obj_index].append(model)

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
        model = SingleTaskGP(train_x_cv, train_y_cv, covar_module=ScaleKernel(batch_shape = [train_x.shape[0]],
                    base_kernel=RBFKernel(ard_num_dims=self.x_dim, batch_shape = [train_x.shape[0]])))
        model.likelihood = GaussianLikelihood(batch_shape=[train_x.shape[0]])
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
    
    def plot_final(self):

        test_x = np.linspace(self.standard_bounds[0], self.standard_bounds[1], 100)
        test_x = torch.from_numpy(test_x).float()
        self.model.eval()

        # with torch.no_grad(), gpytorch.settings.fast_pred_var():
        #     observed_pred = self.model.posterior(test_x)
        #     # observed_pred = self.model.models[i].likelihood(self.model.models[i](test_x))
        #     mean = observed_pred.mean
        #     lower, upper = observed_pred.mvn.confidence_region()
        #     lower = -lower.T
        #     upper = -upper.T
        #     mean = -mean.T
        #     print(f'lower shape: {lower.shape} upper shape: {upper.shape} mean shape: {mean.shape}')

        # Plotting
        # for i in range(lower.shape[1]):
        #     plt.figure(figsize=(10, 6))
        #     # plt.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Training Data')
        #     plt.plot(test_x.numpy(), mean[i].numpy(), 'b', label='Mean')
        #     plt.fill_between(test_x.squeeze().numpy(), lower[i].numpy(), upper[i].numpy(), alpha=0.5, label='Confidence Interval')
        #     plt.xlabel('x')
        #     plt.ylabel('y')
        #     plt.title(f'objective {i+1}')
        #     plt.legend()
        #     plt.show()

        for i, model in enumerate(self.model.models):
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                observed_pred = model.likelihood(model(test_x))
                if isinstance(observed_pred, gpytorch.distributions.MultivariateNormal):
                    mean = -observed_pred.mean
                    try:
                        variance = observed_pred.variance
                    except AttributeError:
                        variance = observed_pred.covariance_matrix.diag()
                else:
                    mean = -observed_pred[0].mean
                    try:
                        variance = observed_pred[0].variance
                    except AttributeError:
                        variance = observed_pred[0].covariance_matrix.diag()

                plt.figure(figsize=(10, 6))
                plt.plot(test_x.numpy().squeeze(), mean.detach().numpy(), 'b', label='Mean')
                plt.fill_between(test_x.numpy().squeeze(), 
                    mean.numpy() - 2 * variance.sqrt().numpy(), 
                    mean.numpy() + 2 * variance.sqrt().numpy(), 
                    alpha=0.3, color='blue')
                plt.scatter(self.x, -self.y[:, i], color='blue')
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title(f'{'RGPE' if self.is_rgpe else 'Regular GP'} objective {i+1}')
                plt.legend()
                plt.show()


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
        if self.is_rgpe:
            save_data = {
                'model': self.model.state_dict(),
                'weights': [self.model.models[0].weights, self.model.models[1].weights],

            }
            torch.save(save_data, model_path) #type: ignore
        else:
            torch.save(self.model.state_dict(), model_path)
        data_save = save_iter_path + '/data.csv'
        x = self.x.detach().cpu().numpy()
        y = self.y.detach().cpu().numpy()
        full_data = np.hstack((x,y))
        np.savetxt(data_save, full_data)
        print(f"model saved successfully at {save_iter_path}")
    
   
    

