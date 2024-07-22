import math
import os
import torch
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qNoisyExpectedImprovement
from botorch.acquisition.analytic import ProbabilityOfImprovement
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Interval
from botorch.sampling import IIDNormalSampler
from botorch.optim import optimize_acqf
from gpytorch.kernels import RBFKernel,ScaleKernel
from gpytorch.priors import NormalPrior
from HIL.optimization.RGPE_model import RGPE
from botorch.sampling.samplers import SobolQMCNormalSampler

# local imports
from HIL.optimization.kernel import SE, Matern 

import numpy as np
import matplotlib.pyplot as plt

# utils
import logging
from typing import Any, Optional, Tuple, Dict

    

import warnings
warnings.filterwarnings("ignore")


class BayesianOptimization(object):
    """
    Bayesian Optimization class for HIL
    """
    def __init__(self, n_parms:int = 1, range: np.ndarray = np.array([0,10]), noise_range :np.ndarray = np.array([0.005, 10]), acq: str = "ei", maximization : bool = True, \
        Kernel: str = "SE", model_save_path : str = "", device : str = "cpu" , plot: bool = False, optimization_iter: int = 500 , kernel_parms: Dict = {}, is_rgpe: bool = False) -> None:
        """Bayesian optimization for HIL

        Args:
            n_parms (int, optional): Number of optimization parameters ( exoskeleton parameters). Defaults to 1.
            range (np.ndarray, optional): Range of the optimization parameters. Defaults to np.array([0,1]).
            noise_range (np.ndarray, optional): Range of noise contraints for optimization. Defaults to np.array([0.005, 10]).
            acq (str, optional): Selecting acquisition function, options are 'ei', 'pi'. Defaults to "ei".
            Kernel (str, optional): Selecting kernel for the GP, options are "SE", "Matern". Defaults to "SE".
            model_save_path (str, optional): Path the new optimization saving directory. Defaults to "".
            device (str, optional): which device to perform optimization, "gpu", "cuda" or "cpu". Defaults to "cpu".
            plot (bool, optional): options to plot the gp and acquisition points. Defaults to False.
        """
        # TODO have an options of sending in the kernel parameters.
        if Kernel == "SE":
            self.kernel = SE(n_parms)
            self.covar_module = self.kernel.get_covr_module()

        else:
            self.kernel = Matern(n_parms)
            self.covar_module = self.kernel.get_covr_module()
        
        self.n_parms = n_parms
        self.range = range
        self.norm_range = np.array([0,1]).reshape(2,1).astype(float) # normalized range of parameters (if Normalization is True in the config file)
        self.maximization = maximization
        
        if len(model_save_path):
            self.model_save_path = model_save_path
        else:
            # this is temp
            self.model_save_path = "tmp_data/"

        self.optimization_iter = optimization_iter

        # place holder for model
        self.model = None

        # place to store the parameters
        self.x = torch.tensor([])
        self.y = torch.tensor([])

        # device 
        self.device = device

        # plotting
        self.PLOT = plot

        # logging
        self.logger = logging.getLogger()

        # Noise constraints
        self._noise_constraints = noise_range 
        self.likelihood = GaussianLikelihood() #noise_constraint=Interval(self._noise_constraints[0], self._noise_constraints[1]))

        # number of sampling in the acquisition points
        self.N_POINTS = 200

        # acquisition function type
        self.acq_type = acq

        if self.n_parms == 2:
            self.fig = plt.figure(figsize = (12,10))
            self.ax = plt.axes(projection='3d')

        self.is_rgpe = is_rgpe
        
        if self.is_rgpe:
            self.base_model_list = self.create_base_models()
            print('\n\n length of base model list: ', len(self.base_model_list))

        
    def create_base_models(self, base_model_path = "base_models", all_positive:bool = False):
        print("Creating base models")
        print(base_model_path)
        print("Base models:",os.listdir(base_model_path))
        model_list = []
        for i in os.listdir(base_model_path):
            #csv_files = self.find_csv_filenames("base_models/"+i)
            with open(base_model_path+"/"+i+"/data.csv") as f:
                data = np.loadtxt(f)
            x = torch.tensor([data[:,self.n_parms]]).to(self.device)
            y = torch.tensor(data[:,-1].reshape(-1,1)).to(self.device)
            print(f'x: {x.shape} \ny: {y.shape}')
            if all_positive:
                if any(y < 0):
                    y = -y
            print(x.shape,y.shape)
            print(y)
            #model_weights = torch.load("base_models/"+i+"/model.pth")
            #model=SingleTaskGP(x,y)
            #model.load_state_dict(torch.load("base_models/"+i+"/model.pth" ))
            model = self.get_fitted_model(x,y,state_dict =None,dimension=self.n_parms)
            model_list.append(model)

        return model_list
    
    def get_fitted_model(self,train_X, train_Y, state_dict=None,dimension=1,target_cv=None):
        """
        Get a single task GP. The model will be fit unless a state_dict with model 
            hyperparameters is provided.
        """
        Y_mean = train_Y.mean(dim=-2, keepdim=True)
        Y_std = train_Y.std(dim=-2, keepdim=True)

        if state_dict is None:
            covar_module  = ScaleKernel(
                    base_kernel=RBFKernel(ard_num_dims=dimension, lengthscale_prior=NormalPrior(0.4,0.56))
                    ,outputscale_prior=NormalPrior(0.2,0.56))
                    
            model = SingleTaskGP(train_X, train_Y,covar_module=covar_module)
            model.likelihood=GaussianLikelihood(noise_prior=NormalPrior(0.1,0.56),noise_constraint=Interval(0.00001,0.5))
            model.Y_mean = Y_mean
            model.Y_std = Y_std
            self._training(model,model.likelihood,train_X, (train_Y - Y_mean)/Y_std)
            # print("model trained")
            # mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
            # fit_gpytorch_model(mll)
        else:
            if target_cv:
                batch_shape = [train_X.shape[0]]#torch.tensor(train_X.shape[0]).view(-1)
                print(batch_shape)
                model = SingleTaskGP(train_X, (train_Y - Y_mean)/Y_std,covar_module=ScaleKernel(batch_shape=batch_shape,
                        base_kernel=RBFKernel(ard_num_dims=dimension, batch_shape=batch_shape,lengthscale_prior=NormalPrior(0.4,0.56))
                        ,outputscale_prior=NormalPrior(0.2,0.56)))
                model.likelihood=GaussianLikelihood(noise_prior=NormalPrior(0.1,0.56),noise_constraint=Interval(0.00001,0.5),batch_shape=batch_shape)
                model.Y_mean = Y_mean
                model.Y_std = Y_std 
                model.load_state_dict(state_dict)
            else:
                model = SingleTaskGP(train_X, (train_Y - Y_mean)/Y_std,covar_module=ScaleKernel(
                        base_kernel=RBFKernel(ard_num_dims=dimension,lengthscale_prior=NormalPrior(0.4,0.56))
                        ,outputscale_prior=NormalPrior(0.2,0.56)))
                model.likelihood=GaussianLikelihood(noise_prior=NormalPrior(0.1,0.56),noise_constraint=Interval(0.00001,0.5))
                model.Y_mean = Y_mean
                model.Y_std = Y_std 
                model.load_state_dict(state_dict)
                
        model.to(self.device)
        return model

    def _step(self) -> np.ndarray:
        """ Fit the model and identify the next parameter, also plots the model if plot is true

        Returns:
            np.ndarray: Next parameter to sampled
        """

        parameter, value = self._fit()
        new_parameter = parameter.detach().cpu().numpy()

        self.logger.info(f"Next parameter is {new_parameter}")

        self._save_model()

        if self.PLOT:
            if self.n_parms == 1:
                self._plot()
            elif self.n_parms == 2:
                self._plot2d()

        return new_parameter

    def _get_data_best(self) -> float:
        """Get the best value predicted by the model

        Returns:
            float: best value
        """
        
        range = np.arange(self.range[0,:], self.range[1,:], self.N_POINTS)
        range = torch.tensor(range)
        self.model.eval() #type: ignore
        output = self.model(range)     #type: ignore
        return torch.max(output).detach().numpy() #type: ignore
    
    def _training(self, model, likelihood,train_x,train_y):

        """
        Train the model using Adam Optimizer and gradient descent
        Log Marginal Likelihood is used as the cost function
        """
           
        parameter = list(model.parameters()) + list(likelihood.parameters())
        optimizer = torch.optim.Adam(parameter, lr=0.01) 
        mll= ExactMarginalLogLikelihood(likelihood, model).to(train_x)
        

        train_y=train_y.squeeze(-1)
        loss = -mll(model(train_x), train_y) #type: ignore
        self.logger.info("before training Loss: ", loss.item())
        for i in range(500):
            
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y) #type: ignore
            
            loss.backward()
            optimizer.step()
        self.logger.info("after training Loss: ", loss.item()) 

    def _fit(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Using the model and likelihood select the next data point to get next data points and acq value at that point

        Returns:
            Tuple[torch.tensor, torch.tensor]: next parmaeter, value at the point
        """
        # tradition method
        # mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        # fit_gpytorch_model(mll) # check I need to change anything
        # using manual gradient descent using adam optimizer.
        self._training(self.model, self.likelihood, self.x, self.y)


        if self.acq_type == "ei":
            acq = qNoisyExpectedImprovement(self.model, self.x, sampler=IIDNormalSampler(self.N_POINTS, seed = 1234)) #type: ignore
        else:
            # TODO add other acquisition functions
            best_f = self._get_data_best()
            acq = ProbabilityOfImprovement(self.model, best_f, sampler=IIDNormalSampler(self.N_POINTS, seed = 1234)) #type: ignore
        pass
        new_point, value  = optimize_acqf(
            acq_function = acq,
            bounds=torch.tensor(self.range).to(self.device),
            # bounds=torch.tensor(self.norm_range).to(self.device),
            q = 1,
            num_restarts=1000,
            raw_samples=2000,
            options={},
        )
        return new_point, value

    # Temp function will be replaced is some way
    def _plot(self) -> None:
        plt.cla()
        x = self.x.detach().numpy()
        y = self.y.detach().numpy()
        plt.plot(x, y, 'r.', ms = 10)
        x_torch = torch.tensor(x).to(self.device)
        y_torch = torch.tensor(y).to(self.device)
        self.model.eval()  #type: ignore
        self.likelihood.eval()
        with torch.no_grad():
            x_length = np.linspace(self.range[0,0],self.range[1,0],100).reshape(-1, 1)
            # print(x_length,self.range)
            observed = self.likelihood(self.model(torch.tensor(x_length))) #type: ignore
            observed_mean = observed.mean.cpu().numpy() #type: ignore
            upper, lower = observed.confidence_region() #type: ignore
            # x_length = x_length.cpu().numpy()
            
        plt.plot(x_length.flatten(), observed_mean)
        plt.fill_between(x_length.flatten(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.2)
        plt.legend(['Observed Data', 'mean', 'Confidence'])
        plt.pause(0.01)

    # Temp function will be replaced is some way
    def _plot2d(self) -> None:
        model=self.model
        model.eval() #type: ignore
        likelihood=self.likelihood
        likelihood.eval()
        self.ax.clear()
        x = self.x.detach().numpy()
        y = self.y.detach().numpy()
        y_mean = y.mean()
        y_std = y.std()
        with torch.no_grad():
            test_x = torch.linspace(self.range[0,0],self.range[1,0], 51).to(self.device)
            test_y = torch.linspace(self.range[0,1],self.range[1,1],51).to(self.device)
            XX,YY=torch.meshgrid(test_x,test_y,indexing='xy')
            #print(test_x.shape)
            XXX=torch.cat((XX.reshape(-1,1),YY.reshape(-1,1)),dim=1).double()
            observed_pred = likelihood(model(XXX)) #type: ignore

            # # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region() #type: ignore
            model_mean = observed_pred.mean*y_std + y_mean
            lower=lower*y_std + y_mean
            upper=upper*y_std + y_mean
            ZZ=model_mean.view(51,51)
            
            if max is False:
                ZZ = -ZZ
                lower = -lower
                upper = -upper
                y = -self.y 

            self.ax.plot_surface(XX.cpu().numpy(), YY.cpu().numpy(), ZZ.cpu().numpy(), cmap = 'winter',alpha=0.9) #type: ignore
            #print(XXX[:,0].shape,XXX[:,1].shape,lower1.shape)
            self.ax.plot_trisurf(XXX[:,0].cpu().numpy(), XXX[:,1].cpu().numpy(), lower.view(-1).cpu().numpy(),  #type: ignore
                    linewidth = 0.2,
                    antialiased = True,color='gainsboro',alpha=0.4,edgecolor='gainsboro') 
            self.ax.plot_trisurf(XXX[:,0].cpu().numpy(), XXX[:,1].cpu().numpy(), upper.view(-1).cpu().numpy(), #type: ignore
                    linewidth = 0.2,
                    antialiased = True,color='gainsboro',alpha=0.4,edgecolor='gainsboro')
            # find the location of minimum value
            print(ZZ.cpu().numpy().shape, XX.cpu().numpy().shape, YY.cpu().numpy().shape)
            ZZ = ZZ.cpu().numpy()
            XX = XX.cpu().numpy()
            YY = YY.cpu().numpy()
            min_index = np.unravel_index(ZZ.argmin(), ZZ.shape)
            logging.info(f"Min value is: {ZZ[min_index]}")
            logging.info(f"Min location (Timing) is: {XX[min_index]}")
            logging.info(f"Min location (Torque) is: {YY[min_index]}")
            self.min_location = np.array([XX[min_index], YY[min_index]])
            self.min_value = ZZ[min_index]

            # finding max value
            max_index = np.unravel_index(ZZ.argmax(), ZZ.shape)
            logging.info(f"Max value is: {ZZ[max_index]}")
            logging.info(f"Max location (Timing) is: {XX[max_index]}")
            logging.info(f"Max location (Torque) is: {YY[max_index]}")
            self.max_location = np.array([XX[max_index], YY[max_index]])
            self.max_value = ZZ[max_index]
            self.ax.scatter(x[:,0],x[:,1],y,color='r',marker='o',s=100,alpha=1) #type: ignore
            self.ax.view_init(25, -135) #type: ignore
            self.ax.set_zlabel("Cost",fontsize=18,rotation=90) #type: ignore
            self.ax.set_xlabel("Timing",fontsize=16)
            self.ax.set_ylabel("Torque",fontsize=16)
            # plt.pause(0.01)

    def _save_model(self) -> None:
        """Save the model and data in the given path
        """
        save_iter_path = self.model_save_path + f'iter_{len(self.x)}'
        os.makedirs(save_iter_path, exist_ok=True)
        model_path = save_iter_path +'/model.pth'
        torch.save(self.model.state_dict(), model_path) #type: ignore
        data_save = save_iter_path + '/data.csv'
        x = self.x.detach().cpu().numpy()
        y = self.y.detach().cpu().numpy()
        full_data = np.hstack((x,y))
        np.savetxt(data_save, full_data)
        self.logger.info(f"model saved successfully at {save_iter_path}")
    
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

    def run(self, x: np.ndarray, y: np.ndarray, reload_hyper: bool  = False ) -> np.ndarray:
        """Run the optimization with input data points

        Args:
            x (NxM np.ndarray): Input parameters N -> n_parms, M -> iter
            y (Mx1): Cost function array
            reload_hyper (bool, optional): Reload the hyper parameter trained in the previous iter. Defaults to True.

        Returns:
            np.ndarray: parameter to sample next
        """

        
        assert len(x) == len(y), "Length should be equal."

        self.x = torch.tensor(x).to(self.device)
        self.y = torch.tensor(y).to(self.device)

        if not reload_hyper:
            self.kernel.reset()
            self.likelihood = GaussianLikelihood(noise_constraint = Interval(self._noise_constraints[0], self._noise_constraints[1]))
            self.model = SingleTaskGP(self.x, self.y, likelihood = self.likelihood, covar_module = self.kernel.get_covr_module()) 
            # TODO check if this ok for multi dimension models
            if self.is_rgpe:
                model_list = self.base_model_list[i] + [self.model]
                weights = self.compute_weights(self.x, self.y.unsqueeze(-1), self.base_model_list, self.model, 256, self.device)
                print(f'\n\n\n the final weights are: {weights}')
                self.model = RGPE(model_list, weights)
            self.model.to(self.device)

        else:
            # keeping the likehood save and kernel parameters so no need to reset those
            self.model = SingleTaskGP(self.x, self.y, likelihood = self.likelihood, covar_module = self.kernel.get_covr_module())
            if self.is_rgpe:
                model_list = self.base_model_list[i] + [self.model]
                weights = self.compute_weights(self.x, self.y.unsqueeze(-1), self.base_model_list, self.model, 256, self.device)
                print(f'\n\n\n the final weights are: {weights}')
                self.model = RGPE(model_list, weights)
            self.model.to(self.device)

        # fit the model and get the next parameter.
        new_parameter = self._step()
        
        return new_parameter
        


if __name__ == "__main__":


    def mapRange(value, inMin=0, inMax=1, outMin=0.2, outMax=1.2):
        return outMin + (((value - inMin) / (inMax - inMin)) * (outMax - outMin))
    # objective function
    def objective(x, noise_std=0.0):
        # define range for input
        # r_min, r_max = 0, 1.2
        x = mapRange(x/100)
        return 	 -(1.4 - 3.0 * x) * np.sin(18.0 * x) + noise_std * np.random.random(len(x))
    # BO = BayesianOptimization(range = np.array([-0.5 * np.pi, 2 * np.pi]))
    BO = BayesianOptimization(range = np.array([0, 100]), plot=True)
    x = np.random.random(3) * 100

    y = objective(x)

    x = x.reshape(-1,1)
    y = y.reshape(-1, 1)
    
    full_x = np.linspace(0,100,100)
    full_y = objective(full_x)

    plt.plot(full_x, full_y)
    plt.plot(x,y, 'r*')
    plt.show()


    new_parameter = BO.run(x, y)
    # length_scale_store = []
    # output_scale_store = []
    # noise_scale_store = []
    for i in range(15):
        x = np.concatenate((x, new_parameter.reshape(1,1)))

        y = np.concatenate((y,objective(new_parameter).reshape(1,1)))
        new_parameter = BO.run(x,y)
        plt.pause(2)
    #     plt.show()
    #     # length_scale_store.append(BO.model.covar_module.base_kernel.lengthscale.detach().numpy().flatten())
    #     # output_scale_store.append(BO.model.covar_module.outputscale.detach().numpy().flatten())
    #     # noise_scale_store.append(BO.likelihood.noise.detach().numpy().flatten()
    #     # )
    # # x = np.linspace(-0.5 * np.pi, 2 * np.pi, 100)
    # x = np.linspace(0, 100, 100)
    # print('here')
    # plt.figure()
    # y = (np.sin(x/100)**3 + np.cos(x/100)**3) * 100

    # plt.plot(x,y,'r')
    # plt.figure()
    # plt.plot(length_scale_store, label='length scale')
    # plt.plot(output_scale_store, label = "sigma")
    # plt.plot(noise_scale_store, label="noise")
    # plt.legend()
    # plt.show()
