import torch
from botorch.sampling.samplers import SobolQMCNormalSampler
import matplotlib.pyplot as plt
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.models import SingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_model
from gpytorch.kernels import RBFKernel,ScaleKernel
from gpytorch.priors import NormalPrior
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints import Interval
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from HIL.optimization.RGPE_model import RGPE
import os
import numpy as np
from tqdm import tqdm
import yaml

class create_rgpe():

    def __init__(self,dimension,device, base_model_path = "base_models", all_positive:bool = False):
        
        self.device = device
        self.num_posteriors = 256
        self.train_iter = 100
        self.lrate = 0.005
        self.n_parms = dimension
        self.x = torch.tensor([])
        self.y = torch.tensor([])
        self.base_model_list = self.create_base_models(base_model_path=base_model_path, all_positive=all_positive)
        print(len(self.base_model_list))

    # def find_csv_filenames(self,path_to_dir, suffix=".csv" ):
    #     filenames = os.listdir(path_to_dir)
    #     return [ filename for filename in filenames if filename.endswith( suffix ) ]


    def create_base_models(self, base_model_path = "base_models", all_positive:bool = False):
        print("Creating base models")
        print(base_model_path)
        print("Base models:",os.listdir(base_model_path))
        model_list = []
        for i in os.listdir(base_model_path):
            #csv_files = self.find_csv_filenames("base_models/"+i)
            with open(base_model_path+"/"+i+"/data.csv") as f:
                data = np.loadtxt(f)
            # x = torch.tensor(data[:,:-1]).to(self.device)
            # y = torch.tensor(data[:,-1].reshape(-1,1)).to(self.device)
            
            x = torch.tensor(data[:, :self.n_parms]).to(self.device)
            y = torch.tensor(data[:, self.n_parms:]).to(self.device)
            
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

    def roll_col(self,X, shift):  
        """
        Rotate columns to right by shift.
        """
        return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)
    


    def training(self,model, likelihood,train_x,train_y):

        """
        Train the model using Adam Optimizer and gradient descent
        Log Marginal Likelihood is used as the cost function
        # """
        # print(model.parameters())
        # for name, param in model.named_parameters():
        #     print(name, param.shape)
        # print(likelihood.parameters())
        # for name, param in likelihood.named_parameters():
        #     print(name, param.shape)
        optimization_parms = list(model.parameters()) + list(likelihood.parameters())
        # print(optimization_parms)
        optimizer = torch.optim.Adam(optimization_parms, lr=self.lrate) 
        model.to(self.device)
        # mll= ExactMarginalLogLikelihood(likelihood, model).to(train_x)
        mll = SumMarginalLogLikelihood(likelihood, model)

        train_y=train_y.squeeze(-1)
        loss = -mll(model(train_x), train_y)
        print("before loss",loss.item())
        for i in range(self.train_iter):
            
            optimizer.zero_grad()
            output = model(train_x)
            loss = -mll(output, train_y)
            #print("losss",loss.shape)
            loss.backward()
            optimizer.step()
            # print(list(likelihood.parameters())[0].cpu().detach().numpy())
        print("after loss",loss.item())



    def get_fitted_model(self,train_X, train_Y, state_dict=None,dimension=1,target_cv=None):
        """
        Get a single task GP. The model will be fit unless a state_dict with model 
            hyperparameters is provided.
        """
        # Y_mean = train_Y.mean(dim=-2, keepdim=True)
        # Y_std = train_Y.std(dim=-2, keepdim=True)

        if state_dict is None:
            covar_module  = ScaleKernel(
                    base_kernel=RBFKernel(ard_num_dims=dimension)) #lengthscale_prior=NormalPrior(0.4,0.56))
                    #,outputscale_prior=NormalPrior(0.2,0.56))
                    
            # model = SingleTaskGP(train_X, (train_Y - Y_mean)/Y_std,covar_module=covar_module)
            # model.likelihood=GaussianLikelihood(noise_prior=NormalPrior(0.1,0.56),noise_constraint=Interval(0.00001,0.5))
            # model.Y_mean = Y_mean
            # model.Y_std = Y_std

            models = []
            for i in range(train_Y.shape[-1]):
                train_objective = train_Y[:, i]
                models.append(
                    SingleTaskGP(train_Y, train_objective.unsqueeze(-1), covar_module=covar_module)
                )
            model = ModelListGP(*models)
            # mll = SumMarginalLogLikelihood(model.likelihood, model)
            self.training(model,model.likelihood,train_X, train_Y)
            # print("model trained")
            # mll = ExactMarginalLogLikelihood(model.likelihood, model).to(train_X)
            # fit_gpytorch_model(mll)
        else:
            if target_cv:
                batch_shape = [train_X.shape[0]]#torch.tensor(train_X.shape[0]).view(-1)
                print(batch_shape)
                covar_module  = ScaleKernel(batch_shape=batch_shape,
                    base_kernel=RBFKernel(ard_num_dims=dimension, batch_shape=batch_shape)) #lengthscale_prior=NormalPrior(0.4,0.56))
                    #,outputscale_prior=NormalPrior(0.2,0.56))
                    
                # model = SingleTaskGP(train_X, (train_Y - Y_mean)/Y_std,covar_module=covar_module)
                # model.likelihood=GaussianLikelihood(noise_prior=NormalPrior(0.1,0.56),noise_constraint=Interval(0.00001,0.5))
                # model.Y_mean = Y_mean
                # model.Y_std = Y_std

                models = []
                for i in range(train_Y.shape[-1]):
                    train_objective = train_Y[:, i]
                    models.append(
                        SingleTaskGP(train_Y, train_objective.unsqueeze(-1), covar_module=covar_module)
                    )
                model = ModelListGP(*models)
                model.load_state_dict(state_dict)
            else:
                covar_module  = ScaleKernel(
                    base_kernel=RBFKernel(ard_num_dims=dimension)) #lengthscale_prior=NormalPrior(0.4,0.56))
                    #,outputscale_prior=NormalPrior(0.2,0.56))
                    
                # model = SingleTaskGP(train_X, (train_Y - Y_mean)/Y_std,covar_module=covar_module)
                # model.likelihood=GaussianLikelihood(noise_prior=NormalPrior(0.1,0.56),noise_constraint=Interval(0.00001,0.5))
                # model.Y_mean = Y_mean
                # model.Y_std = Y_std

                models = []
                for i in range(train_Y.shape[-1]):
                    train_objective = train_Y[:, i]
                    models.append(
                        SingleTaskGP(train_Y, train_objective.unsqueeze(-1), covar_module=covar_module)
                    )
                model = ModelListGP(*models) 
                model.load_state_dict(state_dict)
                
        model.to(self.device)
        return model


    def compute_ranking_loss(self, f_samps, target_y):
        """
        Compute ranking loss for each sample from the posterior over multi-dimensional target points.
        
        Args:
            f_samps: `n_samples x n x n`-dim tensor of samples where n is the number of data points.
            target_y: `n x d`-dim tensor of targets where d is the number of target dimensions.
        Returns:
            Tensor: `n_samples x d`-dim tensor containing the ranking loss across each sample for each dimension.
        """
        n, d = target_y.shape

        if f_samps.ndim == 3:
            # Extend f_samps along a new dimension for each target dimension
            extended_f_samps = f_samps.unsqueeze(-1).expand(-1, -1, -1, d)

            # Extend target_y along new dimensions to match f_samps expanded dimensions for broadcasting
            extended_target_y = target_y.unsqueeze(0).unsqueeze(0).expand(n_samples, n, n, d)

            # Compute ranking loss for each target dimension
            rank_losses = []
            for i in range(d):
                cartesian_y = torch.cartesian_prod(
                    target_y[:, i], target_y[:, i]
                ).view(n, n, 2)
                
                rank_loss = (
                    (extended_f_samps[:, :, :, i].diagonal(dim1=1, dim2=2).unsqueeze(-1) < extended_f_samps[:, :, :, i]) ^
                    (cartesian_y[..., 0] < cartesian_y[..., 1])
                ).sum(dim=-1).sum(dim=-1)

                rank_losses.append(rank_loss)

            # Combine ranking losses across all dimensions
            # Here, we stack and sum, but you can also average or apply other reduction methods
            total_rank_loss = torch.stack(rank_losses, dim=-1).sum(dim=-1)
        else:
            total_rank_loss = torch.zeros((f_samps.shape[0], d), dtype=torch.long, device=target_y.device)
            for dim in range(d):
                y_stack = target_y[:, dim].unsqueeze(1).expand(-1, n)
                for i in range(1, n):
                    rank_loss = (
                        (self.roll_col(f_samps, i) > f_samps) ^
                        (self.roll_col(y_stack, i) > y_stack)
                    ).sum(dim=-1)
                    total_rank_loss[:, dim] += rank_loss

        return total_rank_loss


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
        model = self.get_fitted_model(train_x_cv, train_y_cv, state_dict=state_dict_expanded,dimension=self.n_parms,target_cv=True)
        with torch.no_grad():
            posterior = model.posterior(train_x)
            # Since we have a batch mode gp and model.posterior always returns an output dimension,
            # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
            # the last dimension.
            sampler = SobolQMCNormalSampler(num_samples=num_samples)
            return sampler(posterior).squeeze(-1)
    


    def compute_rank_weights(self,train_x,train_y, base_models, target_model, num_samples, device):
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


    def run(self,x,y):
        
        ## function will fit the target model and create the RGPE model given the base models 
        self.x = x
        self.y = y
        self.target_model = self.get_fitted_model(self.x,self.y,state_dict=None,dimension=self.n_parms)
        self.model_list = self.base_model_list + [self.target_model]
        self.rank_weights = self.compute_rank_weights(self.x,self.y,self.base_model_list,self.target_model,self.num_posteriors,self.device)
        
        self.rgpe_model = RGPE(self.model_list,self.rank_weights)
        return self.rgpe_model,self.target_model

