

from cProfile import label
import math
import os
# from types import NoneType
import torch
import numpy as np
import matplotlib.pyplot as plt

from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_model
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.acquisition import ExpectedImprovement, qExpectedImprovement, qNoisyExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling import SobolQMCNormalSampler, IIDNormalSampler
from gpytorch.kernels import ScaleKernel, RBFKernel, LinearKernel
from gpytorch.constraints import GreaterThan, Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.priors import NormalPrior

import warnings
warnings.filterwarnings("ignore")
import gpytorch
# use a GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
from rgpe_functons import create_rgpe


class BayesianOptimization_rgpe(object):
    """
    Bayesian optimization
    """
    def __init__(self, n_params:int = 1, range: np.ndarray = np.array([2.7, 7.5]),   \
         length_constraint: np.ndarray = np.array([0, 10]), noise_constraint: np.ndarray = np.array([0, 10]) , 
         variance_constrain: np.ndarray = np.array([0, 10]), 
         model_save_path:str = None, device: str = "cpu", hyper_params_start:
          dict = {}, base_model_path:str = "base_models", all_positive:bool = False ) -> None:
        """
        initializing the model with hyperparameters
        """
        self.n_parms = n_params
        # This is for walking
        # self.range = range.reshape(self.n_parms * 2,1).astype(float)
        self.range = range.reshape(2, self.n_parms)
        # self.covar_module = ScaleKernel(RBFKernel(lenghtscale_constraint = Interval(length_constraint[0], length_constraint[1]))+ \
        # LinearKernel(variance_constraint=Interval(variance_constrain[0], variance_constrain[1])))
        self.covar_module = ScaleKernel(base_kernel=RBFKernel(lengthscale_prior=NormalPrior(0.4,0.56)),outputscale_prior=NormalPrior(0.2,0.56))
        self.likelihood = GaussianLikelihood(noise_prior=NormalPrior(0.1,0.2),noise_constraint=Interval(0.05,0.5))

        # place holder for model
        self.model = None

        # storing the x and y used for optimization.
        self.x = torch.tensor([])
        self.y = torch.tensor([])
        
        # check which device they want to use
        self.device = device
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.rgpe_BO = create_rgpe(self.n_parms,self.device, base_model_path=base_model_path, all_positive=all_positive)
        self.hyper = hyper_params_start
        self.model_save_path = model_save_path
        # if self.model_save_path != 'tmp/':
        self.fig = plt.figure(figsize = (12,10))
        self.ax = plt.axes(projection='3d')


    def _step(self, plot: bool = False) -> np.ndarray:
        """ Add the x and y to the sensor array and return the next parameters to perform.
        """

        parameter = self._acq()
        new_parameters = parameter.detach().cpu().numpy()
        print(f"Next parameters is {new_parameters}")

        if type(self.model_save_path) != type(None):
            # pass
            self._save_model_data()
        return new_parameters

    def _save_model_data(self):
        save_iter_path = self.model_save_path + f'iter_{self.n}'
        # if not os.path.exists(save_iter_path):
        os.makedirs(save_iter_path, exist_ok=True)
        model_path = save_iter_path +'/model.pth'
        torch.save(self.model.state_dict(), model_path)
        data_save = save_iter_path + '/data.csv'
        x = self.x.detach().cpu().numpy()
        y = self.y.detach().cpu().numpy()
        full_data = np.hstack((x,y))
        np.savetxt(data_save, full_data)
        print(f"model saved successfully at {save_iter_path}")


    def run(self, x: np.ndarray, y: np.ndarray, n:int = 0, plot:bool = True, maximize:bool = True) -> np.ndarray:
        """
        Start the optimization from stratch.
        X: initial input array (n x n_parms) (n > 2)
        Y: initial outup to the input array (n x 1) (n > 2)
        """
        # assert x.shape[1] == self.n_parms and x.shape[0] == y.shape[0], "Shape does not match"
        self.x = torch.tensor(x).to(self.device)
        self.y = torch.tensor(y).to(self.device)
        self.n = n
        self.y_var = torch.rand_like(self.y).to(self.device) *  0.01
        self.covar_module = ScaleKernel(base_kernel=RBFKernel(lengthscale_prior=NormalPrior(0.4,0.56)),outputscale_prior=NormalPrior(0.2,0.56))
        self.likelihood = GaussianLikelihood(noise_prior=NormalPrior(0.1,0.2),noise_constraint=Interval(0.05,0.5))
        
        self.rgpe_model,self.model = self.rgpe_BO.run(self.x,self.y)
        new_parameter = self._step()

        if plot:
            if self.n_parms == 1:
                self.plot(x,(y-np.mean(y))/np.std(y))
            elif self.n_parms == 2:
                self.plotting3d(x,y,maximize)
            else:
                pass

        return new_parameter


    def _acq(self, mc_sampling : int = 512):
        """ use acquisition function to give the next paprameter to observe
        """
        # ei = ExpectedImprovement(self.model, best_f=self.y.max())
        # ei = qExpectedImprovement(self.model, best_f=self.y.max(), sampler=SobolQMCNormalSampler(mc_sampling, seed = 1234))
        qNei = qNoisyExpectedImprovement(self.rgpe_model, self.x, sampler=SobolQMCNormalSampler(mc_sampling, seed = 1234))
        new_point, _ = optimize_acqf(
            acq_function=qNei,
            bounds=torch.tensor(self.range).to(self.device),
            q = 1,
            num_restarts=100,
            raw_samples=512,
            options={},
        )
        return new_point


    def plotting3d(self,x,y,max):
        model=self.model
        model.eval()
        likelihood=model.likelihood
        likelihood.eval()
        self.ax.clear()
        with torch.no_grad():
            test_x = torch.linspace(self.range[0,0],self.range[1,0], 51).to(self.device)
            test_y = torch.linspace(self.range[0,1],self.range[1,1],51).to(self.device)
            XX,YY=torch.meshgrid(test_x,test_y,indexing='xy')
            #print(test_x.shape)
            XXX=torch.cat((XX.reshape(-1,1),YY.reshape(-1,1)),dim=1).double()
            observed_pred = likelihood(model(XXX))

            # # Get upper and lower confidence bounds
            lower, upper = observed_pred.confidence_region()
            model_mean = observed_pred.mean*model.Y_std + model.Y_mean
            lower=lower*model.Y_std + model.Y_mean
            upper=upper*model.Y_std + model.Y_mean
            ZZ=model_mean.view(51,51)
            
            if max is False:
                ZZ = -ZZ
                lower = -lower
                upper = -upper
                y = -y 

            self.ax.plot_surface(XX.cpu().numpy(), YY.cpu().numpy(), ZZ.cpu().numpy(), cmap = 'winter',alpha=0.9)
            #print(XXX[:,0].shape,XXX[:,1].shape,lower1.shape)
            self.ax.plot_trisurf(XXX[:,0].cpu().numpy(), XXX[:,1].cpu().numpy(), lower.view(-1).cpu().numpy(),
                    linewidth = 0.2,
                    antialiased = True,color='gainsboro',alpha=0.4,edgecolor='gainsboro')
            self.ax.plot_trisurf(XXX[:,0].cpu().numpy(), XXX[:,1].cpu().numpy(), upper.view(-1).cpu().numpy(),
                    linewidth = 0.2,
                    antialiased = True,color='gainsboro',alpha=0.4,edgecolor='gainsboro')
            # find the location of minimum value
            print(ZZ.cpu().numpy().shape, XX.cpu().numpy().shape, YY.cpu().numpy().shape)
            ZZ = ZZ.cpu().numpy()
            XX = XX.cpu().numpy()
            YY = YY.cpu().numpy()
            min_index = np.unravel_index(ZZ.argmin(), ZZ.shape)
            print(f"Min value is: {ZZ[min_index]}")
            print(f"Min location (Timing) is: {XX[min_index]}")
            print(f"Min location (Torque) is: {YY[min_index]}")
            self.min_location = np.array([XX[min_index], YY[min_index]])
            self.min_value = ZZ[min_index]

            # finding max value
            max_index = np.unravel_index(ZZ.argmax(), ZZ.shape)
            print(f"Max value is: {ZZ[max_index]}")
            print(f"Max location (Timing) is: {XX[max_index]}")
            print(f"Max location (Torque) is: {YY[max_index]}")
            self.max_location = np.array([XX[max_index], YY[max_index]])
            self.max_value = ZZ[max_index]
            if 'tmp' not in self.model_save_path and False:
                self.ax.scatter(x[:,0],x[:,1],y,color='r',marker='o',s=100,alpha=1)
            self.ax.view_init(25, -135)
            self.ax.set_zlabel("Cost",fontsize=18,rotation=90)
            self.ax.set_xlabel("Timing",fontsize=16)
            self.ax.set_ylabel("Torque",fontsize=16)
            # plt.pause(0.01)
            if ZZ[max_index] > 500:
                self.ax.set_zlim(200, 800) 
            else:
                self.ax.set_zlim(200, 500)
            if type(self.model_save_path) != type(None):
                save_iter_path = self.model_save_path + f'iter_{self.n}/'
                plt.savefig(save_iter_path+"figure.pdf")


    def plot(self, x, y, ax = None, reverse = False):
        #f , ax = plt.subplots(1,1,figsize=(4,3))
        if ax is not None:
            print("using ax")
            ax.cla()
            if reverse:
                ax.plot(x,y* -1, 'ro', ms = 10)
            else:
                ax.plot(x, y, 'r.', ms = 10)
            x_torch = torch.tensor(x).to(self.device)
            y_torch = torch.tensor(y).to(self.device)
            self.model.eval()
            self.likelihood.eval()
            with torch.no_grad():
                x_length = np.linspace(self.range[0,0],self.range[1,0],100).reshape(-1, 1)
                # print(x_length,self.range)
                observed = self.likelihood(self.model(torch.tensor(x_length)))
                observed_mean = observed.mean.cpu().numpy()
                upper, lower = observed.confidence_region()
                # x_length = x_length.cpu().numpy()
            
            if reverse:
                observed_mean = observed_mean * -1
                ax.plot(x_length, observed_mean)
                print("plotting reverse")
                print( observed_mean - observed.stddev.cpu().numpy() )
                #ax.fill_between(x_length.flatten(), lower.cpu().numpy() * -1 + 5, upper.cpu().numpy() * -1 - 5 , alpha=0.5)
                ax.fill_between(x_length.flatten(), observed_mean + observed.stddev.cpu().numpy() ,  observed_mean -  observed.stddev.cpu().numpy()  , alpha=0.5)
            else:   
                ax.plot(x_length.flatten(), observed_mean)
            # print(lower.cpu().numpy())
            # ax.fill_between(x_length.flatten(), lower.cpu().numpy() - 0.5 , upper.cpu().numpy() + 0.5 , alpha=0.2)
                ax.fill_between(x_length.flatten(), lower.cpu().numpy() - 0.3 , upper.cpu().numpy() + 0.3 , alpha=0.2)
            ax.legend(['Observed Data', 'mean', 'Confidence'])
            # plt.pause(0.01)
            # plt.savefig(f"figure_save/{len(self.y)}.pdf")
        else:
            # print(x,x.shape,y,y.shape)
            plt.cla()
            plt.plot(x.reshape(-1), y, 'r.', ms = 10)
            # x_torch = torch.tensor(x).to(self.device)
            # y_torch = torch.tensor(y).to(self.device)
            # self.model = self.model.to("cpu")
            # self.likelihood = self.likelihood.to("cpu")
            self.model.eval()
            self.likelihood.eval()
            with torch.no_grad():
                x_length = np.linspace(self.range[0,0],self.range[1,0],100).reshape(-1, 1)
                # print(x_length,self.range)
                observed = self.likelihood(self.model(torch.tensor(x_length).to(self.device)))
                #print(observed)
                observed_mean = observed.mean
                #upper, lower = observed.confidence_region()
                # x_length = x_length.cpu().numpy()
            
                
            plt.plot(x_length.reshape(-1), observed_mean.cpu().numpy().reshape(-1))
            #plt.fill_between(x_length.flatten(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.2)
            plt.legend(['Observed Data', 'mean', 'Confidence'])
            plt.pause(0.01)
            plt.savefig(f"figure_save/{len(self.y)}.pdf")


if __name__ == "__main__":
    # BO = BayesianOptimization(range = np.array([-0.5 * np.pi, 2 * np.pi]))
    BO = BayesianOptimization_rgpe(range = np.array([0, 100]))
    x = np.random.random(3) * 100

    y = np.sin(x / 100)**3 + np.cos(x / 100)**3

    y = y * 100 #+ np.random.random(y.shape) * 2
    x = x.reshape(-1,1)
    y = y.reshape(-1, 1)
    print(y.shape)
    new_parameter = BO.run(x, y, plot = True)
    length_scale_store = []
    output_scale_store = []
    noise_scale_store = []
    for i in range(15):
        x = np.concatenate((x, new_parameter.reshape(1,1)))

        y = np.sin(x / 100)**3 + np.cos(x / 100)**3 
        y = y * 100  #+ np.random.random(y.shape)
        y.reshape(-1,1)
        new_parameter = BO.run(x,y, plot=True)
        length_scale_store.append(BO.model.covar_module.base_kernel.lengthscale.detach().numpy().flatten())
        output_scale_store.append(BO.model.covar_module.outputscale.detach().numpy().flatten())
        noise_scale_store.append(BO.likelihood.noise.detach().numpy().flatten()
        )
    # x = np.linspace(-0.5 * np.pi, 2 * np.pi, 100)
    x = np.linspace(0, 100, 100)
    print('here')
    plt.figure()
    y = (np.sin(x/100)**3 + np.cos(x/100)**3) * 100

    plt.plot(x,y,'r')
    plt.figure()
    plt.plot(length_scale_store, label='length scale')
    plt.plot(output_scale_store, label = "sigma")
    plt.plot(noise_scale_store, label="noise")
    plt.legend()
    plt.show()
    













# Train_X, Train_Y = np.arange(0, 1, 0.1).reshape(-1,1), np.sin(np.arange(0, 1, 0.1)).reshape(-1,1)
# Train_X, Train_Y = torch.tensor(Train_X).to(device), torch.tensor(Train_Y).to(device)
# print(Train_X.shape, Train_Y.shape)
# covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lenghtscale_constraint = gpytorch.constraints.GreaterThan(0.01))+ gpytorch.kernels.LinearKernel())
# print(covar_module)
# likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.01))
# likelihood.to(device)
# model = SingleTaskGP(Train_X, Train_Y, covar_module=covar_module, likelihood=likelihood)
# hypers = {
#     'likelihood.noise_covar.noise': torch.tensor(1.),
#     'covar_module.base_kernel.kernels.0.lengthscale': torch.tensor(1.5),
#     'covar_module.base_kernel.kernels.1.variance': torch.tensor(1.0),
#     'covar_module.outputscale': torch.tensor(2.),
# }

# model.initialize(**hypers)Train_X, Train_Y = np.arange(0, 1, 0.1).reshape(-1,1), np.sin(np.arange(0, 1, 0.1)).reshape(-1,1)
# Train_X, Train_Y = torch.tensor(Train_X).to(device), torch.tensor(Train_Y).to(device)
# print(Train_X.shape, Train_Y.shape)
# covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(lenghtscale_constraint = gpytorch.constraints.GreaterThan(0.01))+ gpytorch.kernels.LinearKernel())
# print(covar_module)
# likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(0.01))
# likelihood.to(device)
# model = SingleTaskGP(Train_X, Train_Y, covar_module=covar_module, likelihood=likelihood)
# hypers = {
#     'likelihood.noise_covar.noise': torch.tensor(1.),
#     'covar_module.base_kernel.kernels.0.lengthscale': torch.tensor(1.5),
#     'covar_module.base_kernel.kernels.1.variance': torch.tensor(1.0),
#     'covar_module.outputscale': torch.tensor(2.),
# }

# model.initialize(**hypers)
# model.to(device)

# mll = ExactMarginalLogLikelihood(likelihood=likelihood, model=model)
# fit_gpytorch_model(mll)
# print(Train_Y.max())

# # ei = ExpectedImprovement(model, best_f=Train_Y.max())
# ei = qExpectedImprovement(model, best_f=Train_Y.max(), sampler=SobolQMCNormalSampler(1000))

# new_point, _ = optimize_acqf(
#     acq_function=ei,
#     bounds=torch.tensor(np.array([0.0, 1.0]).reshape(-1,1)).to(device),
#     q = 1,
#     num_restarts=20,
#     raw_samples=100,
#     options={},
# )


# print(new_point)

# model.to(device)

# mll = ExactMarginalLogLikelihood(likelihood=likelihood, model=model)
# fit_gpytorch_model(mll)
# print(Train_Y.max())

# # ei = ExpectedImprovement(model, best_f=Train_Y.max())
# ei = qExpectedImprovement(model, best_f=Train_Y.max(), sampler=SobolQMCNormalSampler(1000))

# new_point, _ = optimize_acqf(
#     acq_function=ei,
#     bounds=torch.tensor(np.array([0.0, 1.0]).reshape(-1,1)).to(device),
#     q = 1,
#     num_restarts=20,
#     raw_samples=100,
#     options={},
# )


# print(new_point)



