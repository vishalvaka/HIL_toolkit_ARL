import time
import numpy as np
import pandas as pd
import threading
import pylsl
import yaml
import matplotlib.pyplot as plt
import torch
# from HIL.optimization.HIL import HIL
from HIL.optimization.MOBO import MultiObjectiveBayesianOptimization
# from scripts.test_cost_function import test_cost_function
# from scripts.base_models import CreateBaseModels

# def get_std_dev(cost_function, range):
#     # samples = np.linspace(start=args['Optimization']['range'][0], 
#     #                       stop=args['Optimization']['range'][1],
#     #                       num=100)
#     samples = np.linspace(start=range[0], stop=range[1], num=100)
    
#     output1 = []
#     output2 = []

#     for sample in samples:
#         result = cost_function(sample)
#         output1.append(result[0])
#         output2.append(result[1])

#     output1 = np.array(output1)
#     output2 = np.array(output2)
#     return np.std(output1), np.std(output2)

class CreateBaseModels:
    def __init__(self, config = yaml.safe_load(open('configs/test_function.yml', 'r'))['Optimization']):
        self.args = config

    def get_std_dev(self, f, shift):
        samples = np.linspace(start=self.args['range'][0], 
                              stop=self.args['range'][1], 
                              num=100)

        output1 = []
        output2 = []

        for sample in samples:
            result = f(sample, shift)
            output1.append(result[0])
            output2.append(result[1])

        output1 = np.array(output1)
        output2 = np.array(output2)
        print(output1.shape, output2.shape)

        return np.std(output1), np.std(output2)

    def run(self, f, shift_values, num_samples, range_arr=None, with_noise:bool = False):
        range_arr = self.args['range'] if range_arr == None else range_arr
        print(range_arr)
        file_paths = []

        for i, shift in enumerate(shift_values):
            data = []
            for _ in range(num_samples):
                noise1, noise2 = self.get_std_dev(f, shift)
                x = np.random.uniform(range_arr[0], 
                                      range_arr[1], 1)
                f_values = f(x, shift)
                if with_noise:
                    data.append([x[0], f_values[0] + np.random.normal(0, noise1), f_values[1] + np.random.normal(0, noise2)])
                else:
                    data.append([x[0], f_values[0], f_values[1]])
            df = pd.DataFrame(data)
            file_name = f"data.csv"
            file_path = f"base_models/model{i+1}/{file_name}"
            df.to_csv(file_path, header=False, index=False, sep=' ')
            file_paths.append(file_path)

        return file_paths

def f_zdt2(x, shift=1): # ZDT2 1-D
    x = torch.tensor(x)
    # shift = 1.0
    f1 = x * shift
    f2 = 1 - (x * shift) ** 2
    return -np.array([f1.item(), f2.item()])

def f_zdt1(x, shift=1): # ZDT1 1-D
    x = torch.tensor(x)
    # shift  = 1.0
    f1 = x * shift
    f2 = 1 - torch.sqrt(x * shift)
    return -np.array([f1.item(), f2.item()])

def fon(x, shift=0): #fronesca and fleming
    x = torch.tensor(x)
    print(type(x))
    n = len(x)
    # shift = 0.0
    f1 = 1 - torch.exp(-torch.sum((x + shift - 1 / torch.sqrt(torch.tensor(n, dtype=torch.float32))) ** 2))
    f2 = 1 - torch.exp(-torch.sum((x + shift + 1 / torch.sqrt(torch.tensor(n, dtype=torch.float32))) ** 2))
    return -np.array([f1.item(), f2.item()])

args = yaml.safe_load(open('configs/synthetic_testing.yml','r'))


for cost_func in [f_zdt2, f_zdt1, fon]:

    #RGPE
    print(f'starting RGPE optimization with cost function: {cost_func.__name__}')
    base_models = CreateBaseModels()
    base_models.run(f=cost_func, shift_values=args["Shifts"][cost_func.__name__], num_samples=args["Optimization"]["n_opt"],
                     range_arr=args["Ranges"][cost_func.__name__])
    # MOBO = MultiObjectiveBayesianOptimization(bounds=args["Ranges"][cost_func.__name__], is_rgpe=False, x_dim=1)
    MOBO_rgpe = MultiObjectiveBayesianOptimization(bounds=args["Ranges"][cost_func.__name__], is_rgpe=True, x_dim=1)
    # x, y = x, cost_func(x) for x in np.random.uniform(args["Ranges"][cost_func.__name__][0], 
                                                    #   args["Ranges"][cost_func.__name__][1], size=args["Optimization"]["n_expl"])

    x_rgpe = torch.tensor(np.random.uniform(args["Ranges"][cost_func.__name__][0], args["Ranges"][cost_func.__name__][1], size=args["Optimization"]["n_expl"]))

    y_rgpe = torch.tensor(np.array([cost_func([xi]) for xi in x_rgpe]))

    hyperparameters_list = []

    for i in range(args["Optimization"]["n_expl"] + 1, args["Optimization"]["n_opt"] + 1):
        new_x_rgpe = MOBO_rgpe.generate_next_candidate(x_rgpe.unsqueeze(-1), y_rgpe)
        # new_x = MOBO.generate_next_candidate(x_rgpe.unsqueeze(-1), y)
        # print(f'x.shape {x.shape}, y.shape: {y.shape}')
        # print(torch.tensor([cost_func(new_x)]).shape)
        print(f'new parameter is {new_x_rgpe}')
        x_rgpe = torch.cat((x_rgpe, new_x_rgpe.squeeze(-1)))
        y_rgpe = torch.cat((y_rgpe, torch.tensor([cost_func(new_x_rgpe)])))

        # Capture and save hyperparameters
        # hyperparameters = {name: param.data.clone().cpu().numpy() for name, param in MOBO_rgpe.named_parameters()}
        last_model = [MOBO_rgpe.model.models[0].models[-1], MOBO_rgpe.model.models[1].models[-1]]
        hyperparameters = {
            "lengthscale": [last_model[0].covar_module.base_kernel.lengthscale.detach().cpu().numpy(), 
                            last_model[1].covar_module.base_kernel.lengthscale.detach().cpu().numpy()],
            "noise": [last_model[0].likelihood.noise.detach().cpu().numpy(),
                      last_model[1].likelihood.noise.detach().cpu().numpy()],
            "outputscale": [last_model[0].covar_module.outputscale.detach().cpu().numpy(),
                            last_model[1].covar_module.outputscale.detach().cpu().numpy()]
        }
        hyperparameters_list.append(hyperparameters)
        # hyperparameters_list.append(hyperparameters)

        print(f'New parameter is {new_x_rgpe}')
        print(f'New x: {x_rgpe[-1]}, New y: {y_rgpe[-1]}')

        # x = torch.cat((x, new_x_rgpe.squeeze(-1)))
        # y = torch.cat((y, torch.tensor([cost_func(new_x_rgpe)])))
        print(f'new x: {x_rgpe[-1]}, new y: {y_rgpe[-1]}')
    
    MOBO_rgpe.plot_final()

    for param_name in hyperparameters_list[0].keys():
        plt.figure(figsize=(10, 6))
        for i in range(2):  # Since there are two models
            param_values = [hyperparams[param_name][i] for hyperparams in hyperparameters_list]
            param_values = np.array(param_values).squeeze()
            if param_values.ndim == 1:
                plt.plot(param_values, label=f'{param_name}_objective_{i+1}')
            else:
                for j in range(param_values.shape[1]):
                    plt.plot(param_values[:, j], label=f'{param_name}_objective_{i+1}_{j}')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'RGPE Hyperparameter: {param_name}')
        plt.legend()
        plt.show()

    #regular GP
    MOBO = MultiObjectiveBayesianOptimization(bounds=args["Ranges"][cost_func.__name__], is_rgpe=False, x_dim=1)

    x = torch.tensor(np.random.uniform(args["Ranges"][cost_func.__name__][0], args["Ranges"][cost_func.__name__][1], size=args["Optimization"]["n_expl"]))

    y = torch.tensor(np.array([cost_func([xi]) for xi in x]))

    hyperparameters_list = []

    for i in range(args["Optimization"]["n_expl"] + 1, args["Optimization"]["n_opt"] + 1):
        new_x = MOBO.generate_next_candidate(x.unsqueeze(-1), y)
        # new_x = MOBO.generate_next_candidate(x_rgpe.unsqueeze(-1), y)
        # print(f'x.shape {x.shape}, y.shape: {y.shape}')
        # print(torch.tensor([cost_func(new_x)]).shape)
        print(f'new parameter is {new_x}')
        x = torch.cat((x, new_x.squeeze(-1)))
        y = torch.cat((y, torch.tensor([cost_func(new_x)])))

        models = [MOBO.model.models[0], MOBO.model.models[1]]

        hyperparameters = {
            "lengthscale": [models[0].covar_module.base_kernel.lengthscale.detach().cpu().numpy(), 
                            models[1].covar_module.base_kernel.lengthscale.detach().cpu().numpy()],
            "noise": [models[0].likelihood.noise.detach().cpu().numpy(),
                      models[1].likelihood.noise.detach().cpu().numpy()],
            "outputscale": [models[0].covar_module.outputscale.detach().cpu().numpy(),
                            models[1].covar_module.outputscale.detach().cpu().numpy()]
        }
        hyperparameters_list.append(hyperparameters)

        print(f'New parameter is {new_x}')
        print(f'New x: {x[-1]}, New y: {y[-1]}')

        # x = torch.cat((x, new_x_rgpe.squeeze(-1)))
        # y = torch.cat((y, torch.tensor([cost_func(new_x_rgpe)])))
        print(f'new x: {x[-1]}, new y: {y[-1]}')
    
    MOBO.plot_final()

    for param_name in hyperparameters_list[0].keys():
        plt.figure(figsize=(10, 6))
        for i in range(2):  # Since there are two models
            param_values = [hyperparams[param_name][i] for hyperparams in hyperparameters_list]
            param_values = np.array(param_values).squeeze()
            if param_values.ndim == 1:
                plt.plot(param_values, label=f'{param_name}_objective_{i+1}')
            else:
                for j in range(param_values.shape[1]):
                    plt.plot(param_values[:, j], label=f'{param_name}_objective_{i+1}_{j}')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.title(f'Regular GP Hyperparameter: {param_name}')
        plt.legend()
        plt.show()


    # for param_name in hyperparameters_list[0].keys():
    #     plt.figure(figsize=(10, 6))
    #     param_values = [hyperparams[param_name] for hyperparams in hyperparameters_list]
    #     param_values = np.array(param_values).squeeze()
    #     if param_values.ndim == 1:
    #         plt.plot(param_values, label=param_name)
    #     else:
    #         for i in range(param_values.shape[1]):
    #             plt.plot(param_values[:, i], label=f'{param_name}_{i}')
    #     plt.xlabel('Iteration')
    #     plt.ylabel('Value')
    #     plt.title(f'Hyperparameter: {param_name}')
    #     plt.legend()
    #     plt.show()

    # MOBO = MultiObjectiveBayesianOptimization(bounds=args["Ranges"][cost_func.__name__], is_rgpe=True, x_dim=1)
    # # x, y = x, cost_func(x) for x in np.random.uniform(args["Ranges"][cost_func.__name__][0], 
    #                                                 #   args["Ranges"][cost_func.__name__][1], size=args["Optimization"]["n_expl"])

    # x_rgpe = torch.tensor(np.random.uniform(args["Ranges"][cost_func.__name__][0], args["Ranges"][cost_func.__name__][1], size=args["Optimization"]["n_expl"]))

    # y_rgpe = torch.tensor(np.array([cost_func([xi]) for xi in x_rgpe]))

    # for i in range(args["Optimization"]["n_expl"] + 1, args["Optimization"]["n_opt"] + 1):
    #     new_x_rgpe = MOBO_rgpe.generate_next_candidate(x_rgpe.unsqueeze(-1), y_rgpe)
    #     # new_x = MOBO.generate_next_candidate(x_rgpe.unsqueeze(-1), y)
    #     # print(f'x.shape {x.shape}, y.shape: {y.shape}')
    #     # print(torch.tensor([cost_func(new_x)]).shape)
    #     print(f'new parameter is {new_x_rgpe}')
    #     x_rgpe = torch.cat((x_rgpe, new_x_rgpe.squeeze(-1)))
    #     y_rgpe = torch.cat((y_rgpe, torch.tensor([cost_func(new_x_rgpe)])))

    #     # x = torch.cat((x, new_x_rgpe.squeeze(-1)))
    #     # y = torch.cat((y, torch.tensor([cost_func(new_x_rgpe)])))
    #     print(f'new x: {x_rgpe[-1]}, new y: {y_rgpe[-1]}')
    
    # MOBO_rgpe.plot_final()