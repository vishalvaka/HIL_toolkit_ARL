from HIL.optimization.MOBO import MultiObjectiveBayesianOptimization
import pandas as pd
import matplotlib.pyplot as plt
import torch
import yaml

args = yaml.safe_load(open('configs/test_function.yml','r'))
df_xy = pd.read_csv('models/iter_10/data.csv', delim_whitespace=True, header = None)


MOBO = MultiObjectiveBayesianOptimization(
    bounds=args['Optimization']['range'], 
    is_rgpe=True,
    x_dim = args['Optimization']['n_parms']
    )

x = torch.tensor(df_xy.iloc[:, args['Optimization']['n_parms']].values, dtype = torch.float64).squeeze(-1)

y = torch.tensor(df_xy.iloc[:, args['Optimization']['n_parms']:].values, dtype = torch.float64).squeeze()

print(y.shape)

_ = MOBO.generate_next_candidate(x, y)

MOBO.plot_final()