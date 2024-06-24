import yaml
# import numpy as np

# from CostFunctions.main import ExtractCost
# from Optimization.main import BayesianOptimization

from HIL.optimization.HIL import HIL






def run():

    args = yaml.safe_load(open('configs/test_function.yml','r'))

    if args['Optimization']['n_parms'] != len(args['Optimization']['range'][0]):
        raise ValueError('Please change the n_parm variable')
    elif len(args['Optimization']['range'][0]) != len(args['Optimization']['range'][1]):
        raise ValueError('Please make sure both range arrays have the same length')
    hil = HIL(args)
    hil.start()


run()