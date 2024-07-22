import time
import numpy as np
import pandas as pd
import threading
import pylsl
import yaml
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
    return np.array([f1.item(), f2.item()])

def f_zdt1(x, shift=1): # ZDT1 1-D
    x = torch.tensor(x)
    # shift  = 1.0
    f1 = x * shift
    f2 = 1 - torch.sqrt(x * shift)
    return np.array([f1.item(), f2.item()])

def fon(x, shift=0): #fronesca and fleming
    x = torch.tensor(x)
    print(type(x))
    n = len(x)
    # shift = 0.0
    f1 = 1 - torch.exp(-torch.sum((x + shift - 1 / torch.sqrt(torch.tensor(n, dtype=torch.float32))) ** 2))
    f2 = 1 - torch.exp(-torch.sum((x + shift + 1 / torch.sqrt(torch.tensor(n, dtype=torch.float32))) ** 2))
    return np.array([f1.item(), f2.item()])

args = yaml.safe_load(open('configs/synthetic_testing.yml','r'))


for cost_func in [f_zdt2, f_zdt1, fon]:

    #RGPE
    print(f'starting RGPE optimization with cost function: {cost_func.__name__}')
    base_models = CreateBaseModels()
    base_models.run(f=cost_func, shift_values=args["Shifts"][cost_func.__name__], num_samples=args["Optimization"]["n_opt"],
                     range_arr=args["Ranges"][cost_func.__name__])

    MOBO = MultiObjectiveBayesianOptimization(bounds=args["Ranges"][cost_func.__name__], is_rgpe=True, x_dim=1)
    # x, y = x, cost_func(x) for x in np.random.uniform(args["Ranges"][cost_func.__name__][0], 
                                                    #   args["Ranges"][cost_func.__name__][1], size=args["Optimization"]["n_expl"])

    x = torch.tensor(np.random.uniform(args["Ranges"][cost_func.__name__][0], args["Ranges"][cost_func.__name__][1], size=args["Optimization"]["n_expl"]))

    y = torch.tensor(np.array([cost_func([xi]) for xi in x]))

    for i in range(args["Optimization"]["n_expl"] + 1, args["Optimization"]["n_opt"] + 1):
        new_x = MOBO.generate_next_candidate(x.unsqueeze(-1), y)
        # print(f'x.shape {x.shape}, y.shape: {y.shape}')
        # print(torch.tensor([cost_func(new_x)]).shape)
        print(f'new parameter is {new_x}')
        x = torch.cat((x, new_x.squeeze(-1)))
        y = torch.cat((y, torch.tensor([cost_func(new_x)])))
        print(f'new x: {x}, new y: {y}')
# class TestCostFunction:
#     """Test the cost function class"""
#     def __init__(self, cost_function, stop_event) -> None:
#         self.channel_count = 1
#         if args["Optimization"]["MultiObjective"]:
#             channel_count = 2

#         info = pylsl.StreamInfo(name='test_function', type='cost', channel_count=self.channel_count,
#                                 nominal_srate=pylsl.IRREGULAR_RATE,
#                                 channel_format=pylsl.cf_double64,
#                                 source_id='test_function_id')
#         self.outlet = pylsl.StreamOutlet(info)

#         if args["Optimization"]["MultiObjective"]:
#             self.outlet2 = pylsl.StreamOutlet(pylsl.StreamInfo(name='test_function_2', type='cost', channel_count=self.channel_count,
#                                 nominal_srate=pylsl.IRREGULAR_RATE,
#                                 channel_format=pylsl.cf_double64,
#                                 source_id='test_function_id'))

#         self.cost_function = cost_function
#         self.inlet = None
#         self._get_streams()
#         self.x_parameter = [1, 1]
#         self.parameters = np.empty((100, 2))
#         self.stop_event = stop_event

#     def _get_streams(self) -> None:
#         """Get the streams from the inlet"""
#         print("looking for an Change_parm stream...")
#         streams = pylsl.resolve_byprop("name", "Change_parm", timeout=1)
#         if len(streams) == 0:
#             pass
#         else:
#             self.inlet = pylsl.StreamInlet(streams[0])

#     def _get_cost(self) -> None:
#         """Get the cost function"""
#         sample, timestamp = self.inlet.pull_sample()
#         sample = sample[:args['Optimization']['n_parms']]
#         if timestamp:
#             self.inlet.flush()
#             self.x_parameter = sample

#     def run(self) -> None:
#         """Main run function for the cost function this will take the parameters and send the cost function to the outlet.

#         Args:
#             x_parmeter (float): parameter for the cost function.
#         """
#         counter = 0
#         noise1, noise2 = get_std_dev(self.cost_function)
#         while not self.stop_event.is_set():
#             time.sleep(0.1)
#             print("running")
#             if self.inlet is None:
#                 self._get_streams()
#             else:
#                 self._get_cost()
#                 x_parmeter = self.x_parameter
#                 # print(f'cost function for {x_parmeter}, {cost_function(x_parmeter)}')
#                 # print(f([x_parmeter], noise_level))
#                 # if not args["Optimization"]["MultiObjective"]:
#                 #     self.outlet.push_sample([cost_function(x_parmeter)])
#                 # else:
#                 # func_result = f([x_parmeter], noise_level=noise_level)
#                 range = args['Optimization']['range']
#                 print(range)
#                 func_result = self.cost_function(x_parmeter)
#                 print('func_result', func_result)
#                 result1 = func_result[0].item()  # Convert to Python scalar
#                 result2 = func_result[1].item()  # Convert to Python scalar
#                 # self.outlet.push_sample([result1 + np.random.normal(0, noise1)])
#                 # self.outlet2.push_sample([result2 + np.random.normal(0, noise1)])

#                 self.outlet.push_sample([result1])
#                 self.outlet2.push_sample([result2])

#         # Close the inlet stream if it's open
#         if self.inlet is not None:
#             self.inlet.close_stream()
#             self.inlet = None

# def get_std_dev(cost_function):
#     samples = np.linspace(start=args['Optimization']['range'][0], 
#                           stop=args['Optimization']['range'][1],
#                           num=100)
    
#     output1 = []
#     output2 = []

#     for sample in samples:
#         result = cost_function(sample)
#         output1.append(result[0])
#         output2.append(result[1])

#     output1 = np.array(output1)
#     output2 = np.array(output2)
#     return np.std(output1), np.std(output2)

# def optimization():
#     if args['Optimization']['n_parms'] != len(args['Optimization']['range'][0]):
#         raise ValueError('Please change the n_parm variable')
#     elif len(args['Optimization']['range'][0]) != len(args['Optimization']['range'][1]):
#         raise ValueError('Please make sure both range arrays have the same length')
#     hil = HIL(args)
#     hil.start()

# def main():
#     cost_functions = [f_zdt2, f_zdt1]
    
#     for cost_function in cost_functions:
#         print(f"\n\n\n\nRunning with {cost_function.__name__}")
        
#         stop_event = threading.Event()  # Create a stop event
#         test = TestCostFunction(cost_function, stop_event)
        
#         # Thread for the cost function streaming
#         thread1 = threading.Thread(target=test.run)
#         thread1.start()

#         # Thread for the other execution task
#         thread2 = threading.Thread(target=optimization)
#         thread2.start()

#         thread2.join()  # Wait for the other execution task to finish

#         print('\n\n\n\noptimization ended')
#         thread1.do_run = False  # Signal the cost function thread to stop
#         thread1.join()  # Wait for the cost function thread to stop

#         print(f"Finished with {cost_function.__name__}")

# if __name__ == "__main__":
#     main()