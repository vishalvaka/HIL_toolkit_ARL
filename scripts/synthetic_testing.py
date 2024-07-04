import time
import numpy as np
import threading
import pylsl
import yaml
from HIL.optimization.HIL import HIL
from HIL.optimization.MOBO import MultiObjectiveBayesianOptimization

def f_zdt2(x, shift): # ZDT2 1-D
    x = np.array(x)
    # shift = 1.0
    f1 = x * shift
    f2 = 1 - (x * shift) ** 2
    return np.array([f1, f2])

def f_zdt1(x, shift): # ZDT1 1-D
    x = np.array(x)
    # shift  = 1.0
    f1 = x * shift
    f2 = 1 - np.sqrt(x * shift)
    return np.array([f1, f2])

args = yaml.safe_load(open('configs/test_function.yml','r'))

n_expl = args["Optimization"]["n_exploration"]
range_arr = args["Optimization"]["range"]

    x = np.linspace(start=range_arr[0], stop=range_arr[1], num=n_expl)
    # np.random.seed(time.time())
    np.random.shuffle(x)
    print(f"###### start functions are {x} ######")



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