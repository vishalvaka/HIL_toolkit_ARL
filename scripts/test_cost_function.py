

from HIL.cost_processing.utils.inlet import InletOutlet
import pylsl
import numpy as np
import time
import torch
import yaml

noise_level = 0.0

args = yaml.safe_load(open('configs/test_function.yml','r'))
# values = np.linspace(args['Optimization']['range'][0][0], args['Optimization']['range'][1][0], num=100)

# output1 = values**2
# output2 = (values - 4)**2

# # noise1 = np.random.normal(0, np.std(output1))
# # noise2 = np.random.normal(0, np.std(output2))

# noise1 = 0.0
# noise2 = 0.0
def cost_function(x):
    x = np.array(x)
    return sum(- (x - 5 ) ** 2  + 100 + np.random.rand(1) * 0.1)

# def f(x): #ZDT1
#     x = np.array(x)

#     g = 1 + (9 / (len(x) - 1)) * np.sum(x[1:])

#     return np.array([x[0], g * (1 - np.sqrt(x[0] / g))])

# def f(x): #ZDT2
#     x = np.array(x)

#     g = 1 + (9 / (len(x) - 1)) * np.sum(x[1:])

#     return np.array([x[0], g * (1 - (x[0] / g) ** 2)])

def f(x): #ZDT2 1-D
    x = np.array(x)

    return np.array([x, 1 - x ** 2])

# def f(x): #ZDT1 1-D
#     x = np.array(x)

#     return np.array([x, 1 - np.sqrt(x)])

# def f(x): #fronesca and fleming
#     n = len(x)
#     f1 = 1 - np.exp(-np.sum((x - 1 / np.sqrt(n)) ** 2))
#     f2 = 1 - np.exp(-np.sum((x + 1 / np.sqrt(n)) ** 2))
#     return np.array([f1, f2])

# def f(x): #Schaffer N1
#     # print('recieved x: ', x)
#     # x = x[0][0]
#     # print(x)
#     x = np.array(x)
#     # print(x.shape)
#     return np.array([x ** 2, (x - 2) ** 2])

# def f(x): #levy and ackley 1d x
#     print('recieved x: ', x)
#     x = np.array(x)
#     f1 = np.sin(3 * np.pi * x)**2 + (x - 1)**2 * (1 + np.sin(3 * np.pi * x)**2)
#     a = 20
#     b = 0.2
#     c = 2 * np.pi
    
#     # Calculate the Ackley function
#     term1 = -a * np.exp(-b * np.sqrt(0.5 * (x**2)))
#     term2 = -np.exp(0.5 * (np.cos(c * x)))
#     f2 = term1 + term2 + a + np.exp(1)
#     return np.array([f1, f2])

def get_std_dev():
    
    samples = np.linspace(start=args['Optimization']['range'][0], 
                                   stop=args['Optimization']['range'][1],
                                   num=100)
    
    output1 = []
    output2 = []


    for sample in samples:
        result = f(sample)
        output1.append(result[0])
        output2.append(result[1])

    output1 = np.array(output1)
    output2 = np.array(output2)
    print(output1.shape, output2.shape)
    
    return np.std(output1), np.std(output2)

# def f(x, noise_level=noise_level):
#     results = []
#     for x_i in x:
#         print('x_i', x_i)
#         # x_i[0] = (x_i[0] - args["Optimization"]["range"][0])/(args["Optimization"]["range"][1] - args["Optimization"]["range"][0])   
#         sub_results = [
#          (np.sin(6 *x_i[0])**3 * (1 - np.tanh(x_i[0] ** 2))) + (-1 + torch.rand(1)[0] * 2) * noise_level,
#          .5 - (np.cos(5 * x_i[0] + 0.7)**3 * (1 - np.tanh(x_i[0] ** 2))) + (-1 + torch.rand(1)[0] * 2) * noise_level,
#       ]
#         results.append(sub_results)
#     return torch.tensor(results, dtype=torch.float32)

# def f(x, range):
#     print('recieved x: ', x)
#     x = x[0][0]
    
#     return np.array([x ** 2 + noise1, (x - 4) ** 2 + noise2])


# def f(x, noise_level = noise_level):
#     results = []
#     for x_i in x:
#         print(x_i)
#         sub_results = [
#             np.array(x_i[0] - 4) ** 2,
#             np.array(x_i[0]) ** 2
#         ]
#         results.append(np.array(sub_results))
#     return torch.tensor(results, dtype=torch.float32)

# def f(x, A=10, noise_level = noise_level):
#     results = []
#     for x_i in x:
#         # Rastrigin function for the first component
#         rastrigin_result = A * len(x_i) + sum([(xi ** 2 - A * np.cos(2 * np.pi * xi)) for xi in x_i])
        
#         # Sphere function for the second component
#         sphere_result = sum([xi ** 2 for xi in x_i])
        
#         sub_results = [
#             np.array(rastrigin_result),
#             np.array(sphere_result)
#         ]
#         results.append(np.array(sub_results))
#     return torch.tensor(results, dtype=torch.float32)



class test_cost_function:
    """Test the cost function class"""
    # create a fake stream
    
    def __init__(self) -> None:
        self.channel_count = 1
        if args["Optimization"]["MultiObjective"]:
            channel_count = 2

        
        info = pylsl.StreamInfo(name='test_function', type='cost', channel_count=self.channel_count,
                                nominal_srate=pylsl.IRREGULAR_RATE,
                                channel_format=pylsl.cf_double64,
                                source_id='test_function_id')
        self.outlet = pylsl.StreamOutlet(info)

        if args["Optimization"]["MultiObjective"]:
            self.outlet2 = pylsl.StreamOutlet(pylsl.StreamInfo(name='test_function_2', type='cost', channel_count=self.channel_count,
                                nominal_srate=pylsl.IRREGULAR_RATE,
                                channel_format=pylsl.cf_double64,
                                source_id='test_function_id'))

        pylsl.local_clock()
        self.inlet = None

        self._get_streams()
        
        self.x_parmeter = [1, 1]
        self.parameters = np.empty((100, 2))

    def _get_streams(self) -> None:
        """Get the streams from the inlet"""
        print("looking for an Change_parm stream...")
        streams = pylsl.resolve_byprop("name", "Change_parm", timeout=1)
        if len(streams) == 0:
           pass
        else:
            self.inlet = pylsl.StreamInlet(streams[0])


    def _get_cost(self) -> None:
        """Get the cost function"""
        sample, timestamp = self.inlet.pull_sample()
        sample = sample[:args['Optimization']['n_parms']]
        # print(sample, timestamp)
        if timestamp:
            # print(timestamp, sample)
            self.inlet.flush()
            pass
            # sample = sample[0]
            print(f"Received {sample} at time {timestamp}")
            self.x_parmeter = sample

    def run(self,) -> None:
        """Main run function for the cost function this will take the parameters and send the cost function to the outlet.

        Args:
            x_parmeter (float): parameter for the cost function.
        """
        counter = 0
        noise1, noise2 = get_std_dev()
        while True:
            time.sleep(0.1)
            print("running")
            if self.inlet is None:
                self._get_streams()
            else:
                self._get_cost()
                x_parmeter = self.x_parmeter
                # print(f'cost function for {x_parmeter}, {cost_function(x_parmeter)}')
                # print(f([x_parmeter], noise_level))
                if not args["Optimization"]["MultiObjective"]:
                    self.outlet.push_sample([cost_function(x_parmeter)])
                else:
                    # func_result = f([x_parmeter], noise_level=noise_level)
                    range = args['Optimization']['range']
                    print(range)
                    func_result = f(x_parmeter)
                    print('func_result', func_result)
                    result1 = func_result[0].item()  # Convert to Python scalar
                    result2 = func_result[1].item()  # Convert to Python scalar
                    # self.outlet.push_sample([result1 + np.random.normal(0, noise1)])
                    # self.outlet2.push_sample([result2 + np.random.normal(0, noise1)])

                    self.outlet.push_sample([result1])
                    self.outlet2.push_sample([result2])
                # if counter > 10:
                #     del self.inlet
                #     self._get_streams()




if __name__ == "__main__":
    test = test_cost_function()
    test.run()