import numpy as np
import pandas as pd
import yaml

# Load configuration
# args = yaml.safe_load(open('configs/test_function.yml', 'r'))

# def f(x, shift):  # Example function definition
#     n = len(x)
#     x = np.array(x)
#     f1 = 1 - np.exp(-np.sum((x + shift - 1 / np.sqrt(n)) ** 2))
#     f2 = 1 - np.exp(-np.sum((x + shift + 1 / np.sqrt(n)) ** 2))
#     return np.array([f1, f2])

# def f(x, shift):
#     f1 = (shift*x)**2
#     f2 = (shift*x - 2)**2
#     return np.array([f1, f2])

# #ZDT1
def f(x, shift):
    x = np.array(x)
    f1 = x * shift
    f2 = 1 - np.sqrt(x * shift)
    return np.array([f1, f2])

# #ZDT2
# def f(x, shift): 
#     x = np.array(x)
#     f1 = x * shift
#     f2 = 1 - (x * shift) ** 2
#     return np.array([f1, f2])

# # FON
# def f(x, shift): 
#     n = len(x)
#     x = np.array(x)
#     f1 = 1 - np.exp(-np.sum((x + shift - 1 / np.sqrt(n)) ** 2))
#     f2 = 1 - np.exp(-np.sum((x + shift + 1 / np.sqrt(n)) ** 2))
    
#     return np.array([f1, f2])


class CreateBaseModels:
    def __init__(self, config = yaml.safe_load(open('configs/test_function.yml', 'r'))['Optimization']):
        self.args = config

    def get_std_dev(self, f, shift):
        samples = np.linspace(start=self.args['range'][0], 
                              stop=self.args['range'][1], 
                              num=100)
        # samples = np.linspace(start=0.0, stop=2.0, num=100)

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

    def run(self, f, shift_values, num_samples, ranges=None, with_noise:bool = False):
        ranges = self.args['range'] if ranges == None else ranges
        file_paths = []

        for i, shift in enumerate(shift_values):
            data = []
            for _ in range(num_samples):
                noise1, noise2 = self.get_std_dev(f, shift)
                x = np.random.uniform(ranges[0], 
                                      ranges[1], 1)
                # x = np.random.uniform(0, 2, 1)
                f_values = f(x, shift)
                if with_noise:
                    data.append([x[0], f_values[0].item() + np.random.normal(0, noise1), f_values[1].item() + np.random.normal(0, noise2)])
                else:
                    data.append([x[0], f_values[0], f_values[1]])
            df = pd.DataFrame(data)
            print(df)
            file_name = f"data.csv"
            file_path = f"base_models/model{i+1}/{file_name}"
            df.to_csv(file_path, header=False, index=False, sep=' ')
            file_paths.append(file_path)

        return file_paths

# Example usage:
if __name__ == "__main__":
    shift_values = [0.9, 1.2, 1.5, 3]
    num_samples = 15

    runner = CreateBaseModels()
    file_paths = runner.run(f, shift_values, num_samples, with_noise=True)

    print(file_paths)


# import numpy as np
# import pandas as pd
# import yaml

# args = yaml.safe_load(open('configs/test_function.yml','r'))

# def get_std_dev(shift):
    
#     samples = np.linspace(start=args['Optimization']['range'][0], 
#                                    stop=args['Optimization']['range'][1],
#                                    num=100)
    
#     output1 = []
#     output2 = []


#     for sample in samples:
#         result = f(sample, shift)
#         output1.append(result[0])
#         output2.append(result[1])

#     output1 = np.array(output1)
#     output2 = np.array(output2)
#     print(output1.shape, output2.shape)
    
#     return np.std(output1), np.std(output2)

# # Define the ZDT1 function with shift parameter
# # def ZDT1(x, shift):
# #     x = np.array(x)
# #     g = 1 + ((9 - shift) / (len(x) - 1)) * np.sum(x[1:])
# #     f1 = x[0] * (1 - shift)
# #     f2 = g * (1 - np.sqrt(x[0] / g))
# #     return np.array([f1, f2])

# # def f(x, shift=0): #ZDT1 1-D
# #     x = np.array(x)

# #     # shift  = 1.0

# #     f1 = x * shift
# #     f2 = 1 - np.sqrt(x * shift)

# #     return np.array([f1, f2])

# # def f(x, shift):
# #     return x * np.sin(x + np.pi + shift) + 0.1 * x

# def f(x, shift): #fronesca and fleming
#     n = len(x)
#     x = np.array(x)
#     # shift = 0.0
#     f1 = 1 - np.exp(-np.sum((x + shift - 1 / np.sqrt(n)) ** 2))
#     f2 = 1 - np.exp(-np.sum((x + shift + 1 / np.sqrt(n)) ** 2))
#     return np.array([f1, f2])

# # Define shift values and number of samples
# shift_values = [-1.0, 0.5, 1.5, 2]
# num_samples = 20

# # Randomly sample points from the function
# data = []

# # Create separate CSV files for each shift value
# file_paths = []



# for i, shift in enumerate(shift_values):
#     data = []
#     for _ in range(num_samples):
#         noise1, noise2 = get_std_dev(shift)
#         x = np.random.uniform(args['Optimization']['range'][0], args['Optimization']['range'][1], 1)  
#         #x = np.random.rand(2) # 2D array with values between 0 and 1
#         f_values = f(x, shift)
#         # data.append([x[0], x[1], f_values[0], f_values[1]])
#         data.append([x[0], f_values[0] + np.random.normal(0, noise1), f_values[1] + np.random.normal(0, noise2)])
#         # data.append([x[0], f_values[0], f_values[1]])
#         # data.append([x[0], f_values[0]])
#     # Convert to DataFrame
#     df = pd.DataFrame(data)
    
#     # Save to individual CSV files without headers and with whitespace separator
#     file_name = f"data.csv"
#     file_path = f"base_models\model{i+1}\{file_name}"
#     df.to_csv(file_path, header=False, index=False, sep=' ')
#     file_paths.append(file_path)

# file_paths
