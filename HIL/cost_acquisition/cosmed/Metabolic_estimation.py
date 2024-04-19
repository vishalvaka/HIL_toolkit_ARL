import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import time, sys, signal,logging, os, datetime


from HIL.cost_processing.metabolic_cost.ppe import PPE  

# cosmed module
from HIL.cost_acquisition.cosmed.cosmed_utils import MainCosmed


# communication module
from HIL.cost_acquisition.cosmed.communication_met import MainCommunication
from HIL.cost_acquisition.cosmed.communication_met import MainReceiveCommunication


class MetCostEstimation:
    def __init__(self, plot: bool = False) -> None:
        """This is the main class for the metabolic cost estimation

        Args:
            plot (bool, optional): Plotting of the metabolic cost with predicted values. Defaults to False.
        """
        # initalizing all the sub class


        # start the cosmed class
        self.start_realtime_cosmed()

        # start plotting, this should be based on a condition
        self.plot = plot
        if plot:
            self.fig = plt.figure()
            self.plot = plt.subplot(111)
            plt.show(block=False)

        # dictionary to store data
        self.store_values = {'result': [],
                             'ICM': [],
                             'smooth data': [],
                             'raw data': []}
        
        # initalize the of the ppe and other important variables
        self._initalize()

        # Communication

        # 2 server default: localhost, prediction: 50005, stopping: 50007
        # This will also send the data to the pylsl
        self.communication = MainCommunication()

        # stop receving server this will look for the pylsl and UDP signal, needs more testing.
        self.stop = MainReceiveCommunication()

    def _initalize(self) -> None:
        """Initialization function for the PPE and other important variables
        """
        # start the prediction class
        # This is where the PPE class will be called.
        # I am adding upsample number to be 5 but if you want smooth data you can increase it 20. If do so, the prediction will be slower.
        self.predict = PPE(upsample_number=5)

        # this is for the while break clause
        self.while_break = False
        # stop flag
        self.stop_flag = False
        self.testing_flag = False
        self.time = time.time()


    def _read_stop(self) -> None:
        """Reading the stop from the BO and restarting the data. The stop is receieved using the UDP and pylsl
        """
        # check if the stop is received from the BO
        current_n = self.current_data.shape
        # print('$' * 20)
        
        if self.stop.receive():
            self.data.reset()


    def start_realtime_cosmed(self) -> None:
        """This is the function to start the cosmed class
        """
        self.cosmed = MainCosmed()
        self.cosmed.start()
        self.data = self.cosmed.data



    def run(self) -> None:
        """This is the main run function for the metabolic cost estimation
        """
        while not self.while_break:
            time.sleep(0.1)
            if self.data.start_prediction:
                if self.data.STATUS:
                    # read the data either realtime or from the data buffer
                    self._data_read()
                    # predict with the data
                    self._predict()

                    # update the plot
                    if self.plot:
                        self._update_plot()

                    # testing the ICM.
                    self.communication.send_pred(self.result[-1])

                    # check if the stop is received
                    self._read_stop()

                else:
                    pass
                    # logging.debug(f'stopping is not performed')
            else:
                pass
                # logging.debug(f'prediction is not performed')
        logging.critical('prediction stopped')
    
    # function to read the data from the data storing class
    def _data_read(self) -> None:
        """This is the function to read the data from the data storing class
        """
        self.current_data = self.data.Read()


    def _predict(self):
        """
        This is the function to predict the metabolic cost.
        """
        data = self.current_data[0,:] * 16.58 / 60 + self.current_data[1,:] * 4.51 / 60
        time = self.current_data[2,:]
        # fix this.
        result, std = self.predict.estimate(data,time)
        self.result = [result]
        
        self.store_values['smooth data'].append(self.predict.smooth_data)
        self.store_values['result'].append(result)
        self.store_values['std'].append(std)
        self.store_values['raw data'].append(data)
        print(f'prediction: {self.result}, start: {self.data.start_index}')




    def _update_plot(self):
        self.plot.cla()
        self.plot.plot(self.store_values['result'], 'b', label = 'prediction')
        self.plot.plot(self.store_values['smooth data'], 'r', label = 'smooth data')
        self.plot.plot(self.store_values['raw data'], 'y', alpha = 0.6, label = 'raw data')
        self.plot.relim()
        plt.legend()
        # (0.001)
        plt.pause(0.001)
        



