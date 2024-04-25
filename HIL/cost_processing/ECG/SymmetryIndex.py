# general imports
import logging
import numpy as np
import math
import pylsl
import time
import os
import copy
import abc

# processing
import neurokit2 as nk
import scipy
from HIL.cost_processing.ECG import ECGComplexity

# typing
from typing import List

from HIL.cost_processing.utils.inlet import InletOutlet

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class SymmetryIndexInOut(InletOutlet):
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, data_length: int, sampling_rate: int = 133, skip_threshold: int = 40, stream_name: str = 'Symmetry_Index') -> None:
        """Main class which handles the input of the ECG data and output of the rmssd data to the pylsl.

        Args:
            info (pylsl.StreamInfo): information aboute the input stream
            data_length (int): Data length to process to get the RMSSD, generally 1000 -> ~9s considering 133 Hz sampling rate
            sampling_rate (int, optional): Sampling rate of the signal. Defaults to 133.
            skip_threshold (int, optional): This is the higher level threshold, above which RMSSD is not sent, (Generally standing threshold is picked). Defaults to 40.
        """
        super().__init__(info, data_length)
        buffer_size = (2 * math.ceil(info.nominal_srate() * data_length), info.channel_count())
        self.buffer = np.empty(buffer_size, dtype = self.dtypes[info.channel_format()])

        # placeholders
        self.store_data = np.array([])

        # Information about the outlet stream
        info  = pylsl.StreamInfo(stream_name,  'Marker', 1, 0, 'float32', 'myuidw43537') #type: ignore
        self.outlet = pylsl.StreamOutlet(info)

        # logging
        self._logger = logging.getLogger()

        # # setting some large RMMSD value at the start.
        # self.previous_HR = 1000
        self.skip_threshold = skip_threshold
        self.SAMPLING_RATE = sampling_rate

        # Main processing class
        self.symmetryIndex = SymmetryIndex(self.SAMPLING_RATE)

        # flags
        self.first_data = True
        self.cleaned = np.array([])


    def get_data(self) -> None:
        print(f'in get data, self.first_data is {self.first_data}')
        """Class to get the ECG data and process it

        Returns:
            None: early return if the stream if not ECG
        """
        _, ts = self.inlet.pull_chunk(timeout = 0.0, 
                max_samples=self.buffer.shape[0],
                dest_obj=self.buffer)
        print('self.buffer.shape[0]: ', self.buffer.shape)

        if not ts or self.name != "polar accel":
            self._logger.warning(f"Time stamp is: {ts}, name of the stream: {self.name}")
            return None
        
        ts = np.array(ts)
        # self.buffer = np.array(self.buffer).T
        # For first time
        if self.first_data:
            print('first data \n\n\n\n')
            self.store_data = np.array(self.buffer[0:ts.size,:]).T[0]
            print('self.store_data.size ', self.store_data.size)
            self.first_data = False

        else:
            print('calling add data to instatialize self.raw_data')
            self.store_data = np.append(self.store_data.flatten(), np.array(self.buffer[0:ts.size,:]).T[0].flatten())
            print('self.store_data.size ', self.store_data.size)
            self.symmetryIndex.add_data(self.store_data)
            

            # check if there is a nan in the data
            if np.isnan(self.store_data).any(): #type: ignore
                self._logger.warn(f"nan found in data")

    def send_data(self) -> None:
        """Send the processed data to the pylsl

        """
        symmetryIndex = self.symmetryIndex.get_symmetryindex()
        print('symmetry Index: ',symmetryIndex)

        # if self.previous_HR == 1000: # this is the first
        #     self.previous_HR = symmetryIndex

        # if len(self.cleaned) < 2000: #type: ignore
        #     self._logger.info(f"Not enough clean data {len(self.cleaned)}") #type: ignore
        #     return

        # if rmssd == -1:
        #     # something wrong with the data do not send
        #     return 


        # if self.skip_threshold < rmssd:
        #     self._logger.warn(f"rmssd is greater than the threshold: {rmssd}")
        #     self.store_data = np.array([])
        #     self.first_data = True
        #     return

        # symmetryIndex = - symmetryIndex # Multiplying by -1 because we want to maximize RMSSD
        self.outlet.push_sample([symmetryIndex])

        self.new_data = True

        self._logger.info(f"Sending the Symmetry Index value {symmetryIndex}")

class SymmetryIndex():
    def __init__(self, sampling_rate: int ) -> None:
        """Main processing class for the RMSSD data

        Args:
            sampling_rate (int): Sampling rate of the data
        """
        self.raw_data = np.array([])
        self.peaks = np.array([])
        self.SAMPLING_RATE = sampling_rate
        self.cleaned = np.array([])
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')  # Line for raw data
        self.peaks_plot, = self.ax.plot([], [], 'r^')  # Peaks in the data

    def update_plot(self, frame):
        if not hasattr(self, 'inlets') or not self.inlets:
            return self.line, self.peaks_plot
        
        # Assuming there's only one inlet for simplicity
        inlet = self.inlets[0]
        self.line.set_data(np.arange(len(inlet.symmetryIndex.raw_data)), inlet.symmetryIndex.raw_data)
        self.peaks_plot.set_data(inlet.symmetryIndex.peaks, inlet.symmetryIndex.raw_data[inlet.symmetryIndex.peaks])
        self.ax.set_xlim(0, 3000)  # Adjust according to your data length
        self.ax.set_ylim(-2000, 2000)  # Adjust according to your data amplitude
        self.ax.relim()  # Recompute the ax.dataLim
        self.ax.autoscale_view()  # Update the view limits

        return self.line, self.peaks_plot


        
    def add_data(self, data: np.ndarray) -> None:
        """Add data which needs to processed

        Args:
            data (np.ndarray): The ECG data in the form of np.ndarray
        """

        print('adding to self.raw_data')
        self.raw_data = data
        print('raw data: \n\n\n', self.raw_data)

    # def _clean_quality(self) -> None:

    #     """Clean the data and perform a quality check
    #     """
    #     self.cleaned = nk.ecg_clean(self.raw_data[-8000:], sampling_rate=self.SAMPLING_RATE)

    #     try:
    #         self.quality = nk.ecg_quality(self.cleaned, method = 'zhao', sampling_rate=self.SAMPLING_RATE)
    #     except ValueError:
    #         self.quality = 0


    
    def _process_data(self) -> float:
        """Process the cleaned data

        Returns:
            float: symmetry index mean
        """

        peaks, _ = scipy.signal.find_peaks(-self.raw_data[-24000:], height=1250, distance=75) #modify height and distance attribute depending on the postion of subject
        print(f'peaks: {peaks}')
        intervals = np.diff(peaks)

        stride_times_left = np.array([(intervals[i] + intervals[i + 1])/self.SAMPLING_RATE for i in range(0, len(intervals) - 2, 2)])
        stride_times_right =  np.array([(intervals[i + 1] + intervals[i + 2])/self.SAMPLING_RATE for i in range(0, len(intervals) - 2, 2)])
        step_time_left = np.array([intervals[i]/self.SAMPLING_RATE for i in range(0, len(intervals) - 2, 2)])
        step_time_right = np.array([intervals[i + 1]/self.SAMPLING_RATE for i in range(0, len(intervals) - 2, 2)])
        # print(f'step_time_left: {step_time_left}')            
        symmetry_index = abs((2 * (step_time_left - step_time_right) / (step_time_left + step_time_right)) * 100)
        # print(f'symmetry Index in process_data function {symmetry_index}')

        return symmetry_index.mean()
    
    

    def get_symmetryindex(self) -> float:
        """Send the processed Symmetry Index value"""
        return self._process_data()

    
class SymmetryIndexFromStream():
    def __init__(self, config: dict) -> None:
        """Class for getting RMSSD live from stream

        Args:
            config (dict): Configs parsed output of the yaml config files
        """
        config = config['RMSSD_config']
        self.inlets: List[InletOutlet] = []
        self.streams = pylsl.resolve_streams()
        self.wait_time = config['Pubrate']

        for info in self.streams:
            print(info.name(), info.name() == config['Stream_name'])
            if info.name() == config['Stream_name']:
                print("#" * 50)
                self.inlets.append(SymmetryIndexInOut(info, config['Data_buffer_length'], 
                        sampling_rate=config['Sampling_rate'], skip_threshold=config['Skip_threshold'], stream_name=config['Output_stream_name']))

    def run(self) -> None:
        print('run\n\n\n\n\n\n\n\n\n\n')
        """Main run ( in the while loop )
        """

        # This is the main while loop
        print(self.inlets)
        # self.ani = FuncAnimation(self.fig, self.update_plot, blit=True, interval=1000)
        # plt.show()
        while True:
            time.sleep(self.wait_time)
            for inlet in self.inlets:
                # self.ani = FuncAnimation(inlet.symmetryIndex.fig, inlet.symmetryIndex.update_plot, blit=True, interval=1000)
                # plt.show()
                inlet.get_data()
                print('store data length: ', len(inlet.store_data))
                # Checking the inlet data size and send the data to the pylsl stream.
                if len(inlet.store_data) > 500:
                    inlet.send_data()
                else:
                    logging.warn(f"{__name__}: no data to send")