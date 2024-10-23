import logging
import numpy as np
import math
import pylsl
import time
import os
import copy
import abc
# import antropy as ant

# processing
import neurokit2 as nk
import scipy
from HIL.cost_processing.ECG import ECGComplexity

# typing
from typing import List

from HIL.cost_processing.utils.inlet import InletOutlet

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Signal processing library
from scipy import signal
from scipy.signal import butter, filtfilt

# sampling rate of Polar accelerometer = 200 Hz
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
        info  = pylsl.StreamInfo('Polar_symmetry',  'Marker', 1, 0, 'float32', 'myuidw43537') #type: ignore
        self.outlet = pylsl.StreamOutlet(info)

        # logging
        self._logger = logging.getLogger()

        # # setting some large RMMSD value at the start.
        # self.previous_HR = 1000
        self.skip_threshold = skip_threshold
        self.SAMPLING_RATE = 200 # sampling_rate

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
            # self.store_data = np.array(self.buffer[0:ts.size,:]).T[0] # vertical acceleration
            self.store_data = np.array(self.buffer[0:ts.size,:]).T[2] # forward acceleration
            print('self.store_data.size ', self.store_data.size)
            self.first_data = False

        else:
            print('calling add data to instatialize self.raw_data')
            # self.store_data = np.append(self.store_data.flatten(), np.array(self.buffer[0:ts.size,:]).T[0].flatten()) # vertical acceleration
            self.store_data = np.append(self.store_data.flatten(), np.array(self.buffer[0:ts.size,:]).T[2].flatten()) # forward acceleration
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

        
        if not math.isnan(symmetryIndex):
            self.outlet.push_sample([symmetryIndex])

            self.new_data = True

            self._logger.info(f"Sending the Symmetry Index value {symmetryIndex}")
        else:
            print(f'got {symmetryIndex} from the calculation. please check.')

class SymmetryIndex():
    def __init__(self, sampling_rate: int ) -> None:
        """Main processing class for the Symmetry data

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

        # Initialize the animation
        # self.ani = FuncAnimation(self.fig, self.update_plot, blit=True, interval=100)

    # def update_plot(self, frame):
    #     if self.raw_data.size == 0:
    #         return self.line, self.peaks_plot
        
    #     data_to_plot = self.raw_data[-24000:] if self.raw_data.size > 24000 else self.raw_data
    #     self.line.set_data(np.arange(len(data_to_plot)), data_to_plot)
        
    #     peaks_to_plot = self.peaks[self.peaks >= len(self.raw_data) - len(data_to_plot)] - (len(self.raw_data) - len(data_to_plot))
    #     self.peaks_plot.set_data(peaks_to_plot, data_to_plot[peaks_to_plot])

    #     self.ax.set_xlim(0, len(data_to_plot))
    #     self.ax.set_ylim(min(data_to_plot), max(data_to_plot))
    #     self.ax.relim()
    #     self.ax.autoscale_view()

    #     return self.line, self.peaks_plot
        
    def add_data(self, data: np.ndarray) -> None:
        """Add data which needs to processed

        Args:
            data (np.ndarray): The ECG data in the form of np.ndarray
        """
        print('adding to self.raw_data')
        self.raw_data = data
        print('raw data: \n\n\n', self.raw_data)
    

    # Filtering step to smooth the signal and reduce noise
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def _process_data(self) -> float:
        """Process the cleaned data

        Returns:
            float: symmetry index mean
        """
        mean = np.mean(self.raw_data)
        std_dev = np.std(self.raw_data)

        # Set height as mean + some factor of standard deviation
        height = mean + 1 * std_dev

        # Set distance based on the typical spacing observed or a fraction of the signal length
        distance = len(self.raw_data) // 10

        # Filter the data
        # Sampling frequency (Hz)
        fs = 200  # same as self.SAMPLING_RATE (need to merge both later)

        # Low-pass filter parameters (cutoff frequency in Hz)
        cutoff = 5.0  # This will help smooth out high-frequency noise

        # Filtered signal
        filtered_signal = self.butter_lowpass_filter(-self.raw_data[-24000:], cutoff, fs) # last 2 minutes of the signal ~ 24,000 points considering 200 Hz sampling rate
        peaks, _ = scipy.signal.find_peaks(filtered_signal, height=-100, distance=75) # forward acceleration # modify height and distance attribute depending on the postion of subject
        
        # Unfiltered signal
        # peaks, _ = scipy.signal.find_peaks(-self.raw_data[-24000:], height=height, distance=distance) 
        # peaks, _ = scipy.signal.find_peaks(-self.raw_data[-24000:], height=-50, distance=75) # forward acceleration # modify height and distance attribute depending on the postion of subject
        
        self.peaks = peaks
        
        print(f'length of peaks: {len(peaks)}')
        intervals = np.diff(peaks)

        stride_times_left = np.array([(intervals[i] + intervals[i + 1])/fs for i in range(0, len(intervals) - 2, 2)])
        stride_times_right =  np.array([(intervals[i + 1] + intervals[i + 2])/fs for i in range(0, len(intervals) - 2, 2)])
        step_time_left = np.array([intervals[i]/fs for i in range(0, len(intervals) - 2, 2)])
        step_time_right = np.array([intervals[i + 1]/fs for i in range(0, len(intervals) - 2, 2)])
        print(f'step_time_left: {step_time_left}')            
        symmetry_index = abs((2 * (step_time_left - step_time_right) / (step_time_left + step_time_right)) * 100)
        print(f'symmetry Index in process_data function {symmetry_index.mean()}')
        
        # Step time array
        # Interleave the left and right step times
        combined_step_times = np.empty(step_time_left.size + step_time_right.size, dtype=step_time_left.dtype)
        combined_step_times[0::2] = step_time_left
        combined_step_times[1::2] = step_time_right

        # Stride time array
        # Interleave the left and right stride times
        combined_stride_times = np.empty(stride_times_left.size + stride_times_right.size, dtype=stride_times_left.dtype)
        combined_stride_times[0::2] = stride_times_left
        combined_stride_times[1::2] = stride_times_right

        # # Symmetry cost
        cost = symmetry_index.mean()
        
        # Step time variability
        # cost = np.std(combined_step_times)
        # cost = ECGComplexity.ETC(combined_step_times, "percentile")  # using effort-to-compress
        
        # # Stride time variability
        # cost = np.std(combined_stride_times)
        # cost = ECGComplexity.ETC(combined_stride_times, "percentile")  # using effort-to-compress

        # # Step time regularity (Detrended Fluctuation Analysis)
        # dfa_index_step_times = ant.detrended_fluctuation(combined_step_times)
        # cost = dfa_index_step_times

        # # Stride time regularity (Detrended Fluctuation Analysis)
        # dfa_index_stride_times = ant.detrended_fluctuation(combined_stride_times)
        # cost = dfa_index_stride_times

        # # Step time mean
        # cost = np.mean(combined_step_times)

        # # Stride time mean
        # cost = np.mean(combined_stride_times)

        return cost
    
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
        while True:
            time.sleep(self.wait_time)
            for inlet in self.inlets:
                inlet.get_data()
                print('store data length: ', len(inlet.store_data))
                # Checking the inlet data size and send the data to the pylsl stream.
                if len(inlet.store_data) > 500:
                    inlet.send_data()
                    # plt.show()
                else:
                    logging.warn(f"{__name__}: no data to send")