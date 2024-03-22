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
from HIL.cost_processing.ECG import ECGComplexity

# typing
from typing import List

from HIL.cost_processing.utils.inlet import InletOutlet

import matplotlib.pyplot as plt

class ETCInOut(InletOutlet):
    dtypes = [[], np.float32, np.float64, None, np.int32, np.int16, np.int8, np.int64]

    def __init__(self, info: pylsl.StreamInfo, data_length: int, sampling_rate: int = 133) -> None:
        """Main class which handles the input of the ECG data and output of the ETC data to the pylsl.

        Args:
            info (pylsl.StreamInfo): information aboute the input stream
            data_length (int): Data length to process to get the ETC, generally 4000 -> ~30s considering 133 Hz sampling rate
            sampling_rate (int, optional): Sampling rate of the signal. Defaults to 133.

        """
        super().__init__(info, data_length)
        buffer_size = (2 * math.ceil(info.nominal_srate() * data_length), info.channel_count())
        self.buffer = np.empty(buffer_size, dtype = self.dtypes[info.channel_format()])

        # placeholders
        self.store_data = np.array([])

        # Information about the outlet stream
        # info  = pylsl.StreamInfo('ECG_complexity', 'Marker', 1, 0, 'float32', 'myuidw43539') #type: ignore
        self.outlet = pylsl.StreamOutlet(pylsl.StreamInfo('ECG_complexity', 'Marker', 1, 0, 'float32', 'myuidw43539'))

        # logging
        self._logger = logging.getLogger()

        # setting some large ETC value at the start.
        self.previous_HR = 1000
        self.SAMPLING_RATE = sampling_rate

        # Main processing class
        self.etc = ETC(self.SAMPLING_RATE)
        
        # flags
        self.first_data = True
        self.cleaned = np.array([])


    def get_data(self) -> None:
        """Class to get the ECG data and process it

        Returns:
            None: early return if the stream if not ECG
        """

        print('in get data')
        _, ts = self.inlet.pull_chunk(timeout = 0.0, 
                max_samples=self.buffer.shape[0],
                dest_obj=self.buffer)

        if not ts or self.name != "polar ECG 2":
            print('ts ', ts)
            self._logger.warning(f"Time stamp is: {ts}, name of the stream: {self.name}")
            return None
        
        ts = np.array(ts)
        # For first time
        if self.first_data:
            self.store_data = self.buffer[0:ts.size,:]
            self.first_data = False

        else:
            self.store_data = np.append(self.store_data.flatten(), self.buffer[0:ts.size].flatten())
            self.etc.add_data(self.store_data)
            

            # check if there is a nan in the data
            if np.isnan(self.store_data).any(): #type: ignore
                self._logger.warn(f"nan found in data")

    def send_data(self) -> None:
        """Send the processed data to the pylsl

        """
        etc = self.etc.get_etc()

        if self.previous_HR == 1000: # this is the first
            self.previous_HR = etc

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
        etc = -etc # multiplying by -1 because we want to maximize ETC 
        self.outlet.push_sample([etc])

        self.new_data = True

        self._logger.info(f"Sending the ETC value {etc}")

class ETC():
    def __init__(self, sampling_rate: int ) -> None:
        """Main processing class for the ETC data

        Args:
            sampling_rate (int): Sampling rate of the data
        """
        self.SAMPLING_RATE = sampling_rate
        self.cleaned = np.array([])
        # self.raw_data = None
        
    def add_data(self, data: np.ndarray) -> None:
        """Add data which needs to processed

        Args:
            data (np.ndarray): The ECG data in the form of np.ndarray
        """
        self.raw_data = data

    def _clean_quality(self) -> None:

        """Clean the data and perform a quality check
        """
        self.cleaned = nk.ecg_clean(self.raw_data[-16000:], sampling_rate=self.SAMPLING_RATE)
        try:
            self.quality = nk.ecg_quality(self.cleaned, method = 'zhao', sampling_rate=self.SAMPLING_RATE)
        except ValueError:
            self.quality = 0

    def _process_data(self) -> float:
        """Process the cleaned data

        Returns:
            float: ETC value
        """
        if self.cleaned is None:
            return -1
        
        self._clean_quality()
        cleaned = copy.copy(self.cleaned)

        peaks, info = nk.ecg_peaks(cleaned, sampling_rate = self.SAMPLING_RATE)
        
        # Access indices of R peaks
        peaks_idx = info['ECG_R_Peaks']  

        # Get successive differences (RR intervals)
        res = []
        res = [peaks_idx[i + 1] - peaks_idx[i] for i in range(len(peaks_idx) - 1)]
        peaks_RR = res
    
        # Convert RR intervals to milliseconds
        peaks_RR = np.asarray(peaks_RR[1:], dtype=np.float32)
        peaks_ms = np.round(peaks_RR/self.SAMPLING_RATE*1000,0) # in Hz sampling rate
        print("RR intervals:", peaks_ms)
        
        try:
            # Calculate ETC
            if len(peaks_ms)>3: # If length of the input series is less than 3, there will be errors in the ETC calculation
                cost = ECGComplexity.ETC(peaks_ms, "difference")
                # P0V, P1V, P2V = ECGComplexity.symbolic_dynamics(peaks_ms, "difference")
                # cost = P0V  # Percentage of '0V' patterns
            else:
                cost = 0
            return cost
        
        except IndexError:
            return -1

        

    def get_etc(self) -> float:
        """Send the processed ETC value"""
        return self._process_data()

    
class ETCFromStream():
    def __init__(self, config: dict) -> None:
        """Class for getting ETC live from stream

        Args:
            config (dict): Configs parsed output of the yml config files
        """
        config = config['ETC_config']
        self.inlets: List[InletOutlet] = []
        self.streams = pylsl.resolve_streams()
        self.wait_time = config['Pubrate']

        for info in self.streams:
            print(info.name(), info.name() == config['Stream_name'])
            if info.name() == config['Stream_name']:
                print("#" * 50)
                self.inlets.append(ETCInOut(info, config['Data_buffer_length'], 
                        sampling_rate=config['Sampling_rate']))

    def run(self) -> None:
        """Main run ( in the while loop )
        """

        # This is the main while loop
        while True:
            time.sleep(self.wait_time)
            for inlet in self.inlets:
                inlet.get_data()
                # Checking the inlet data size and send the data to the pylsl stream.
                if len(inlet.store_data) > 250:
                    inlet.send_data()
                else:
                    logging.warn(f"{__name__}: no data to send")


'''ETCInOut class takes streamInfo class which is the inlet stream.
   but it is being overwritten in the constructor'''