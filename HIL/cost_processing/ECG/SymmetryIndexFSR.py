import asyncio
import serial
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks, peak_widths
import time
import pylsl

def analyze_data(timestamps, pressures):
    if pressures:
        peaks, _ = find_peaks(pressures, prominence=50)
        widths, width_heights, left_ips, right_ips = peak_widths(pressures, peaks, rel_height=0.1)
        return peaks, widths
    else:
        return [], []

class SymmetryIndexFSRFromStream():
    def __init__(self, config: dict) -> None:
        self.config = config['RMSSD_config']

        info = pylsl.StreamInfo(config['Output_stream_name'], 'Marker', 1, 0, 'float32', 'myuidw43537')  # type: ignore
        self.outlet = pylsl.StreamOutlet(info)

        self.ser = serial.Serial('COM3', 9600)
        self.ser.flushInput()

        self.timestamps_fsr1 = []
        self.pressures_fsr1 = []

        self.timestamps_fsr2 = []
        self.pressures_fsr2 = []

    async def collect_data(self):
        while True:
            line = await asyncio.to_thread(self.ser.readline)
            line = line.decode().strip()

            if line.startswith('FSR1'):
                data_str = line[len('FSR1 '):]
                _, pressure_str = data_str.split(', ')
                timestamp = datetime.now()
                pressure = float(pressure_str.rstrip(')'))
                self.timestamps_fsr1.append(timestamp)
                self.pressures_fsr1.append(pressure)

            elif line.startswith('FSR2'):
                data_str = line[len('FSR2 '):]
                _, pressure_str = data_str.split(', ')
                timestamp = datetime.now()
                pressure = float(pressure_str.rstrip(')'))
                self.timestamps_fsr2.append(timestamp)
                self.pressures_fsr2.append(pressure)

    async def send_data(self):
        while True:
            await asyncio.sleep(self.config['Pubrate'])

            if len(self.timestamps_fsr1) > 300 and len(self.timestamps_fsr2) > 300:
                if self.pressures_fsr1 and self.pressures_fsr2:
                    _, widths_fsr1 = analyze_data(self.timestamps_fsr1, self.pressures_fsr1)
                    _, widths_fsr2 = analyze_data(self.timestamps_fsr2, self.pressures_fsr2)

                    if widths_fsr1.size and widths_fsr2.size:
                        # avg_width_fsr1 = widths_fsr1.mean()
                        # avg_width_fsr2 = widths_fsr2.mean()
                        if len(widths_fsr1) > len(widths_fsr2):
                            widths_fsr1 = widths_fsr1[:len(widths_fsr2)]
                        else:
                            widths_fsr2 = widths_fsr2[:len(widths_fsr1)]
                        symmetry_index = abs(widths_fsr1 - widths_fsr2) / ((widths_fsr1 + widths_fsr2) / 2) * 100
                        print(f'sending the symmetry index value: {symmetry_index.mean()}')
                        self.outlet.push_sample([symmetry_index.mean()])
            else:
                print('no data to send')

    async def run(self):
        task1 = asyncio.create_task(self.collect_data())
        task2 = asyncio.create_task(self.send_data())
        await task1
        await task2

    def close_serial(self):
        self.ser.close()


    
    
    

# Open the serial port
# ser = serial.Serial('COM3', 9600)
# ser.flushInput()

# timestamps_fsr1 = []
# pressures_fsr1 = []

# timestamps_fsr2 = []
# pressures_fsr2 = []

# try:
#     # User input for duration of data collection
#     duration_minutes = int(input("Enter the duration of data collection (minutes): "))
    
#     # Calculate end time for data collection
#     end_time = time.time() + duration_minutes * 60
    
#     # Data collection loop
#     while time.time() < end_time:
#         line = ser.readline().decode().strip()
        
#         if line.startswith('FSR1'):
#             data_str = line[len('FSR1 '):]
#             _, pressure_str = data_str.split(', ')
#             timestamp = datetime.now()
#             pressure = float(pressure_str.rstrip(')'))
#             timestamps_fsr1.append(timestamp)
#             pressures_fsr1.append(pressure)
        
#         elif line.startswith('FSR2'):
#             data_str = line[len('FSR2 '):]
#             _, pressure_str = data_str.split(', ')
#             timestamp = datetime.now()
#             pressure = float(pressure_str.rstrip(')'))
#             timestamps_fsr2.append(timestamp)
#             pressures_fsr2.append(pressure)
    
#     # Close the serial port
#     ser.close()
    
#     def analyze_data(timestamps, pressures):
#         peaks, _ = find_peaks(pressures, prominence=50)  
#         widths, width_heights, left_ips, right_ips = peak_widths(pressures, peaks, rel_height=0.5)
#         return peaks, widths
    
#     # Analyze FSR data
#     peaks_fsr1, widths_fsr1 = analyze_data(timestamps_fsr1, pressures_fsr1)
#     peaks_fsr2, widths_fsr2 = analyze_data(timestamps_fsr2, pressures_fsr2)
    
#     # Calculate symmetry index
#     avg_width_fsr1 = widths_fsr1.mean()
#     avg_width_fsr2 = widths_fsr2.mean()
#     symmetry_index = abs(avg_width_fsr1 - avg_width_fsr2) / ((avg_width_fsr1 + avg_width_fsr2) / 2) * 100

#     print(f"Average width FSR1: {avg_width_fsr1} seconds")
#     print(f"Average width FSR2: {avg_width_fsr2} seconds")
#     print(f"Symmetry Index: {symmetry_index:.2f}%")

# except KeyboardInterrupt:
#     print("Data collection interrupted by user.")
# finally:
#     # Ensure the serial port is closed on exit
#     ser.close()
