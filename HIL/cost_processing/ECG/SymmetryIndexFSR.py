import asyncio
import serial
import serial.tools.list_ports
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import find_peaks, peak_widths
import time
import pylsl
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def find_arduino_ports():
    ports = list(serial.tools.list_ports.comports())
    for port in ports:
        return port.device if 'Arduino' in port.description else None

def analyze_data(pressures):
    if not pressures:
        return [], []
    peaks, _ = find_peaks(pressures, prominence=50)
    widths, _, _, _ = peak_widths(pressures, peaks, rel_height=0.1)
    return peaks, widths

class SymmetryIndexFSRFromStream():
    def __init__(self, config: dict) -> None:
        self.config = config['RMSSD_config']
        # print(self.config)
        self.outlet = pylsl.StreamOutlet(pylsl.StreamInfo(
            self.config['Output_stream_name'], 'Marker', 1, 0, 'float32', 'myuidw43537'))
        
        self.port = find_arduino_ports()

        if self.port is None:
            raise Exception("Arduino not found on any port")

        # self.ser = serial.Serial(self.port, 9600) 
        self.ser = serial.Serial(self.port, 115200)
        self.ser.flushInput()

        self.pressures_fsr1 = []
        self.pressures_fsr2 = []

        # self.timestamps_fsr1 = []
        # self.timestamps_fsr2 = []

        self.timestamps_fsr = []

        # self.fig, self.ax = plt.subplots()
        # self.line1, = self.ax.plot([], [], 'r-', label='FSR1')
        # self.line2, = self.ax.plot([], [], 'b-', label='FSR2')
        # self.ax.set_xlim(0, 2400)
        # self.ax.set_ylim(0, 1024)  # assuming FSR output range 0-1024
        # self.ax.legend()
        # self.ax.set_xlabel('Sample Number')
        # self.ax.set_ylabel('Pressure')


    # def update_plot(self, frame):
    #     # Update plot data
    #     self.line1.set_data(list(range(len(self.timestamps_fsr1[-2400:]))), self.pressures_fsr1[-2400:])
    #     self.line2.set_data(list(range(len(self.timestamps_fsr2[-2400:]))), self.pressures_fsr2[-2400:])
        
    #     # Redraw the plot
    #     self.ax.draw_artist(self.ax.patch)
    #     self.ax.draw_artist(self.line1)
    #     self.ax.draw_artist(self.line2)
    #     return self.line1, self.line2
    

    # def run_plotting(self):
    #     self.fig, self.ax = plt.subplots()
    #     self.line1, = self.ax.plot([], [], 'r-', label='FSR1')
    #     self.line2, = self.ax.plot([], [], 'b-', label='FSR2')
    #     self.ax.legend()
    #     self.ani = FuncAnimation(self.fig, self.update_plot, interval=1000)
    #     plt.show()
 
    async def collect_data(self):
    # def collect_data(self):
        try:
            while True:
                if self.ser.in_waiting > 0:  # Check if there is data waiting in the buffer
                    # line = await asyncio.to_thread(self.ser.readline)
                    line = self.ser.readline()
                    line = line.decode().strip()
                    # if line.startswith('FSR1') or line.startswith('FSR2'):
                    #     tag, data = line.split(' ')
                    #     timestamp, pressure = data.split(', ')
                    #     pressure = float(pressure.rstrip(')'))
                    #     if tag == 'FSR1':
                    #         self.pressures_fsr1.append(pressure)
                    #         self.timestamps_fsr1.append(timestamp)
                    #     else:
                    #         self.pressures_fsr2.append(pressure)
                    #         self.timestamps_fsr2.append(timestamp)
                    if line.startswith('(') and line.endswith(')'):
                        timeStamp, pressureCount, pressure1, pressure2 = line[1:-1].split(',')
                        self.pressures_fsr1.append(pressure1)
                        self.pressures_fsr2.append(pressure2)
                        self.timestamps_fsr.append(timeStamp)
                        # self.timestamps_fsr2.append(timeStamp)
                        # print(self.pressures_fsr1)
                else:
                    await asyncio.sleep(0.01)  # Small delay to yield control and prevent tight loop
                    # time.sleep(0.01)
        except serial.SerialException as e:
            print(f"Serial error: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

    async def send_data(self):
    # def send_data(self):
        try:
            while True:
                await asyncio.sleep(self.config['Pubrate'])
                # time.sleep(self.config["Pubrate"])
                if self.pressures_fsr1 and self.pressures_fsr2:
                    # _, widths_fsr1 = analyze_data(self.pressures_fsr1[-2400:])
                    # _, widths_fsr2 = analyze_data(self.pressures_fsr2[-2400:])
                    # if widths_fsr1.size and widths_fsr2.size:
                    #     symmetry_index = abs(widths_fsr1.mean() - widths_fsr2.mean()) / ((widths_fsr1.mean() + widths_fsr2.mean()) / 2) * 100
                    # if len(pressures1) > 1 and len(pressures2) > 1:
                    pressures_fsr1 = self.pressures_fsr1[-2400:]
                    pressures_fsr2 = self.pressures_fsr2[-2400:]
                    timestamps_fsr = self.timestamps_fsr[-2400:]

                    peaks1, _ = find_peaks(pressures_fsr1, height=0)
                    peaks2, _ = find_peaks(pressures_fsr2, height=0)

                    time_diffs = []

                    for peak1 in peaks1:
                        next_peaks2 = [peak for peak in peaks2 if peak > peak1]
                        if next_peaks2:
                            next_peak2 = next_peaks2[0]
                            time_diff = timestamps_fsr[next_peak2] - timestamps_fsr[peak1]
                            time_diffs.append(time_diff)

                    for peak2 in peaks2:
                        next_peaks1 = [peak for peak in peaks1 if peak > peak2]
                        if next_peaks1:
                            next_peak1 = next_peaks1[0]
                            time_diff = timestamps_fsr[next_peak1] - timestamps_fsr[peak2]
                            time_diffs.append(time_diff)

                    if time_diffs:
                        symmetry_index = np.std(time_diffs) / np.mean(time_diffs) * 100
                        self.outlet.push_sample([symmetry_index])
                        print(f'Symmetry index sent: {symmetry_index}')
                    else:
                        print('No time differences calculated.')
                else:
                    print('Insufficient data to send.')
        except Exception as e:
            print(f"Error sending data: {e}")
    
    async def run_async_tasks(self):
        task1 = asyncio.create_task(self.collect_data())
        task2 = asyncio.create_task(self.send_data())
        await asyncio.gather(task1, task2)

    async def run(self):
        await self.run_async_tasks()
        # self.run_plotting()
        # self.collect_data()
        # self.send_data()
        # try:
        #     while True:
        #         if self.ser.in_waiting > 0:  # Check if there is data waiting in the buffer
        #             # line = await asyncio.to_thread(self.ser.readline)
        #             line = self.ser.readline()
        #             line = line.decode().strip()
        #             # if line.startswith('FSR1') or line.startswith('FSR2'):
        #             #     tag, data = line.split(' ')
        #             #     timestamp, pressure = data.split(', ')
        #             #     pressure = float(pressure.rstrip(')'))
        #             #     if tag == 'FSR1':
        #             #         self.pressures_fsr1.append(pressure)
        #             #         self.timestamps_fsr1.append(timestamp)
        #             #     else:
        #             #         self.pressures_fsr2.append(pressure)
        #             #         self.timestamps_fsr2.append(timestamp)
        #             if line.startswith('(') and line.endswith(')'):
        #                 timeStamp, pressureCount, pressure1, pressure2 = line[1:-1].split(',')
        #                 self.pressures_fsr1.append(pressure1)
        #                 self.pressures_fsr2.append(pressure2)
        #                 self.timestamps_fsr1.append(timeStamp)
        #                 self.timestamps_fsr2.append(timeStamp)
        #                 print(self.pressures_fsr1)
        #         # else:
        #         #     # await asyncio.sleep(0.01)  # Small delay to yield control and prevent tight loop
        #         #     time.sleep(0.01)
        #         # time.sleep(self.config["Pubrate"])
        #         if self.pressures_fsr1 and self.pressures_fsr2:
        #             _, widths_fsr1 = analyze_data(self.pressures_fsr1[-2400:])
        #             _, widths_fsr2 = analyze_data(self.pressures_fsr2[-2400:])
        #             if widths_fsr1.size and widths_fsr2.size:
        #                 symmetry_index = abs(widths_fsr1 - widths_fsr2) / ((widths_fsr1 + widths_fsr2) / 2) * 100
        #                 self.outlet.push_sample([symmetry_index.mean()])
        #                 print(f'Symmetry index sent: {symmetry_index.mean()}')
        #         else:
        #             print('Insufficient data to send.')
        #         time.sleep(self.config["Pubrate"])
        # except serial.SerialException as e:
        #     print(f"Serial error: {e}")
        # except Exception as e:
        #     print(f"Unexpected error: {e}")


    def close_serial(self):
        if self.ser.is_open:
            self.ser.close()