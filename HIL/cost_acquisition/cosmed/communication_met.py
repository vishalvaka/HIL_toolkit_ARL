import socket,logging,time,select
from unittest import expectedFailure
import numpy as np
import matplotlib.pyplot as plt
from signal import SIGINT, signal
from sys import exit
import time, logging
import pylsl


# object to create and initialize the hander class for sending
# prediction and stopping information
class MainCommunication(object):
    def __init__(self, target_ip = 'localhost', prediction_port = 50005,stopping_port = 50007):
        self.target_ip = 'localhost'
        self.prediction_port = prediction_port
        self.stopping_port = stopping_port
        self._prediction = _UDP(self.target_ip, self.prediction_port)
        self._stoping  = _UDP(self.target_ip, self.stopping_port)
        logging.debug(f'started comm port with ip {self.target_ip},\
                  prediction port {self.prediction_port}, stopping port \
                  {self.stopping_port}')
        info = pylsl.StreamInfo(name="Met_cost", type="Marker")
        self._pylsl_stream_out = pylsl.StreamOutlet(info)

    def send_all(self, prediction, stopping):
        print('sending the data to',self.prediction_port, self.stopping_port)
        self._prediction.send(prediction)
        if stopping == 1:
            self._stoping.send('STOP')

    def send_pred(self,prediction):
        print(self._pylsl_stream_out.channel_count)
        print(prediction)
        self._pylsl_stream_out.push_sample([prediction])
        self._prediction.send(prediction)
        # input('wait')

    def send_stopping(self, stoping):
        self._stoping.send(stoping)

    def close(self):
        self._prediction.close()
        self._stoping.close()


class MainReceiveCommunication(object):
    def __init__(self, target_ip = 'localhost', port = 30005):
        self._stopping = _UDP(ip = target_ip, receiving= True, receiving_ip = port)
        self.stream = None

    def pylsl_stop_setup(self):
        self.info = pylsl.StreamInfo(name="Change_parms", type = "Marker", source_id='12345')
        if self.stream is None:
            streams = pylsl.resolve_streams()
            
            for stream in streams:
                if "Change_parm" == stream.name():
                    self.stream = pylsl.StreamInlet(stream, recover=1)
                else:
                    self.stream = None

    def receive(self):
        data = self._stopping.receive()
        print(data, 'in receive')
        
        if self.stream is not None:
            
            try:
                sample, timeout = self.stream.pull_sample(timeout = 0.1)
            except:
                self.pylsl_stop_setup()
                print("Disconnected.")


            if timeout is not None:
                data = sample
        else:
            self.pylsl_stop_setup()
        
        if type(data) != type(None):
            print('returning True')
            return True
        else:
            return False
# base UDP class which will send the data to matlab or any socket
class _UDP():
    def __init__(self, ip = 'localhost', port = '5005', receiving = False, receiving_ip = 30005):
        logging.warning(f'starting port at {ip}, port {port}')
        # setup the communication
        self.sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
        self.sock.setblocking(0)
        self.port = port
        self.ip = ip
        if receiving:
            self.sock.bind(('127.0.0.1', receiving_ip))
            self.sock.settimeout(0.1)
            self.select = select.select([],[self.sock], [], 1)

    def send(self, i):
        # send the message by encode the string to bytes
        MESSAGE = str(i).encode('utf-8')
        self.sock.sendto(MESSAGE, (self.ip,self.port ))

    def close(self):
        logging.warning('closing the socket')
        self.sock.close()
    
    def receive(self):
        data = None
        try:
            if self.select[1]:
                data = self.sock.recv(1024) # buffer size is 1024 bytes
                while data != None:
                    data = self.sock.recv(1024) # buffer size is 1024 bytes
                    print(data.decode(), 'recieved')
                    new_data = data.decode()
                data = new_data
        except:
            pass
        return data







# simple script to test the communication
if __name__ == "__main__":
    MESSAGE = b'test'
    sock = socket.socket(socket.AF_INET, # Internet
                     socket.SOCK_DGRAM) # UDP
    sock.sendto(MESSAGE, ('localhost', 5005))
    an = _UDP()
    i = 1
    while i < 10000:
        i += 1
        an.send(i)
        time.sleep(1)
        print(i)
    an.close()

