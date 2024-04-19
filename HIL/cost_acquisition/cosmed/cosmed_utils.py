import xml.etree.ElementTree as ET
import os,threading,time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import numpy as np
import time



# second class
class MyHandler(FileSystemEventHandler):
    def __init__(self,parser,data):
        self.parser = parser(data)

    def on_modified(self, event):
        self.new_data = True
        logging.debug(f'event type: {event.event_type}  path : {event.src_path}')
        self.parser.start_parser(event.src_path)

# third class
class ParserStorer(object):
    def __init__(self,data):
        self.data = data

    def start_parser(self,path):
        HR = 0
        vo = 0
        vco2 = 0
        self.data.STATUS = True
        try:
            main_root = ET.parse(path)
            main_root = main_root.getroot()
            for child in main_root:
                # print(child.tag, child.attrib)
                Omnia = child
            for child in Omnia:
                # print(child.tag, child.attrib)
                Realtime = child
            if Realtime[0].tag == 'VO2':
                logging.debug(f'{__name__},No Heart rate sensor detected')
                HR = 0
                vo = float(Realtime[0].text)
                vco2 = float(Realtime[1].text)
            if Realtime[0].tag == 'HR':
                HR = float(Realtime[0].text)
                vo = float(Realtime[1].text)
                vco2 = float(Realtime[2].text)
                logging.debug(f'{__name__},Heart rate information found,{HR}')
            self.data.load(np.array([vo,vco2]).reshape(2,1))
        except:
            pass


# Main class
class MainCosmed(threading.Thread):
    def __init__(self):
        self.path = str('C:\Program Files (x86)\COSMED\Omnia\Standalone\BCP\\')
        self.status = True
        self.data = None
        self.lock = threading.Lock()
        self.time = time.time()
        threading.Thread.__init__(self)
        logging.debug(f"{__name__}:cosmed connection created")

    def _timer_status(self):
        logging.debug(f"{__name__}:nothing changed")

    def run(self):
        self.data = DataStore()
        self.data.METABOLIC_PRESENT = True
        event_handler = MyHandler(ParserStorer,self.data)
        observer = Observer()
        observer.schedule(event_handler, path=self.path, recursive=False)
        observer.start()
        while self.status:
            time.sleep(0.1)
        observer.stop()

# data store class

class DataStore(object):
    def __init__(self):
        self.start_prediction = False
        self.start_index = 0
        self.time = time.time()
        self.data = np.array([0,0,0]).reshape(3,1)
        # This is the wether the new data is added
        self.STATUS = False

    def load(self,data):
        logging.debug('TEST,TEST')
        if data.shape != (2,1):
            logging.error(f'{__name__}, the data shape is {data.shape}, required shape is {(2,1)}')
        else:
            delta_time = time.time() - self.time
            data = np.append(data, delta_time).reshape(3,1)
            if self.data[0,-1] != data[0]:
                if self.data.shape[1] == 1 and self.data[2,0] == 0:
                    # removing the initial 0
                    self.data = data.reshape(3,1)
                self.data = np.append(self.data, data,axis = 1)
                logging.debug(f'{__name__} data stored,: {self.data[:,-1]} {self.data.shape}')
                if self.data.shape[1] > 8:
                    print(self.data.shape)
                    self.start_prediction = True

    def Read(self):
        self.start_prediction = False
        return self.data[:, self.start_index:]
    
    def reset(self):
        self.start_index =  len(self.data[0])
        logging.debug(f'{__name__}, data reset, start index is {self.start_index}')




if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    MainCosmed().run()
