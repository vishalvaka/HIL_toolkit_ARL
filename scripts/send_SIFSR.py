from HIL.cost_processing.ECG.SymmetryIndexFSR import SymmetryIndexFSRFromStream
import yaml
import logging



logging.basicConfig(level=logging.DEBUG)

logger_blocklist = [
    "fiona",
    "rasterio",
    "matplotlib",
    "PIL",
]

for module in logger_blocklist:
    logging.getLogger(module).setLevel(logging.WARNING)
config_file = open("configs/FSR.yml", 'r')

symmetryIndexFSRConfig = yaml.safe_load(config_file)

# cost function
symmetryIndex = SymmetryIndexFSRFromStream(config=symmetryIndexFSRConfig)

# start the cost function
symmetryIndex.run()