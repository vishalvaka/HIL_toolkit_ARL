from HIL.cost_processing.ECG.SymmetryIndex import SymmetryIndexFromStream
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
config_file = open("configs/accel.yml", 'r')

symmetryIndexConfig = yaml.safe_load(config_file)

# cost function
symmetryIndex = SymmetryIndexFromStream(config=symmetryIndexConfig)

# start the cost function
symmetryIndex.run()