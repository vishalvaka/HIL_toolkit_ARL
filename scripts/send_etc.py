from HIL.cost_processing.ECG.ETC import ETCFromStream
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
config_file = open("configs/ETC.yml", 'r')

etc_config = yaml.safe_load(config_file)

# cost function
etc = ETCFromStream(config=etc_config)

# start the cost function
etc.run()