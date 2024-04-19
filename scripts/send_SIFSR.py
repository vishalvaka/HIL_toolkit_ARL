from HIL.cost_processing.ECG.SymmetryIndexFSR import SymmetryIndexFSRFromStream
import yaml
import logging
import asyncio



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
if __name__ == "__main__":
    sym_index_stream = SymmetryIndexFSRFromStream(symmetryIndexFSRConfig)
    try:
        asyncio.run(sym_index_stream.run())
    except KeyboardInterrupt:
        sym_index_stream.close_serial()
        print("Shutdown requested by user.")