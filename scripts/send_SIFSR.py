import asyncio
import threading
from HIL.cost_processing.ECG.SymmetryIndexFSR import SymmetryIndexFSRFromStream
import yaml
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger_blocklist = ["fiona", "rasterio", "matplotlib", "PIL"]
for module in logger_blocklist:
    logging.getLogger(module).setLevel(logging.WARNING)

# Load configuration
with open("configs/FSR.yml", 'r') as config_file:
    symmetryIndexFSRConfig = yaml.safe_load(config_file)

# Define function to run matplotlib in a separate thread
def run_plot(sym_index_stream):
    sym_index_stream.run_plotting()

if __name__ == "__main__":
    sym_index_stream = SymmetryIndexFSRFromStream(symmetryIndexFSRConfig)
    # plot_thread = threading.Thread(target=run_plot, args=(sym_index_stream,))
    # plot_thread.start()
    # sym_index_stream.run()
    # try:
    #     asyncio.run(sym_index_stream.run())
    # except KeyboardInterrupt:
    #     sym_index_stream.close_serial()
    #     print("Shutdown requested by user.")
    # finally:
    #     plot_thread.join()
    asyncio.run(sym_index_stream.run())
