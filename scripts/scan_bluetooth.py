import asyncio
from bleak import BleakScanner

async def scan_for_devices():
    # Start the BLE scanner
    devices = await BleakScanner.discover()
    # Print details of each device found
    for device in devices:
        print(f"Device: {device.name}, MAC Address: {device.address}")

# Run the asynchronous function
loop = asyncio.get_event_loop()
loop.run_until_complete(scan_for_devices())