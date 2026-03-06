"""
Connection module for Toptica DLC Pro laser controller.

Supports both Ethernet (TCP/IP) and USB serial connections.
"""

import struct
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection, SerialConnection


def decode_scope_data(raw_data: bytes) -> list[float]:
    """Decode binary scope data to float values."""
    # Header format: yNNNN\x00 where NNNN is length (e.g., 'y4088\x00')
    # Find the null byte that ends the header
    header_end = raw_data.find(b'\x00') + 1
    data = raw_data[header_end:]
    
    # Data is 32-bit floats (4 bytes each), little-endian
    num_floats = len(data) // 4
    return list(struct.unpack(f'<{num_floats}f', data[:num_floats*4]))


def connect_ethernet(ip_address: str, port: int = 1998) -> DLCpro:
    """
    Connect to DLC Pro via Ethernet.
    
    Args:
        ip_address: IP address of the DLC Pro (check device settings or DHCP lease)
        port: TCP port (default: 1998, taken from the device settings)
    
    Returns:
        DLCpro connection object
    """
    connection = NetworkConnection(ip_address, port)
    dlc = DLCpro(connection)
    return dlc


def connect_usb(device: str = "/dev/ttyUSB0", baudrate: int = 115200) -> DLCpro:
    """
    Connect to DLC Pro via USB serial.
    
    Args:
        device: Serial device path (e.g., /dev/ttyUSB0 or /dev/ttyACM0)
        baudrate: Baud rate (default: 115200)
    
    Returns:
        DLCpro connection object
    """
    connection = SerialConnection(device, baudrate)
    dlc = DLCpro(connection)
    return dlc


def print_laser_info(dlc: DLCpro):
    """Print basic laser information."""
    print(f"System Type: {dlc.system_type.get()}")
    print(f"Serial Number: {dlc.serial_number.get()}")
    print(f"Firmware Version: {dlc.fw_ver.get()}")
    print(f"Laser uptime: {dlc.uptime_txt.get()} s")
    print(f"laser1 type: {dlc.laser1.type.get()}")
    print(f"laser1 product: {dlc.laser1.product_name.get()}")

    print("")   #Emission
    dlc.laser1.dl.cc.enabled.set(False)      #Turn on laser emission
    print(f"Laser emission: {dlc.laser1.emission.get()}")

    print("")   #Current
    print(f"Laser current setpoint: {dlc.laser1.dl.cc.current_set.get()} mA")
    print(f"Laser current: {dlc.laser1.dl.cc.current_act.get()} mA")
    
    print("")   #Temperature
    print(f"Temperature: {dlc.laser1.dl.tc.temp_act.get()} °C")
    print(f"Temperature setpoint: {dlc.laser1.dl.tc.temp_set.get()} °C")

    print("")   #Piezo
    print(f"Piezo voltage: {dlc.laser1.scan.offset.get()} V")

    # print("")   #Wavelength
    # print(f"Wavelength setpoint: {dlc.laser1.ctl.wavelength_set.get()} nm")
    # print(f"Wavelength: {dlc.laser1.ctl.wavelength_act.get()} nm")
    """Our DLpro doesn't have the CTL (Continuous Tuning Loop) wavelength measurement module"""

    


if __name__ == "__main__":
    # --- Ethernet connection ---
    IP = "192.168.1.50"  
    
    with connect_ethernet(IP) as dlc:
        print_laser_info(dlc)
        
        # Read scope data
        print("")
        dlc.laser1.scope.channel1.signal.set(4)
        print("Signal name:", dlc.laser1.scope.channel1.name.get())
        print("Scope channel1 signal:", dlc.laser1.scope.channel1.signal.get())
        raw_data = dlc.laser1.scope.data.get()
        values = decode_scope_data(raw_data)
        print(f"Scope data ({len(values)} points): {values[:5]}")  # First 10 points
    
    # --- USB connection ---
    # with connect_usb("/dev/ttyUSB0") as dlc:
    #     print_laser_info(dlc)