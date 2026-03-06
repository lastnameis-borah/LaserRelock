import sys
from toptica.lasersdk.dlcpro.v2_0_3 import DLCpro, NetworkConnection

# Toptica testbed689 laser
TOPTICA_IP = "172.29.13.247"
TOPTICA_PORT = 1998
LASER_NAME = "testbed689"

print(f"Connecting to {LASER_NAME} at {TOPTICA_IP}...")
conn = NetworkConnection(TOPTICA_IP, TOPTICA_PORT)
dlc = DLCpro(conn)

try:
    dlc.open()
    print(f"Connected to {LASER_NAME} (SN: {dlc.serial_number.get()})\n")

    lock = dlc.laser1.dl.lock

    # Lock state
    state = lock.state.get()
    state_txt = lock.state_txt.get()
    print(f"Lock State:   {state_txt} (code: {state})")

    # Lock enabled
    enabled = lock.lock_enabled.get()
    print(f"Lock Enabled: {enabled}")

    # Setpoint
    setpoint = lock.setpoint.get()
    print(f"Lock Setpoint: {setpoint}")

    # Unlock the laser
    print("\n--- Unlocking laser ---")
    lock.lock_enabled.set(True)

    # Confirm new state
    state_txt = lock.state_txt.get()
    enabled = lock.lock_enabled.get()
    print(f"Lock State:   {state_txt}")
    print(f"Lock Enabled: {enabled}")

    # Piezo voltage
    piezo = dlc.laser1.dl.pc.voltage_act.get()
    print(f"Piezo Voltage: {piezo:.3f} V")

    # Current
    current = dlc.laser1.dl.cc.current_act.get()
    print(f"Laser Current: {current:.2f} mA")

except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
finally:
    dlc.close()
    print("\nDone.")
