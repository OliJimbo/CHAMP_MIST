import serial
import time
import numpy as np

ints = np.arange(1,256)
try:
    sr=serial.Serial("COM8", 115200, timeout=1)
    
    print("Serial Connected")
except serial.serialutil.SerialException:
    print("Serial Not Connected")
evntlg = 'sr' in locals() or 'sr' in globals()
if evntlg :
    sr.write('RR'.encode())
    time.sleep(1)
    sr.write('00'.encode())

if evntlg :
    ("FF".encode())
    time.sleep(0.1)
    sr.write('RR'.encode())
    
for i in ints:
    thisbyte=str(hex(i).split('x')[-1]).encode()
    if evntlg: 
        sr.write(thisbyte)
        time.sleep(0.025)
    else: print(thisbyte)
    if evntlg:
        sr.write('RR'.encode())
7        time.sleep(0.025)
    else: print('RR'.encode())

if evntlg:
    sr.flush()
    sr.close()