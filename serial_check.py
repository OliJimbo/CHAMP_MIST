import serial
import time
import numpy as np
import random
random.seed(17954362)
ints = np.arange(1,256)
random.shuffle(ints)
try:
    sr=serial.Serial("COM8", 115200, timeout=1)
    
    print("Serial Connected")
except serial.serialutil.SerialException:
    print("Serial Not Connected")
evntlg = 'sr' in locals() or 'sr' in globals()
if evntlg :
    sr.write('RR'.encode())
    time.sleep(0.025)
    sr.write('00'.encode())


for i in ints:
    thisbyte=str(hex(i).split('x')[-1]).encode()
    if evntlg: 
        sr.write(thisbyte)
        time.sleep(0.025)
    else: print(thisbyte)
    if evntlg:
        sr.write('RR'.encode())
        time.sleep(0.025)
    else: print('RR'.encode())

if evntlg:
    sr.flush()
    sr.close()