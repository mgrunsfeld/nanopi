import serial
import time

ser = serial.Serial('/dev/ttyS1',57600,timeout = 2)

while True:
	name = ser.read(2)
	if str(name) == 'A1':
		ser.write('\r\n You called?\r\n')
		continue
