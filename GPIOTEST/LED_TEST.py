import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)
channel =(5,13,15,24)
#channel = int(channel)
n = 0
for i in channel:
	GPIO.setup(i,GPIO.OUT,initial = 1)
#GPIO.setup(channel,GPIO.OUT,initial = 1)
while n <5:
	time.sleep(1)
	for i in channel:
		GPIO.output(i,0)
	time.sleep(1)
	for i in channel:
		GPIO.output(i,1)
	n = n+1
for i in channel:
	GPIO.cleanup(i)
