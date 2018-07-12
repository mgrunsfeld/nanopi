import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)
channel = 24
n=0

GPIO.setup(channel,GPIO.OUT, initial = 1)
while n<5:
	time.sleep(1)
	GPIO.output(channel,1)
	time.sleep(1)
	GPIO.output(channel,0)
	n = n+1
GPIO.cleanup(channel)
