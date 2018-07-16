import RPi.GPIO as GPIO
import time
stepPIN = 3  #orange
dirPIN = 2 #yellow
GPIO.setmode(GPIO.RAW)
GPIO.setwarnings(False)
GPIO.setup(3,GPIO.OUT,initial = 0)
GPIO.setup(2,GPIO.OUT, initial = 0)


for i in range(0,1000):
	GPIO.output(stepPIN,1)
	time.sleep(.000003)
	GPIO.output(stepPIN,0)
	time.sleep(.0003)
for i in range(1001,2000):
	GPIO.output(dirPIN, 1)
	GPIO.output(stepPIN,1)
	time.sleep(.000003)
	GPIO.output(stepPIN,0)
	time.sleep(.0003)
GPIO.cleanup()
	
