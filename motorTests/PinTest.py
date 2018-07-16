import RPi.GPIO as GPIO
import time
stepPIN = 3  #orange
dirPIN = 2 #yellow
GPIO.setmode(GPIO.RAW)
GPIO.setwarnings(False)
GPIO.setup(3,GPIO.OUT,initial = 1)
GPIO.setup(2,GPIO.OUT, initial = 1)


for i in range(0,10):
	GPIO.output(stepPIN,0)
	GPIO.output(dirPIN,0)
	#time.sleep(.000003)
	time.sleep(1)
	GPIO.output(stepPIN,1)
	GPIO.output(dirPIN,1)
	#time.sleep(.000003)
	time.sleep(1)
GPIO.cleanup()
	
