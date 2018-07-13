import RPi.GPIO as GPIO
import time
stepPIN = 3  #orange
dirPIN = 2 #yellow
GPIO.setmode(GPIO.RAW)
GPIO.setwarnings(False)
GPIO.setup(3,GPIO.OUT)
GPIO.setup(2,GPIO.OUT, initial = 1)

step = GPIO.PWM(3,1)
step.start(50)
time.sleep(10)
step.stop()
	
GPIO.cleanup()
	
