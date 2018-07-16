import RPi.GPIO as GPIO
import time
stepPIN = 3  #orange
dirPIN = 2 #yellow
GPIO.setmode(GPIO.RAW)
GPIO.setwarnings(False)
GPIO.setup(3,GPIO.OUT,initial = 0)
GPIO.setup(2,GPIO.OUT, initial = 1)
freq = 240000
dc = 50
step = GPIO.PWM(3, freq)
step.start(dc)
time.sleep(15)
step.stop()
GPIO.cleanup()
	
