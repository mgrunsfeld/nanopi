import RPi.GPIO as GPIO
import time
stepPIN = 3  #orange
dirPIN = 2 #yellow
m0PIN = 1 #brown
m1PIN = 0 #red
GPIO.setmode(GPIO.RAW)
GPIO.setwarnings(False)
GPIO.setup(stepPIN,GPIO.OUT,initial = 0)
GPIO.setup(dirPIN,GPIO.OUT, initial = 1)
GPIO.setup(m0PIN,GPIO.OUT, initial = 0)
GPIO.setup(m1PIN,GPIO.OUT, initial = 0)
freq = 240000
dc = 50
step = GPIO.PWM(3, freq)
step.start(dc)
time.sleep(10)
GPIO.output(dirPIN,0)
time.sleep(10)
GPIO.output(dirPIN,1)
GPIO.output(m1PIN,1)
time.sleep(10)
step.stop()
GPIO.cleanup()
	
