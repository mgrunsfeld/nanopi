import RPi.GPIO as GPIO
import time
import serial
import smbus
import numpy as np

GPIO.setmode(GPIO.RAW)
GPIO.setwarnings(False)

#program parameters
z = np.zeros(4)					#dynamics vector [z, zdot, zddot, zdddot] in [m m/s m/s2 m/s3]
zd = np.zeros(4)				#vector of desired dynamics [z, zdot, zddot, zdddot] in [m m/s m/s2 m/s3]
P_target = 0					#target pressure in [hPa]
P_0 = 0							#initial pressure in [hPa]


#LED pin number declaration
ACTIVE_LED = 200
SWARM_CONNECTION_LED = 201
SMC_LED = 1
DATA_LED = 67

#Motor Controller 
#M1 = ??? 		#driven high by 3.3v
#M0 = ??? 		#left floating  
STEP = 2
DIR = 3
SLEEP = 0
freq = 240000
dc = 50

#Push Button Pin Declaration
Switch_Pin = 6

#Pressure Sensor Pin Declaration
bus = smbus.SMBus(1)

#RF Sensor Declaration 
ser = serial.Serial('/dev/ttyS1',57600,timeout = 2)  #timeout might need to change

#SLIDING-MODE-OBSERVER (SMO)
#Declarations
#Model parameters
SMO_rho_default = 2.5       #observer parameter
SMO_tau_default = 0.2       #observer parameter
SMO_phi_default = 0.5       #observer parameter

#coordinates
xhat1 = 0.0                 #estimated depth in [m]
xhat2 = 0.0                 #estimated velocity in [m/s]
xhat1_prev = [0.0, 0.0]  #estimated depth at previous time step in [m]
xhat2_prev = [0.0, 0.0] #estimated velocity at previous time step in [m/s]

#time parameters
timestep = 0					#counter for ellapsed time steps
iterationTime = .025			#time per loop [ms]
iterationtimeSC = .025			#loop time under SC control
maxOverallTime = 0				#max program time	
			
#parameters of SC control
SMC_lambda_default = 0.40
SMC_eta_default = 0.10
SMC_phi_default = 0.50
SMC_a_default = 20.0
SMC_timeMax_default = 300  		#runs for 5 minutes
zdesired_default = .3 			#30 cm deep
z_prev = 0

SinAmp_default = .5
SinPeriod_default = 10
StepTime_default = 5
StepAmp_default = .5

#Variables
stepMode      = 32                         #step modes 1=full step; 2=half step, 4=1/4 step, 8 =1/8 step, 32 = 1/32 step;
stepFrequency = 500 * stepMode            #max step frequency allowed [Hz] (full step=500, half step 1000, 1/4 step = 2000)
iplanet       = 256                        #transmission of the gear
anglePerStep  = (18/stepMode)            #rotation angle of the motor per step [deg], depends on the step mode
alpha		=  0                                      #desired angle of the motor [deg]
alpha_max     = 130                        #max angle of the motor [deg]
angle		= 0                                      #acutal angle of the motor [deg]. Start in fully extended position
stepTime      = (1000000 / stepFrequency)    #time the motor has for a step in [µs]
stepCounter   = 0                          #counter of steps starting for angle 0 [deg]
Switch_Steps  = 20 * stepMode              #number of steps from position reset to neutral position

#Battery voltage control
#batteryPin = ???    #pin for measuring the battery voltage
vBat = 0                  #voltage of LiPo in [V]
vBat_old = 0              #voltage of LiPo in [V]

#Pressure Sensor Functions
def calibratePressure():
	global C1
	global C2
	global C3
	global C4
	global C5
	global C6
	# MS5803_01BA address, 0x77(118)
	#		0x1E(30)	Reset command
	bus.write_byte(0x77, 0x1E)

	time.sleep(0.5)

	# Read 12 bytes of calibration data
	# Read pressure sensitivity
	data = bus.read_i2c_block_data(0x77, 0xA2, 2)
	C1 = data[0] * 256 + data[1]

	# Read pressure offset
	data = bus.read_i2c_block_data(0x77, 0xA4, 2)
	C2 = data[0] * 256 + data[1]

	# Read temperature coefficient of pressure sensitivity
	data = bus.read_i2c_block_data(0x77, 0xA6, 2)
	C3 = data[0] * 256 + data[1]

	# Read temperature coefficient of pressure offset
	data = bus.read_i2c_block_data(0x77, 0xA8, 2)
	C4 = data[0] * 256 + data[1]

	# Read reference temperature
	data = bus.read_i2c_block_data(0x77, 0xAA, 2)
	C5 = data[0] * 256 + data[1]

	# Read temperature coefficient of the temperature
	data = bus.read_i2c_block_data(0x77, 0xAC, 2)
	C6 = data[0] * 256 + data[1]

	# MS5803_01BA address, 0x77(118)
	#		0x40(64)	Pressure conversion(OSR = 256) command
	bus.write_byte(0x77, 0x40)
	time.sleep(0.5) 

def calculatePressure():
	# Read digital pressure value
	# Read data back from 0x00(0), 3 bytes
	# D1 MSB2, D1 MSB1, D1 LSB
	value = bus.read_i2c_block_data(0x77, 0x00, 3)
	D1 = value[0] * 65536 + value[1] * 256 + value[2]

	# MS5803_01BA address, 0x77(118)
	#		0x50(64)	Temperature conversion(OSR = 256) command
	bus.write_byte(0x77, 0x50)

	time.sleep(0.5)

	# Read digital temperature value
	# Read data back from 0x00(0), 3 bytes
	# D2 MSB2, D2 MSB1, D2 LSB
	value = bus.read_i2c_block_data(0x77, 0x00, 3)
	D2 = value[0] * 65536 + value[1] * 256 + value[2]

	dT = D2 - (C5) * 256
	TEMP = 2000 + dT * (C6) / 8388608
	OFF = (C2) * 65536 + ((C4) * dT) / 128
	SENS = (C1) * 32768 + ((C3) * dT ) / 256
	T2 = 0
	OFF2 = 0
	SENS2 = 0

	if TEMP >= 2000 :
		T2 = 0
		OFF2 = 0
		SENS2 = 0
		if TEMP > 4500 :
			SENS2 = SENS2 - ((TEMP - 4500) * (TEMP - 4500)) / 8
	elif TEMP < 2000 :
		T2 = (dT * dT) / 2147483648
		OFF2 = 3 * ((TEMP - 2000) * (TEMP - 2000))
		SENS2 = 7 * ((TEMP - 2000) * (TEMP - 2000)) / 8
		if TEMP < -1500 :
			SENS2 = SENS2 + 2 * ((TEMP + 1500) * (TEMP + 1500))

	TEMP = TEMP - T2
	OFF = OFF - OFF2
	SENS = SENS - SENS2
	pressure = ((((D1 * SENS) / 2097152) - OFF) / 32768.0) / 100.0
	# pressure value = calculatePressure(pressure)
	cTemp = TEMP / 100.0
	fTemp = cTemp * 1.8 + 32
	return pressure

#LED functions
def allLEDLow():
	GPIO.output(ACTIVE_LED, 0)
	GPIO.output(SWARM_CONNECTION_LED, 0)
	GPIO.output(SMC_LED, 0)
	GPIO.output(DATA_LED, 0)
	
def allLEDHigh():
	GPIO.output(ACTIVE_LED, 1)
	GPIO.output(SWARM_CONNECTION_LED, 1)
	GPIO.output(SMC_LED, 1)
	GPIO.output(DATA_LED, 1)
	
def LEDBlink(number,duration,delaytime):
	for i in range(number):
		digitalWrite(ACTIVE_LED, 1)
		digitalWrite(SWARM_CONNECTION_LED, 1)
		digitalWrite(SMC_LED, 1)
		digitalWrite(DATA_LED, 1)
		time.sleep(duration/1000)
		digitalWrite(ACTIVE_LED, 0)
		digitalWrite(SWARM_CONNECTION_LED, 0)
		digitalWrite(SMC_LED, 0)
		digitalWrite(DATA_LED, 0)
		time.sleep((delaytime - duration)/1000)
		
def SMC_LEBBlink(Delay):
	digitalWrite(ACTIVE_LED, 1)
	digitalWrite(SMC_LED, 1)
	time.sleep(200/1000)
	digitalWrite(ACTIVE_LED, 0)
	digitalWrite(SMC_LED, 0)
	time.sleep((Delay - 200)/1000)

#This function drives the motor to a desired alpha
#Thereby the whole looptime is used
#A ramp function for the step frequency ia used for better dynamics
def step(alpha,angle,stepCounter):
	global stepCounter
	global angle
	global alpha
	angle = ((stepCounter) * (anglePerStep)) / (iplanet)
	dalpha = alpha - angle
	requiredSteps = (int)abs((dalpha) * iplanet / anglePerStep)
	localStepCounter = 0
	accelerationSteps = 15 * stepMode
	loopCheck = time.time()
	looptime = (loopCheck - loopTimeStart)*1000 #1000 to make it ms
	
	while (looptime < (iterationtime - 1)):
		if (requiredSteps > 1):
			if (dalpha > 0):
				GPIO.output(DIR, 1)
				while ((localStepCounter <= requiredSteps) and (looptime < (iterationtime - 1)) and  angle < alpha):
					motor_step.start(dc)
					time.sleep(0.00000833333)  #1 full cycle: (1/240000 Hz)*2(for 
					motor_step.stop()
					if (localStepCounter < accelerationSteps):
							time.sleep(stepTime - .004)
					else:
							time.sleep(stepTime - .004)
					stepCounter += 1
					localStepCounter+= 1
					angle = ((stepCounter) * (anglePerStep)) / (iplanet)
					
			elif (dalpha < 0):
				GPIO.output(DIR, 0)
				while ((localStepCounter <= requiredSteps) and (looptime < (iterationtime - 1)) and angle >= 0 and angle > alpha):
					motor_step.start(dc)
					time.sleep(0.00000833333)  #1 full cycle: (1/240000 Hz)*2(for 
					motor_step.stop()
					if (localStepCounter < accelerationSteps): #minimal 13 steps
						time.sleep(stepTime - .004)
					else:
						time.sleep(stepTime - .004) # Halbe Schrittzeit warten
					stepCounter -= 1
					#reset calibration if push botton is pressed
					if (GPIO.input(Switch_Pin) == True):
						stepCounter = -Switch_Steps
					localStepCounter+=1
					angle = ((stepCounter) * (anglePerStep)) / (iplanet)
	return angle
	
#Motor reset function
def motorResetPosition(alpha, angle, stepCounter):
	resetOk = 0                      #1 when reset successful
	global stepCounter
	global angle
	global alpha
	#reset loop
	while resetOk != 1:
		
		#if push button is reached set stepCounter to -Switch_Steps
		if GPIO.input(Switch_Pin) == True:
			stepCounter = -Switch_Steps
			angle = ((stepCounter) * (anglePerStep)) / (iplanet)
			#drive stepCounter to zero
			time.sleep(.200)
		  
			while stepCounter < 0:
				GPIO.output(DIR, 1)
				motor_step.start(dc)
				time.sleep(0.00000833333)  #1 full step: (1/240000 Hz)*2
				motor_step.stop()
				time.sleep(stepTime - 0.00000833333)
				stepCounter += 1
			
			#drive motor to desired alpha
			while abs(angle - alpha) > 0.08:
				angle = step(alpha, angle, stepCounter)
				loopTimeStart = time.time()
			print("position reset ok")
			resetOk = 1
			#drive in the direction of the push button
		else:
			GPIO.output(DIR, 0)
			motor_step.start(dc)
			time.sleep(0.00000833333)  #1 full cycle: (1/240000 Hz)*2(for 
			motor_step.stop()
			stepCounter -= 1
			time.sleep(stepTime - .004)
		#calculate acutal angle
		angle = ((stepCounter) * (anglePerStep)) / (iplanet)

	
#This function lets the diving cell emerge
def emerge(angle,stepCounter)
	allLEDHigh()
	while (abs(angle - alpha_max) > 0.16):
		angle = step(alpha_max, angle, stepCounter)
		print("Emerge")
		print("angle =")
		print(angle)
		print("alpha =")
		print(alpha)
		loopTimeStart = time.time()
		allLEDLow()

################
#SMO

#sat function
def sat(x, gamma):
	double y = max(min(1.0, x / gamma), -1.0)
	return y

# Estimate ADA depth
def get_xhat1( x1,  n,  rho,  phi,  tau):
  xhat1 = xhat1_prev[n] - ((iterationtime) / (1000)) * rho * sat(xhat1_prev[n] - x1, phi);
  xhat1_prev[n] = xhat1
  return xhat1

  ???

# Esitmate derivative of x1
def get_xhat2(x1,n,rho,phi,tau):
  xhat1 = get_xhat1(x1, n, rho, phi, tau)
  xhat2 = xhat2_prev[n] + ((iterationtime) / ((1000) * tau)) * (-xhat2_prev[n] - rho * sat(xhat1 - x1, phi))
  xhat2_prev[n] = xhat2
  return xhat2
  
##########################
 
#SMC for diving cell
def slidingModeControler(z, zd,z_prev,angle,lambda, eta,phi,a):
	m_c = 0.3793 #mass in kg
	rho_w = 1000 #density of water in kg/m^3
	g = 9.81 #gravity in m/s^2
	k = 1 #viscosian damping parameter in kg/m
	#double V_0 = 0.000379 #DIVE volume in m^3
	A_dia = 0.000707 #membrane area in m^2
	h_max = 0.02042 #maximum stroke from  null position
	s = 0
	V_var = 0
	z_err[3] = 0
	#
	#calculating error values
	z_err[0] = z[0] - zd[0]
	z_err[1] = z[1] - zd[1]
	z_err[2] = z[2] - zd[2]
	z_err[3] = z[3] - zd[3]
	#
	#calculate actual varialbe volume
	V_act = A_dia * h_max * ((angle - (alpha_max / 2)) / alpha_max)
	#volume through controller
	#SMC 3rd order
	s = z_err[2] + 2 * lambda * z_err[1] + lambda * lambda * z_err[0]
	V_var = (1.0 / (a * rho_w * g)) * (a * rho_w * g * V_act - 2.0 * k * z[2] * abs(z[1]) - m_c * (zd[3] - 2.0 * lambda * z_err[2] - lambda * lambda * z_err[1]) + eta * sat(s, phi))
	#
	#robust Feedback SMC
	#s = z_err[2] + 2 * lambda * z_err[1] + lambda * lambda * z_err[0]
	#V_var = (1.0 / (a * rho_w * g)) * (a * rho_w * g * V_act - 2.0 * k * z[2] * abs(z[1]) - m_c * (zd[3] - 2.0 * lambda * z_err[2] -  pow(lambda, 2) * z_err[1]) + eta * sat(s, phi))
	#
	#3rd order SMC Integral
	#double z_integ = 0.5*(z_prev + z[0])*iterationtime
	#z_prev = z[0]
	#s=z_err[2]+3*lambda*z_err[1]+3*pow(lambda,2)*z_err[0]+ pow(lambda,3)*z_integ
	#V_var = (1.0 / (a * rho_w * g)) * (a * rho_w * g * V_act - 2.0 * k * z[2] * abs(z[1]) - m_c * (zd[3] - 3.0 * lambda * z_err[2] - 3.0 * pow(lambda, 2) * z_err[1] - pow(lambda, 3) * z_err[0]) + eta * sat(s, phi))
	#determine if control volume breaches max volume
	alpha_c = alpha_max / 2 + (V_var / (A_dia * h_max / 2)) * alpha_max / 2
	#ensure alpha is bounded
	if (alpha_c > alpha_max):
		alpha_c = alpha_max
	elif (alpha_c < 0):
		alpha_c = 0
	else:
		alpha_c = alpha_c
	return alpha_c
 

  
  #This function meassures the voltage of the LiPo battery.
#If voltage drops below 3.4V a warning is given.
#For meauring the voltage, Pin 14 is uses as batteryPin.

#Needs a analog to digital converter to measure voltage due to the lack of analog pins on the Nanopi
# def voltageBattery():
	# print("Battery voltage =")
	# print(vBat)
	# sensorValueBat = 0                      #measured value of analogous signal (0-1023)
	# vCorrectionFactor = 4.00 / 3.90         #correction factor (3.64V/3.57V)
	# vBatMeasured = 0                        #calculated voltage of measurement (R1=32.95kOhm, R2=33.07kOhm)
	# vBatSum = 0                             #Sum of measurments for determing average
	# Vcrit = 0
 
	# #store last values
	# vBat_old = vBat
	# #average voltage of batteryPin of 5 measurments
	# for i in range(5):
		# sensorValueBat =analogRead(batteryPin)
		# vBatMeasured = (3.3 / 1023) * sensorValueBat * 2
		# vBatSum = vBatSum + vBatMeasured
		# vBat = (vBatSum / 5)  * vCorrectionFactor

	# #if battery voltage of two consecutive measurements is below 3.4 stop program
	# if (vBat < 3.40 and vBat_old < 3.35):
		# print("Warning Battery Voltage To Tow")
		# #ESP8266DeepSleep()
		# LEDBlink(5, 200, 800)
    # #if diving cell is operation, emerge
    # if (response == 1):
		# emerge(angle, stepCounter)
    # LEDBlink(5, 200, 800)
    # Vcrit = 1
	# return Vcrit

#SETUP 

#GPIO SETUP
GPIO.setup(M0,GPIO.OUT)
GPIO.setup(M1,GPIO.OUT)
GPIO.setup(STEP,GPIO.OUT)
GPIO.setup(DIR,GPIO.OUT)
GPIO.setup(SLEEP,GPIO.OUT)
GPIO.setup(Switch_Pin,GPIO.IN)
  
motor_step = GPIO.PWM(STEP, freq)

GPIO.setup(ACTIVE_LED,GPIO.OUT, initial = 1)
time.sleep(.2)
GPIO.setup(SMC_LED,GPIO.OUT, initial = 1)
time.sleep(.2)
GPIO.setup(SWARM_CONNECTION_LED,GPIO.OUT, initial = 1)
time.sleep(.2)
GPIO.setup(DATA_LED,GPIO.OUT, initial = 1)
time.sleep(.2)
LEDBlink(2, 200, 400)	#function letting all LEDs blink 2 times

GPIO.setup(M0,GPIO.OUT)
GPIO.setup(M1,GPIO.OUT)

#check battery voltage
#voltageBattery()
#voltageBattery()

print(anglePerStep)

calibratePressure()


#determine step mode of the motor
if stepmode == 1:
	GPIO.output(M1, 0)
	GPIO.output(M0, 0)
elif stepmode == 2:
	GPIO.output(M1, 0)
	GPIO.output(M0, 1)
elif stepmode == 4:
	GPIO.output(M1, 0)
elif stepmode == 8:
	GPIO.output(M1, 1)
	GPIO.output(M0, 0)
elif stepmode == 16:
	GPIO.output(M1, 1)
	GPIO.output(M0, 1)
elif stepmode == 32:
	GPIO.output(M1, 1)


#set motor to neutral position at sprogram start
alpha = alpha_max / 2
GPIO.output(SLEEP, 1)
motorResetPosition(alpha, angle, stepCounter)
GPIO.output(SLEEP, 0)


SMC_lambda_set = SMC_lambda_default
SMC_eta_set = SMC_eta_default
SMC_phi_set = SMC_phi_d
SMC_a_set = SMC_a_default

SMC_timeMax_set = SMC_timeMax_default

zdesired_set = zdesired_default

SinAmp_set = SinAmp_default
SinPeriod_set = SinPeriod_default
StepTime_set = StepTime_default
StepAmp_set = StepAmp_default

SMO_rho_set = SMO_rho_default
SMO_phi_set = SMO_phi_default
SMO_tau_set = SMO_tau_default

SMC = 0
MotorReset = 0
MotorPosition = 0
closeCylinder = 0

while True:
	print("____________________________________ ")
	print("|                                  | ")
	print("| 1:  run SMC                      | ")
	print("| 2:  run SMC with sin wave        | ")
	print("| 3:  run SMC with step wave       | ")
	print("| 4:  update SMC parameters        | ")
	print("| 5:  update SMO parameters        | ")
	print("| 6:  update sine wave parameters  | ")
	print("| 7:  update step wave parameters  | ")
	print("| 8:  motor reset                  | ")
	print("| 9:  go to motor position         | ")
	print("| 0:  close cylinder               | ")
	print("| A:  Show All Parameters          | ")
	print("____________________________________ ")
	response = input("chose an option")
	if response == 1:
		SMC = 1
		break
	elif response == 2:
		SMC = 1
		SinWave = 1
		break
	elif response == 3:
		SMC = 1
		StepWave = 1
		break
	elif response == 4:
		SMC_lambda_set = input("What is the desired SMC Lambda value?")
		print("Lambda =")
		print(SMC_lambda_set)
		SMC_eta_set = input("What is the desired SMC Eta value?")
		print("Eta =")
		print(SMC_eta_set )
		SMC_phi_set = input("What is the desired SMC Phi value?")
		print("Phi =")
		print(SMC_phi_set)
		SMC_a_set = input("What is the desired SMC a value?")
		print("a =")
		print(SMC_a_set )
		SMC_timeMax_set = input("What is the desired Max Time in seconds?")
		print("Max Time =")
		print(SMC_timeMax_set )
	elif response == 5:
		SMO_rho_set = input("What is the desired SMO rho value?")
		print("Rho =")
		print(SMO_rho_set )
		SMO_phi_set  = input("What is the desired SMO phi value?")
		print("Phi =")
		print(SMO_phi_set )
		SMO_tau_set = input("What is the desired SMO tau value?")
		print("Tau =")
		print(SMO_tau_set )
	elif response == 6:
		SinAmp_set = input("What is the desired Sine Amplitude?")
		print("Sine Amplitude =")
		print(SinAmp_set)
		SinPeriod_set = input("What is the desired Sine Period?")
		print("Sine Period =")
		print(SinPeriod_set )
	elif response == 7:
		StepAmp_set= input("What is the desired Step Amplitude?")
		print("Step Amplitude =")
		print(StepAmp_set )
		StepTime_set = input("What is the desired Step Time?")
		print("Step Time =")
		print(StepTime_set )
	elif response == 8:
		MotorReset = 1
		break
	elif response == 9:
		MotorPosition = 1
		desiredPosition = input("what is desired position?")
		alphaDesired = desiredPosition
		break
	elif response == 0:
		closeCylinder = 1
		break
	elif response == "a" or "A":
		print("run time = ")
		print(SMC_timeMax_set)
		print("target depth = ")
		print(zdesired_set)
		print("Lambda = ")
		print(SMC_lambda_set)
		print("Eta = ")
		print(SMC_eta_set)
		print("Phi = ")
		print(SMC_phi_set)
		print("a = ")
		print(SMC_a_set)
		print("rho = ")
		print(SMO_rho_set)
		print("SMO_phi = ")
		print(SMO_phi_set)
		print("SMO_tau = ")
		print(SMO_tau_set)
		print("Sine Amplitude = ")
		print(SinAmp_set)
		print("Sine Period = ")
		print(SinPeriod_set)
		print("Step Amplitude = ")
		print(StepAmp_set)
		print("Step Time = ")
		print(StepTime_set)
	else
		print("That's not one of the options")
		print("Please choose one of the listed ones")

if (SMC == 1):
    batteryloop = 0                   #loop for batterycheck
    GPIO.output(SMC_LED, HIGH)
    #activate motor
    GPIO.output(SLEEP, HIGH)
    #
    #get running parameters
    iterationtime         = iterationtimeSC
    SMC_timeMax 		= SMC_timeMax_set
    zd[0]                 = zdesired_set
    zd[1]                 = 0
    zd[2]                 = 0
    zd[3]                 = 0
    zd0 = 0
    SMC_lambda     = SMC_lambda_set
    SMC_eta        = SMC_eta_set
    SMC_phi        = MC_phi_set
    SMC_a          = SMC_a_set
    SMO_rho        = SMO_rho_set
    SMO_phi        = SMO_phi_set
    SMO_tau        = SMO_tau_set
    SinWave       = 0
    SinAmp         = SinAmp_set
    SinPeriod      = SinPeriod_set
    Per				= 0
    StepWave      = 0
    StepTime       = StepTime_set
    StepAmp        = StepAmp_set
    stepWaveCounter = 0
    alpha = angle
   
    print("run time = ")
    print(SMC_timeMax)
    print("target depth = ")
    print(zd[0])
    print("Lambda = ")
    print(SMC_lambda)
    print("Eta = ")
    print(SMC_eta)
    print("Phi = ")
    print(SMC_phi)
    print("a = ")
    print(SMC_a)
    print("rho = ")
    print(SMO_rho)
    print("SMO_phi = ")
    print(SMO_phi)
    print("SMO_tau = ")
    print(SMO_tau)
	
    if (SinWave == 1):
		zd0 = zd[0]

		pi           = 3.14159265
		Per                 = 2*pi/SinPeriod
		print("SinAmp = ")
		print(SinAmp)
		print("SinPeriod = ")
		print(SinPeriod)
	
    if (StepWave == 1):
		zd0 = zd[0]
		print("Step Time = ")
		print(StepTime)
		print("Step Amplitude = ")
		print(StepAmp)
    #
  
    #countdown
    motorResetPosition(alpha_max / 2, angle, stepCounter)
    LEDBlink(3, 200, 800)
    GPIO.output(SMC_LED, HIGH)
     #
    #calibrate pressure sensor and calculate target pressure
    P_0 = calculatePressure()
    xhat1 = 0.0              #estimated depth in [m]
    xhat2 = 0.0                 #estimated velocity in [m/s]
    xhat1_prev[1] = 0.0  #estimated depth at previous time step in [m]
    xhat2_prev[1] = 0.0  #estimated velocity at previous time step in [m/s]
    xhat1_prev[2] = 0.0  #estimated depth at previous time step in [m]
    xhat2_prev[2] = 0.0  #estimated velocity at previous time step in [m/s]
    calibration = 0
	for(calibration <120):
		P = calculatePressure()
		z[0] = 10.0 * ((P - P_0) / 1000.0);
		z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau);
		z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau);
		delay(25)
		calibration += 1
    #
    #reset running parameters
    #timeStep = 0;
    #valuesToSave[0] = timeStep;
    #valuesToSave[1] = z[0];
    #valuesToSave[2] = z[1];
    #valuesToSave[3] = z[2];
    #valuesToSave[4] = angle;
    #valuesToSave[5] = alpha;
    #valuesToSave[6] = zd[0];
    #valuesToSave[7] = zd[1];
    #
    #store inital values
    #writeArrayBufferLine(valuesToSave, iArray, dataArray);
    #
    #program loop
    overallTime = 0;
    loopTimeStart = time.time()
	startTime = time.time()
	functionTimeStart = time.time()	#Start measuring time
    while (checkTime < (SMC_timeMax + 1) and P < 1400):
		# check battery        unusable until ADC 
		# if (batteryloop >= 400):
			# Vcrit = voltageBattery()
			# if (Vcrit == 1):
				# break;
			# batteryloop = 0
     
		P = calculatePressure()
		z[0] = 10.0 * ((P - P_0) / 1000.0)
		z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau)
		z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau)
		if(SinWave == 1):
			elapsed = time.time()
			timeElapsed = (elasped - startTime) *1000
			zd[0] =   zd0 + SinAmp*sin(Per*(timeElapsed))
			zd[1] =         SinAmp*cos(Per*(timeElapsed)) * Per *1000
			zd[2] =       - SinAmp*sin(Per*(timeElapsed)) * Per * 1000 * Per * 1000
			zd[3] =       - SinAmp*cos(Per*(timeElapsed)) * Per * 1000 * Per * 1000 * Per * 1000
			print("zd[0] = ")
			print(zd[0])
			print("zd[1] = ")
			print(zd[1])
			print("zd[2] = ")
			print(zd[2])
			
		if(StepWave == 1):
			functionTimeCheck = time.time()
			funtionTime = (functionTimeCheck - functionTimeStart) *1000
			if(functionTime/1000 < StepTime):
				zd[0] =   zd0 + StepAmp
			
			else:
				zd[0] = zd0 - StepAmp;
			
			if(functionTime/1000 >= 2*StepTime):
			   functionTime = time.time()
			
			print("zd[0] = ");
			print(zd[0]);
			
		alpha = slidingModeControler(z, zd, z_prev, angle, SMC_lambda, SMC_eta, SMC_phi, SMC_a)
		angle = step(alpha, angle, stepCounter)
		
		#store vaules every storeValueCounter timeStep
		#if (saveLoop == storeValueCounter) {
		#	valuesToSave[0] = timeStep;
			#valuesToSave[1] = P;
		#valuesToSave[0] = timeStep;
		#valuesToSave[1] = z[0];
		#valuesToSave[2] = z[1];
		#valuesToSave[3] = z[2];
		#valuesToSave[4] = angle;
		#valuesToSave[5] = alpha;
		#valuesToSave[6] = zd[0];
		#valuesToSave[7] = zd[1];
		#	writeArrayBufferLine(valuesToSave, iArray, dataArray);
		#	saveLoop = 0;
		#  }
		loopTimeStart = time.time()
		batteryloop += 1
		saveLoop+= 1
		timeStep+= 1
		check = time.time()
		checkTime = check - startTime
		#
    #end programm
    emerge(angle, stepCounter)
    digitalWrite(motorSleep_Pin, LOW)
  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  #
  //if selected, the motor is driven to neutral position
if (MotorReset == 1):
	GPIO.out(SLEEP, 1)
	motorResetPosition(alpha_max / 2, angle, stepCounter)
	print("Motor position reset" );
	GPIO.out(SLEEP, 0);
if (MotorPosition == 1) :
    GPIO.out(SLEEP,1)
    alpha = alphaDesired
    print("alpha read = ")
    print(alpha + )
    time.sleep(1)
    loopTimeStart = time.time()
    while (abs(angle - alpha) > 0.08):
		angle = step(alpha, angle, stepCounter)
		loopTimeStart = time.time()
    GPIO.output(SLEEP, LOW)

if (closeCylinder == 1):
	GPIO.output(SLEEP, HIGH);
	time.sleep(1);
		while (abs(angle - alpha_max) > 0.08):
			angle = step(alpha_max, angle, stepCounter)
			loopTimeStart = time.time()
		for (i in range (50*stepMode):
			GPIO.output(DIR, HIGH)
			motor_step.start(dc)
			time.sleep(0.00000833333)  #1 full cycle: (1/240000 Hz)*2(for 
			motor_step.stop()
			time.sleep(stepTime/1000)
			stepCounter+= 1
		GPIO.output(SLEEP, LOW)


#WRITE DESIRED RESULZTS TO FILE IF REQUESTED