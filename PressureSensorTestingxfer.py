import RPi.GPIO as GPIO
import time
import spidev

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
CSB_pin = 4
GPIO.setup(CSB_pin,GPIO.OUT,initial = 1)

spi=spidev.SpiDev()
CMD_RESET = 0x1E				#ADC reset command (ADC = Analgo-Digital-Converter)
CMD_ADC_READ=0x00               #ADC read command
CMD_ADC_CONV=0x40               #ADC conversion command
CMD_ADC_D1=0x00                 #ADC D1 conversion
CMD_ADC_D2=0x10                 #ADC D2 conversion
CMD_ADC_256=0x00                #ADC OSR=256 (OSR = Oversamplingrate)
CMD_ADC_512=0x02                #ADC OSR=512
CMD_ADC_1024=0x04               #ADC OSR=1024
CMD_ADC_2048=0x06               #ADC OSR=2056
CMD_ADC_4096=0x08               #ADC OSR=4096
CMD_PROM_RD=0xA0                #Prom read command
C = [0,0,0,0,0,0,0,0]			#calibration coefficients
P = 0.0							#compensated pressure value
T = 0.0							#compensated temperature value
MISO_pin = 21
MOSI_pin = 19
CLK_pin = 23
CS_pin = 24
spi.max_speed_hz = 20000000
spi.mode = 0b00

def cmd_reset()
	spi.cshigh = False
	spi.writebytes([CMD_RESET])
	time.sleep(.003)
	spi.cshigh = True
	
def cmd_prom(coef_num)
	ret = 0
	rC = 0
	spi.cshigh = False
	spi.xfer([CMD_PROM_RD + coef_num *2])
	ret = spi.xfer([0x00])
	rC = 256* ret
	ret = spi.xfer([0x00])
	rC = rC + ret
	spi.cshigh = True
	return rC
	
def PScalibrate()
	spi.open(0,0)
	cmd_reset()
	for i in range(0,7)
		C[i] = cmd_prom(i) 
	spi.close()
	return C
def cmd_adc(cmd,delaytime)
	ret = 0 
	temp = 0
	spi.cshigh = False
	spi.xfer([CMD_ADC_CONV + cmd])
	cmd = spi.xfer([0x00])
	time.sleep(delaytime*.001)
	spi.cshigh = True
	spi.cshigh = False
	spi.xfer([CMD_ADC_READ])
	ret = spi.xfer([0x00])
	temp = 65536*ret
	ret = spi.xfer([0x00])
	temp = temp+256 *ret
	ret = spi.xfer([0x00])
	temp = temp + ret
	spi.cshigh = True
	return temp

	
def calcPressure()
	D1 = 0
	D2 = 0
	dT = 0.0
	OFF = 0.0
	SENS = 0.0
	T2 = 0.0
	OFF2 = 0.0
	SENS2 = 0.0
	spi.open(0,0)
	D1 = cmd_adc(CMD_ADC_D1 + CMD_ADC_256,700)
	D2 = cmd_adc(CMD_ADC_D2 + CMD_ADC_256,700)
	dT = D2 - C[5] * pow(2,8)
	OFF = C[2] * pow(2,16) + dT*C[4] / pow(2,7)
	SENS = C[1] * pow(2,15) + dT*C[3]/ pow(2,8)
	T = (2000 + (dT*C[6])/pow(2,23))/100
	P = (((D1 * SENS)/pow(2,21)-OFF)/pow(2,15))/100
	if T>20:
		T2 = 0
		OFF2 = 0
		SENS2 = 0
		if T>45:
			SENS2 = SENS2 - pow(T-4500,2)/pow(2,3)
	else 
		T2 = pow(dT,2)/pow(2,3)
		OFF2 = 3*pow(100*T - 2000,2)
		SENS2 = 7*pow(100*T - 2000,2)/pow(2,3)
		if T<15:
			SENS2 = SENS2 + 2*pow(100*T+1500,2)
	T = T-T2
	OFF = OFF - OFF2
	SENS = SENS - SENS2
	P = (((D1*SENS)/pow(2,21)-OFF)/pow(2,15))/100
	spi.close()
	return P
	
PScalibrate()
calcPressure()
print P
