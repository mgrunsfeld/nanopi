#include <Eigen324.h>                         // Calls main Eigen matrix class library
#include <Eigen/Sparse>
#include <vector>
using namespace Eigen;
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <EEPROM.h>
#include <SPI.h>
#include<string.h>
//
//Defining min and max and abs operations
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define abs(a) ((a)<(0)? -(a):(a))
//
//Program parameters
double z[4];                                  //dynamics vector [z, zdot, zddot, zdddot] in [m m/s m/s2 m/s3]
double zd[4];                                 //vector of desired dynamics [z, zdot, zddot, zdddot] in [m m/s m/s2 m/s3]
double P_target;                              //target pressure in [hPa]
double P_0;                                   //initial pressure in [hPa]
float programParameter[25];                   //array for program parameters
//
//time parameters
uint32_t timeStep = 0;                        //counter for ellapsed time steps
elapsedMillis looptime = 0;                   //time for a loop in [ms]
elapsedMillis overallTime = 0;                //overall time in [ms]
elapsedMillis functionTime = 0;
uint16_t iterationtime = 25;                  //time per loop [ms]
uint16_t iterationtimeRL = 100;               //loop time under RL control
uint16_t iterationtimeSC = 25;                //loop time under SC control
uint64_t maxOverallTime;                      //max program time
//
/////parameters of RL algorithm
//setting parameters
const uint16_t failuresExploStop = 400;       //number of failures before exploration moves stop
const uint8_t terminalState = 110;            //number of states for discretization
const float exploProb0 = 0.1;                 //probability of exploration moves in percent
const float discount = 0.99;                  //discounting factor of value iteration (influences foresight of the algorithm)
const float tolerance = 0.0001;               //convergence border for value iteration
const float posTol = 0.15;                    //learning region in [m]
const float velTol = 0.03;                    //velocity region in [m/2]
//running parameters
uint8_t state = 110;                          //state variable, maximum of 255 states using uint8_t
uint8_t newState = 0;                         //new state to determine transition probabilites
uint8_t action = 1;                           //action for DICE
float randNum = 0;                            //random number for determine next action and exploration
uint16_t numFailures = 0;                     //maximum of 65535 failures before overflow
double alpha_new;                             //new alpha variable
uint8_t insideBorders = 0;                    //determines if RL is active (1) or not (0)
uint32_t timeStepsTrial = 0;                  //time steps of current trial
uint32_t timeStepsTotal = 0;                  //time steps in total
uint16_t stabilizationCounter = 0;            //counter for stabilization 
uint8_t randAction = 0;                       //factor for storing random actions
//VectorXf z(4);                              //[z, z_dot, zddot, z_dddot]
//transition and reward
SparseMatrix<int> transitionCounts1(terminalState + 1, terminalState + 1);  //counts transition from state to newState under action1
SparseMatrix<int> transitionCounts2(terminalState + 1, terminalState + 1);  //counts transition from state to newState under action2
SparseMatrix<float> transitionProbs1(terminalState + 1, terminalState + 1); //probability of transition from state to newState under action1
SparseMatrix<float> transitionProbs2(terminalState + 1, terminalState + 1); //probability of transition from state to newState under action2
SparseMatrix<float> reward(terminalState + 1, 2);                           //reward vector
SparseMatrix<int> rewardCounts(terminalState + 1, 2);                       //vector with number of rewards in state s
SparseMatrix<float> rewardAmounts(terminalState + 1, 2);                    //vector with amount of rewards in state s
VectorXf value(terminalState + 1);                                          //value of state s under action taken
VectorXf rewardCol1(2);                                                     //vector to calculate the first column of the reward Matrix
VectorXf rewardCol2(2);                                                     //vector to calculate the first column of the reward Matrix
VectorXi timeStepsBeforeFailure;                                            //vector to count number of time steps before algorithm terminates
//
//parameters of SC control
double SMC_Lambda_default = 0.40;
double SMC_eta_default = 0.10;
double SMC_phi_default = 0.50;
double SMC_a_default = 20.0;
double z_prev = 0;              //previous z for 3rd order SC and integral
//
//parameters of iSC control
double iSMC_gam_default = 1;
double iSMC_neu_default = 0;
//
//LED pin declaration
uint8_t ACTIVE_LED = 2;         //LED showing activity of DICE
uint8_t RL_LED = 3;             //LED showing usage of RL control
uint8_t SMC_LED = 4;            //LED showing usage of SC control
uint8_t DATA_LED = 5;           //LED showing access of memory
//
//ESP8266
#include <SoftwareSerial.h>               //including software serial library
SoftwareSerial ESP8266(9, 10);            //softwareserial for the ESP8266
const int ESP8266Reset_Pin = 17;          //reset pin on Teensy for waking module
#define DEBUG true
//
//M24M02 EEPROM
#include <i2c_t3.h>
#define M24M02DRC_1_DATA_ADDRESS 0x50     //address of the first 1024 page M24M02DRC EEPROM data buffer, 2048 bits (256 8-bit bytes) per page
#define M24M02DRC_1_IDPAGE_ADDRESS 0x58   //address of the single M24M02DRC lockable ID page of the first EEPROM
#define M24M02DRC_2_DATA_ADDRESS 0x54     //address of the second 1024 page M24M02DRC EEPROM data buffer, 2048 bits (256 8-bit bytes) per page
#define M24M02DRC_2_IDPAGE_ADDRESS 0x5C   //address of the single M24M02DRC lockable ID page of the second EEPROM
#define EEPROMVIN 7                       //VIN of EEPROM in case of DICE Pin 7
#define EEPROMGND 6                       //GND of EEPROM in case of DICE Pin 6
const uint16_t storeValueCounter = 4;     //store values every storeValueCounter time steps
const uint8_t valuesPerRow = 8;           //number of values to save per time-step
const uint8_t rowPageMax = 7;             //number of rows per EEPROM page ((64/valuesPerRow)-1)
float valuesToSave[valuesPerRow];         //array of values to save
long dataArray[rowPageMax][valuesPerRow]; //array of data to write to EEPROM page
//storage parameters fot the first EEPROM chip (used for measurments during operation)
uint8_t device_quarter = 0;               //quarter of EEPROM-chip 1
uint16_t data_address1 = 0;               //page of data
uint8_t data_address2 = 0;                //data address
uint8_t iArray = 0;                       //row variable
int saveLoop = 0;                         //loop variable
//storage parameters fot the first EEPROM chip (used for saving the RL related matrices)
uint8_t device_quarter_C2 = 0;            // Quarter of EEPROM-chip 2
uint16_t data_address1_C2 = 0;            // Page of data
uint8_t data_address2_C2 = 0;             // Data address
//
//DRUCKSENSOR
// Declaration of Copyright
// Copyright (c) for pressure sensor 2009 MEAS Switzerland
// Edited 2015 Johann Lange and 2016 Gerrit Brinkmann
// @brief This C code is for starter reference only. It was based on the
// MEAS Switzerland MS56xx pressure sensor modules and Atmel Atmega644p
// microcontroller code and has been by translated Johann Lange
// to work with Teensy 3.1 microcontroller.
//Macros
#define TRUE 1
#define FALSE 0
#define CMD_RESET 0x1E                  //ADC reset command (ADC = Analgo-Digital-Converter)
#define CMD_ADC_READ 0x00               //ADC read command
#define CMD_ADC_CONV 0x40               //ADC conversion command
#define CMD_ADC_D1 0x00                 //ADC D1 conversion
#define CMD_ADC_D2 0x10                 //ADC D2 conversion
#define CMD_ADC_256 0x00                //ADC OSR=256 (OSR = Oversamplingrate)
#define CMD_ADC_512 0x02                //ADC OSR=512
#define CMD_ADC_1024 0x04               //ADC OSR=1024
#define CMD_ADC_2048 0x06               //ADC OSR=2056
#define CMD_ADC_4096 0x08               //ADC OSR=4096
#define CMD_PROM_RD 0xA0                //Prom read command
// declarations for pressure sensor
unsigned int C[8];                      //calibration coefficients
double P;                               //compensated pressure value
double T;                               //compensated temperature value
// SPI connection for pressure sensor
const int MISO_Pin  = 12;
const int MOSI_Pin  = 11;
const int SCK_Pin   = 13;
const int CS_Pin_PS = 20;                                 //Chip Select Pressure Sensor
SPISettings settings_PS(4000000, MSBFIRST, SPI_MODE0);    //Grundeinstellungen fuer das SPI-Protokoll, Pressure Sensor
//
//SLIDING-MODE-OBSERVER (SMO)
//Declarations
///Model parameters
double SMO_rho_default = 2.5;       //observer parameter
double SMO_tau_default = 0.2;       //observe rparameter
double SMO_phi_default = 0.5;       //observe rparameter
///coordinates
double xhat1 = 0.0;                 //estimated depth in [m]
double xhat2 = 0.0;                 //estimated velocity in [m/s]
double xhat1_prev[2] = {0.0, 0.0};  //estimated depth at previous time step in [m]
double xhat2_prev[2] = {0.0, 0.0};  //estimated velocity at previous time step in [m/s]
//
//MOTOR
///Pins
const int Switch_Pin      = 15;     //push botton pin for motor reset
const int motorDir_Pin    = 21;     //motor direction pin
const int motorStep_Pin   = 22;     //motor step pin
const int motorSleep_Pin  = 23;     //motor sleep pin
const int M0              = 0;      //pin determining step modus
const int M1              = 1;      //pin determining step modus
//Variables
const int     stepMode      = 32;                         //step modes 1=full step; 2=half step, 4=1/4 step, 8 =1/8 step ;
const double  stepFrequency = 500 * stepMode;             //max step frequency allowed [Hz] (full step=500, half step 1000, 1/4 step = 2000)
const int     iplanet       = 256;                        //transmission of the gear
const double  anglePerStep  = ((double)18 /(double)stepMode);            //rotation angle of the motor per step [deg], depends on the step mode
double        alpha;                                      //desired angle of the motor [deg]
const double  alpha_max     = 130;                        //max angle of the motor [deg]
double        angle;                                      //acutal angle of the motor [deg]. Start in fully extended position
double        stepTime      = 1000000 / stepFrequency;    //time the motor has for a step in [µs]
int32_t       stepCounter   = 0;                          //counter of steps starting for angle 0 [deg]
const int     Switch_Steps  = 20 * stepMode;              //number of steps from position reset to neutral position
//
//Battery voltage control
const int batteryPin = 14;    //pin for measuring the battery voltage
double vBat;                  //voltage of LiPo in [V]
double vBat_old;              //voltage of LiPo in [V]
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Teensy setup function
void setup() {
  //
  //configuring outputs for LEDs
  pinMode(ACTIVE_LED, OUTPUT);
  digitalWrite(ACTIVE_LED, HIGH);
  delay(200);
  pinMode(RL_LED, OUTPUT);
  digitalWrite(RL_LED, HIGH);
  delay(200);
  pinMode(SMC_LED, OUTPUT);
  digitalWrite(SMC_LED, HIGH);
  delay(200);
  pinMode(DATA_LED, OUTPUT);
  digitalWrite(DATA_LED, HIGH);
  delay(200);
  digitalWrite(ACTIVE_LED, HIGH);
  LEDBlink(2, 200, 400); //function letting all LEDs blink 2 times 
  //
  //check battery voltage
  voltageBattery();
  voltageBattery();
  //
  //establish serial connection
  Serial.begin(115200);
  Serial.print(anglePerStep);
  //
  //seed random number generator for RL using unused pin 16
  srand(analogRead(16));
  //
  //ESP8266
  pinMode(ESP8266Reset_Pin, OUTPUT);
  digitalWriteFast(ESP8266Reset_Pin, HIGH);
  delay(1000);
  ESP8266.begin(115200);    //establishing communication to ESP module
  ESPserverConfiguration(); //configuring ESP module
  //
  //M24M20 EEPROM
  Wire.begin(I2C_MASTER, 0x00, I2C_PINS_18_19, I2C_PULLUP_EXT, I2C_RATE_400);
  pinMode(EEPROMVIN, OUTPUT);           //can use traditional 3V3 and GND from the Teensy 3.1 or Arduino Pro Mini
  digitalWrite(EEPROMVIN, HIGH);        //or use GPIO pins for power and ground since tha max current drawn is 2 mA!
  pinMode(EEPROMGND, OUTPUT);
  digitalWrite(EEPROMGND, LOW);
  //
  //Teensy LED
  pinMode(13, OUTPUT);
  //
  //Pressure sensor
  pinMode(CS_Pin_PS, OUTPUT);
  pinMode(MISO_Pin, OUTPUT);
  pinMode(SCK_Pin, OUTPUT);
  pinMode(MOSI_Pin, INPUT);
  SPI.setMOSI(MOSI_Pin);
  SPI.setMISO(MISO_Pin);
  SPI.setSCK(SCK_Pin);
  SPI.begin();
  PressureSensorCalibration();
  //
  //Motor
  pinMode(Switch_Pin, INPUT);
  pinMode(motorDir_Pin, OUTPUT);
  pinMode(motorStep_Pin, OUTPUT);
  pinMode(motorSleep_Pin, OUTPUT);
  digitalWriteFast(motorDir_Pin, LOW);
  digitalWriteFast(motorStep_Pin, LOW);
  digitalWriteFast(motorSleep_Pin, HIGH);
  //determine step mode of the motor
  if (stepMode == 1) {    //full step
    pinMode(M0, OUTPUT);
    pinMode(M1, OUTPUT);
    digitalWrite(M0, LOW);
    digitalWrite(M1, LOW);
  }
  if (stepMode == 2) {    //half step
    pinMode(M0, OUTPUT);
    pinMode(M1, OUTPUT);
    digitalWrite(M0, HIGH);
    digitalWrite(M1, LOW);
  }
  if (stepMode == 4) {    //1/4 step
    pinMode(M1, OUTPUT);
    digitalWrite(M1, LOW);
  }
  if (stepMode == 8) {    //1/8 step
    pinMode(M0, OUTPUT);
    pinMode(M1, OUTPUT);
    digitalWrite(M0, LOW);
    digitalWrite(M1, HIGH);
  }
  if (stepMode == 16) {    //1/16 step
    pinMode(M0, OUTPUT);
    pinMode(M1, OUTPUT);
    digitalWrite(M0, HIGH);
    digitalWrite(M1, HIGH);
  }
  if (stepMode == 32) {    //1/32 step
    pinMode(M1, OUTPUT);
    digitalWrite(M1, HIGH);
  }
  //reset motor to neutral position at sprogram start
  alpha = alpha_max / 2;
  motorResetPosition(alpha, angle, stepCounter);
  digitalWrite(motorSleep_Pin, LOW);
  //
  //RL matrix definition col1 and col2 for calculating rows 1 and 2 in RL algorithm
  rewardCol1[0] = 1;
  rewardCol1[1] = 0;
  rewardCol2[0] = 0;
  rewardCol2[1] = 1;

}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Teensy loop function
void loop() {
  //This part is used to determine which program is chosen via the webpage
  ESP8266programStart(programParameter);              //determine running parameters sent from website
  uint8_t   SMC             = programParameter[0];
  uint8_t   readEEPROM      = programParameter[1];
  uint8_t   MotorReset      = programParameter[2];
  uint8_t   MotorPosition   = programParameter[3];
  uint8_t   closeCylinder   = programParameter[4];
  uint8_t   RL              = programParameter[5];
  uint8_t   RLstoreMatrices = programParameter[6];
  uint8_t   iSMC            = programParameter[7];
  uint8_t   SinWave         = programParameter[8];
  uint8_t   StepWave        = programParameter[9];
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////RL
  if (RL == 1) {
    //
    //sent ESP to sleep
    ESP8266DeepSleep();
    //
    iterationtime                   = iterationtimeRL;  //setting loop period of RL control as defined above
    //
    //get running parameters from website
    uint32_t  numFailuresMax        = programParameter[10] / 1000;
    zd[0]                           = (double)programParameter[11];
    zd[1]                           = 0;  
    zd[2]                           = 0;
    double zd_SMC                   = zd[0];
    double SMC_lambda               = (double)programParameter[12];
    double SMC_eta                  = (double)programParameter[13];
    double SMC_phi                  = (double)programParameter[14];
    double SMC_a                    = SMC_a_default;
    double SMO_rho                  = (double)programParameter[17];
    double SMO_phi                  = (double)programParameter[18];
    double SMO_tau                  = (double)programParameter[19];
    uint8_t useExistingProperties   = programParameter[15];
    uint16_t numFailuresExplo       = programParameter[16];
    //
    Serial.print("max Failures = ");
    Serial.println(numFailuresMax);
    Serial.print("Failures until exploration stop = ");
    Serial.println(numFailuresExplo);
    Serial.print("target depth = ");
    Serial.println(zd[0]);
    Serial.print("Lambda = ");
    Serial.println(SMC_lambda);
    Serial.print("Eta = ");
    Serial.println(SMC_eta);
    Serial.print("Phi = ");
    Serial.println(SMC_phi);
    //
    double alpha_avg    = alpha_max/2;              //average alpha set to neutral postition
    double alpha_RLmin  = 0.15 * alpha_max;         //limiting maximum of alpha
    double alpha_RLmax  = 0.15 * alpha_max;         //limiting minimum of alpha
    double dalpha       = 0.021*iterationtimeRL;    //change rate of alpha per action selection
    //
    timeStep = 0;               //reset time steps     
    timeStepsTrial = 0;         //time steps of current trial
    timeStepsTotal = 0;         //time steps in total
    uint16_t batteryloop = 0;   //loop for batterycheck
    //
    //clear all matrices and vectors
    transitionCounts1.setZero();
    transitionCounts2.setZero();
    transitionProbs1.setZero();
    transitionProbs2.setZero();
    rewardCounts.setZero();
    rewardAmounts.setZero();
    value.setZero();
    timeStepsBeforeFailure.setZero();
    numFailures = 0;
    //
    //if previouse parameters are used read matrices from EEPROM
    if (useExistingProperties == 1) {
      readRLMatrizesFromEEPROM(transitionCounts1, transitionCounts2, value, rewardAmounts, rewardCounts, numFailures);
      //
      //calculate transitionProbs1 and transitionProbs2 through the value iteration function
      valueIteration(transitionCounts1, transitionCounts2, transitionProbs1, transitionProbs2, rewardCounts, rewardAmounts, reward, value);
      //Serial.println();
      Serial.println("transitionProbs1");
      rSMatF(transitionProbs1);
      Serial.println("transitionProbs2");
      rSMatF(transitionProbs2);
      Serial.println("Reading successfull");
      delay(1000);
    }
    //
    //determining presure of the environment
    P_0 = calculatePressure();
    delay(100);
    P = calculatePressure();
    //determine actual depth, velocity and acceleration
    z[0] = 10.0 * ((P - P_0) / 1000.0);
    z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau);
    z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau);
    //activate motor
    digitalWrite(motorSleep_Pin, HIGH);
    //countdown
    LEDBlink(3, 200, 800);
    //
    //reset time parameters
    overallTime = 0;
    looptime = 0;
    //control loop
    while (numFailures < numFailuresMax && P < 1200) {
      //
      //check battery
      if (batteryloop >= 400) {
        uint8_t Vcrit = voltageBattery();
        if (Vcrit == 1) {
          break;
        }
        batteryloop = 0;
      }
      //
      //if diving cell is inside boundaries: RL algorithm
      if (insideBorders == 1) {
        iterationtime = iterationtimeRL;
        //define goal of 300s for terminating algorithm
        if (timeStepsTrial * iterationtime > 300000) {
          Serial.println("Learning Success");
          break;
        }
        digitalWriteFast(RL_LED, HIGH);
        digitalWriteFast(ACTIVE_LED, LOW);
        //select next action
        action = selectNextAction(randAction, transitionProbs1, transitionProbs2, reward, value, numFailures, numFailuresExplo);
        //determining alpha
        if (action == 1) {
          if (angle >= ((alpha_avg - alpha_RLmin) + dalpha)) {
            alpha_new = angle - dalpha;
          }
          else {
            alpha_new = (alpha_avg - alpha_RLmin);
          }
        }
        else if (action == 2) {
          if (angle <= ((alpha_avg+alpha_RLmax) - dalpha)) {
            alpha_new = angle + dalpha;
          }
          else {
            alpha_new = (alpha_avg + alpha_RLmax);
          }
        }
        alpha = alpha_new;
        //
        //drive motor angle to alpha
        angle = step(alpha, angle, stepCounter);
        //count total time steps of programm
        timeStepsTrial = timeStepsTrial + 1;                         
        //calculate new state
        P = calculatePressure();
        z[0] = 10.0 * ((P - P_0) / 1000.0);
        z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau);
        z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau);
        newState = get_state(z);
        //
        //update state transition matrix
        rewardAndTransitionUpdate(z, zd, newState, action, transitionCounts1, transitionCounts2, rewardCounts, rewardAmounts);
        //if boundaies are left update value function
        if (newState == terminalState) {
            valueIteration(transitionCounts1, transitionCounts2, transitionProbs1, transitionProbs2, rewardCounts, rewardAmounts, reward, value);
            updateRunningParameter(timeStepsTrial, numFailures, timeStepsBeforeFailure);
          insideBorders = 0;
          //alternate starting position of algorithm
          zd_SMC = zd[0] - posTol / 2 + double(rand() % int(posTol * 10000)) / 10000;
          alpha_avg = 0;
        }
        state = newState;                                            //change newState to state for next iteration loop
        timeStep = timeStep + 1;
        saveLoop = saveLoop + 1;
        batteryloop = batteryloop + 1;
      }
      //
      //if boundaries are left: SMC controller
      else {
        iterationtime = iterationtimeSC;
        digitalWriteFast(RL_LED, LOW);
        digitalWriteFast(ACTIVE_LED, HIGH);
        P = calculatePressure();
        z[0] = 10.0 * ((P - P_0) / 1000.0);
        z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau);
        z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau);      
        double zd_RL = zd[0];
        zd_SMC = 0;
        zd[0] = zd_SMC;
        alpha = slidingModeControler(z, zd, z_prev, angle, SMC_lambda, SMC_eta, SMC_phi, SMC_a);
        zd[0] = zd_RL;
        angle = step(alpha, angle, stepCounter);
        //timeStep = timeStep + 1;
        //saveLoop++;
        batteryloop++;
        //
        if (fabs(z[0] - zd_SMC) < 0.01 && z[1] < 0.01) {
          stabilizationCounter ++;
          if (stabilizationCounter > 50) {
          alpha_avg = ((stabilizationCounter-51) * alpha_avg + alpha)/(stabilizationCounter-50);
          }
          //if stabilized for 10s, switch to RL algorithm
          if (stabilizationCounter > 250) {
            insideBorders = 1;
            newState = get_state(z);                                                                                                                                                                      
            timeStep = timeStep + 80;
          }
          if (stabilizationCounter > 4000) {
            break;
          }
        }
        else {
          stabilizationCounter = 0;
        }
      }
      //save values
      state = newState;
      if (saveLoop >= storeValueCounter) {
        valuesToSave[0] = timeStep;
        valuesToSave[1] = z[0];
        valuesToSave[2] = z[1];
        valuesToSave[3] = state;
        valuesToSave[4] = angle;
        valuesToSave[5] = action;
        valuesToSave[6] = randAction;
        valuesToSave[7] = numFailures;
        writeArrayBufferLine(valuesToSave, iArray, dataArray);
        saveLoop = 0;
      }
      looptime = 0;
    }

    emerge(angle, stepCounter);
    writeLastPageToEEPROM(dataArray, device_quarter, data_address1);
    digitalWrite(motorSleep_Pin, LOW);
    ESP8266WakeUp();
  }
  //
  //if selected, SMC program is run
  if (SMC == 1) {
    uint16_t batteryloop = 0;                   //loop for batterycheck
    digitalWrite(SMC_LED, HIGH);
    //
    //sent ESP to sleep
    ESP8266DeepSleep();
    //
    //activate motor
    digitalWrite(motorSleep_Pin, HIGH);
    //
    //get running parameters
    iterationtime         = iterationtimeSC;
    uint32_t SMC_timeMax  =  programParameter[10];
    zd[0]                 = (double)programParameter[11];
    zd[1]                 = 0;
    zd[2]                 = 0;
    zd[3]                 = 0;
    double zd0;
    double SMC_lambda     = (double)programParameter[12];
    double SMC_eta        = (double)programParameter[13];
    double SMC_phi        = (double)programParameter[14];
    double SMC_a          = 20.0;
    double SMO_rho        = (double)programParameter[17];
    double SMO_phi        = (double)programParameter[18];
    double SMO_tau        = (double)programParameter[19];
    uint8_t SinWave       = programParameter[8];
    double SinAmp         = 0;
    double SinPeriod      = 0;
    double Per;
    uint8_t StepWave      = programParameter[9];
    double StepTime       = 0;
    double StepAmp        = 0;
    uint16_t stepWaveCounter = 0;
    alpha = angle;
   
    Serial.print("run time = ");
    Serial.println(SMC_timeMax);
    Serial.print("target depth = ");
    Serial.println(zd[0]);
    Serial.print("Lambda = ");
    Serial.println(SMC_lambda);
    Serial.print("Eta = ");
    Serial.println(SMC_eta);
    Serial.print("Phi = ");
    Serial.println(SMC_phi);
    Serial.print("a = ");
    Serial.println(SMC_a);
    Serial.print("rho = ");
    Serial.println(SMO_rho);
    Serial.print("SMO_phi = ");
    Serial.println(SMO_phi);
    Serial.print("SMO_tau = ");
    Serial.println(SMO_tau);
    Serial.print("a = ");
    Serial.println(SMC_a);
    if (SinWave == 1){
    zd0 = zd[0];
    SinAmp              = programParameter[15];
    SinPeriod           = programParameter[16]*1000;
    double pi           = 3.14159265;
    Per                 = 2*pi/SinPeriod;
    Serial.print("SinAmp = ");
    Serial.println(SinAmp);
    Serial.print("SinPeriod = ");
    Serial.println(SinPeriod);
  }
   if (StepWave == 1){
    zd0 = zd[0];
    StepTime     = programParameter[15];
    StepAmp      = programParameter[16];
    Serial.print("Step Time = ");
    Serial.println(StepTime);
    Serial.print("Step Amplitude = ");
    Serial.println(StepAmp);
  }
    //
    Serial.println("Parameters taken: ");
  for(uint8_t u=0; u<25;u++){
  Serial.println(programParameter[u]);
  }
    //countdown
    motorResetPosition(alpha_max / 2, angle, stepCounter);
    LEDBlink(3, 200, 800);
    digitalWrite(SMC_LED, HIGH);
     //
    //calibrate pressure sensor and calculate target pressure
    P_0 = calculatePressure();
    xhat1 = 0.0;                 //estimated depth in [m]
    xhat2 = 0.0;                 //estimated velocity in [m/s]
    xhat1_prev[1] = 0.0;  //estimated depth at previous time step in [m]
    xhat2_prev[1] = 0.0;  //estimated velocity at previous time step in [m/s]
    xhat1_prev[2] = 0.0;  //estimated depth at previous time step in [m]
    xhat2_prev[2] = 0.0;  //estimated velocity at previous time step in [m/s]
    for(uint16_t calibration=0; calibration <120; calibration++){
    P = calculatePressure(); 
    z[0] = 10.0 * ((P - P_0) / 1000.0);
    z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau);
    z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau);
    delay(25);
    }
    //
    //reset running parameters
    timeStep = 0;
    valuesToSave[0] = timeStep;
    valuesToSave[1] = z[0];
    valuesToSave[2] = z[1];
    valuesToSave[3] = z[2];
    valuesToSave[4] = angle;
    valuesToSave[5] = alpha;
    valuesToSave[6] = zd[0];
    valuesToSave[7] = zd[1];
    //
    //store inital values
    writeArrayBufferLine(valuesToSave, iArray, dataArray);
    //
    //program loop
    overallTime = 0;
    looptime = 0;
    functionTime = 0;
    while (overallTime < (SMC_timeMax + 1) && P < 1400) {
      // check battery
      if (batteryloop >= 400) {
        uint8_t Vcrit = voltageBattery();
        if (Vcrit == 1) {
          break;
        }
        batteryloop = 0;
      }
      P = calculatePressure();
      z[0] = 10.0 * ((P - P_0) / 1000.0);
      z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau);
      z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau);
      if(SinWave == 1){
      zd[0] =   zd0 + SinAmp*sin(Per*(overallTime));
      zd[1] =         SinAmp*cos(Per*(overallTime)) * Per *1000;
      zd[2] =       - SinAmp*sin(Per*(overallTime)) * Per * 1000 * Per * 1000;
      zd[3] =       - SinAmp*cos(Per*(overallTime)) * Per * 1000 * Per * 1000 * Per * 1000;
      Serial.print("zd[0] = ");
      Serial.println(zd[0],4);
      Serial.print("zd[1] = ");
      Serial.println(zd[1],4);
      Serial.print("zd[2] = ");
      Serial.println(zd[2],4);
      }      
      if(StepWave == 1){
      if(functionTime/1000 < StepTime){
      zd[0] =   zd0 + StepAmp;
      }
      else{
      zd[0] = zd0 - StepAmp;
      }
      if(functionTime/1000 >= 2*StepTime){
         functionTime = 0;
      }
      Serial.print("zd[0] = ");
      Serial.println(zd[0]);
      }      
      alpha = slidingModeControler(z, zd, z_prev, angle, SMC_lambda, SMC_eta, SMC_phi, SMC_a);
      angle = step(alpha, angle, stepCounter);
      //
      //store vaules every storeValueCounter timeStep
      if (saveLoop == storeValueCounter) {
        valuesToSave[0] = timeStep;
        //valuesToSave[1] = P;
    valuesToSave[0] = timeStep;
    valuesToSave[1] = z[0];
    valuesToSave[2] = z[1];
    valuesToSave[3] = z[2];
    valuesToSave[4] = angle;
    valuesToSave[5] = alpha;
    valuesToSave[6] = zd[0];
    valuesToSave[7] = zd[1];
        writeArrayBufferLine(valuesToSave, iArray, dataArray);
        saveLoop = 0;
      }
      looptime = 0;
      batteryloop++;
      saveLoop++;
      timeStep++;
    }
    //
    //end programm
    emerge(angle, stepCounter);
    writeLastPageToEEPROM(dataArray, device_quarter, data_address1);
    digitalWrite(motorSleep_Pin, LOW);
    ESP8266WakeUp();
  }
 //
  //if selected, SMC program is run
  if (iSMC == 1) {
    Serial.println("intelligent SMC is chosen");
    uint16_t batteryloop = 0;                   //loop for batterycheck
    digitalWrite(SMC_LED, HIGH);
    //
    //sent ESP to sleep
    ESP8266DeepSleep();
    //
    //activate motor
    digitalWrite(motorSleep_Pin, HIGH);
    //
    //get running parameters
    iterationtime         = iterationtimeSC;
    uint32_t iSMC_timeMax =  programParameter[10];
    zd[0]                 = (double)programParameter[11];
    zd[1]                 = 0;
    zd[2]                 = 0;
    zd[3]                 = 0;
    double zd0;
    double iSMC_lambda    = (double)programParameter[12];
    double iSMC_eta       = (double)programParameter[13];
    double iSMC_phi       = (double)programParameter[14];
    double iSMC_a         = 20.0;
    double iSMC_gamma     = (double)programParameter[20];
    double iSMC_neuron    = (uint8_t)programParameter[21];
    double d_hat          = 0;
    double SMO_rho        = (double)programParameter[17];
    double SMO_phi        = (double)programParameter[18];
    double SMO_tau        = (double)programParameter[19];
    uint8_t SinWave       = programParameter[8];
    double SinAmp         = 0;
    double SinPeriod      = 0;
    double Per;
    uint8_t StepWave      = programParameter[9];
    double StepTime       = 0;
    double StepAmp        = 0;
    uint16_t stepWaveCounter = 0;
    alpha = angle; 
    //////////inform user about parameter  
    Serial.print("run time = ");
    Serial.println(iSMC_timeMax);
    Serial.print("target depth = ");
    Serial.println(zd[0]);
    Serial.print("Lambda = ");
    Serial.println(iSMC_lambda);
    Serial.print("Eta = ");
    Serial.println(iSMC_eta);
    Serial.print("Phi = ");
    Serial.println(iSMC_phi);
    Serial.print("gamma = ");
    Serial.println(iSMC_gamma);
    Serial.print("neuron = ");
    Serial.println(iSMC_neuron);
    Serial.print("a = ");
    Serial.println(iSMC_a);
    Serial.print("rho = ");
    Serial.println(SMO_rho);
    Serial.print("SMO_phi = ");
    Serial.println(SMO_phi);
    Serial.print("SMO_tau = ");
    Serial.println(SMO_tau);
    //////////implementation of weights and neurons
    //weights
    double Vw[7];
    for(uint8_t i=0; i<7; i++){
      Vw[i] = 0;
    }
    //neurons
    //parameters for the Triangular type neuron: Vp[N+2] = {Left limit, 1st center, ... , Nth center, Right limit}
    double Vp[14] = {iSMC_phi, -3.0*iSMC_phi/4.0, -iSMC_phi/2.0, -iSMC_phi/4.0, 0.0, iSMC_phi/4.0, iSMC_phi/2.0, 3.0*iSMC_phi/4.0, iSMC_phi, 0 , 0, 0, 0 ,0};
    //parameters for the Gaussian type neuron: Vp[2*N] = {1st center, ... , Nth center, 1st width, ... , Nth width}
    if(iSMC_neuron == 1){
     Serial.println("neuron 1 is chosen");
    //Vp[14] = {-3.0*iSMC_phi/4.0, -iSMC_phi/2.0, -iSMC_phi/4.0, 0.0, iSMC_phi/4.0, iSMC_phi/2.0, 3.0*iSMC_phi/4.0, iSMC_phi/10.0, iSMC_phi/10.0, iSMC_phi/10.0, iSMC_phi/10.0, iSMC_phi/10.0, iSMC_phi/10.0, iSMC_phi/10.0};
    Vp[0] = -3.0*iSMC_phi/4.0;
    Vp[1] = -iSMC_phi/2.0;
    Vp[2] = -iSMC_phi/4.0;
    Vp[3] = 0.0;
    Vp[4] = iSMC_phi/4.0;
    Vp[5] = iSMC_phi/2.0;
    Vp[6] = 3.0*iSMC_phi/4.0;
    Vp[7] = iSMC_phi/10.0;
    Vp[8] = iSMC_phi/10.0;
    Vp[9] = iSMC_phi/10.0;
    Vp[10] = iSMC_phi/10.0;
    Vp[11] = iSMC_phi/10.0;
    Vp[12] = iSMC_phi/10.0;
    Vp[13] = iSMC_phi/10.0;
    }
    //////////trajectory tracking
    if (SinWave == 1){
    zd0 = zd[0];
    SinAmp              = programParameter[15];
    SinPeriod           = programParameter[16]*1000;
    double pi           = 3.14159265;
    Per                 = 2*pi/SinPeriod;
    Serial.println("Sin Tracking is chosen");
    Serial.print("SinAmp = ");
    Serial.println(SinAmp);
    Serial.print("SinPeriod = ");
    Serial.println(SinPeriod);
  }
   if (StepWave == 1){
    zd0 = zd[0];
    StepTime     = programParameter[15];
    StepAmp      = programParameter[16];
    Serial.println("Step Tracking is chosen");
    Serial.print("Step Time = ");
    Serial.println(StepTime);
    Serial.print("Step Amplitude = ");
    Serial.println(StepAmp);
  }
    //
    Serial.println("Program Parameters taken: ");
  for(uint8_t u=0; u<25;u++){
  Serial.println(programParameter[u]);
  }
    //countdown
    motorResetPosition(alpha_max / 2, angle, stepCounter);
    LEDBlink(3, 200, 800);
    digitalWrite(SMC_LED, HIGH);
     //
    //calibrate pressure sensor and calculate target pressure
    P_0 = calculatePressure();
    xhat1 = 0.0;                 //estimated depth in [m]
    xhat2 = 0.0;                 //estimated velocity in [m/s]
    xhat1_prev[1] = 0.0;  //estimated depth at previous time step in [m]
    xhat2_prev[1] = 0.0;  //estimated velocity at previous time step in [m/s]
    xhat1_prev[2] = 0.0;  //estimated depth at previous time step in [m]
    xhat2_prev[2] = 0.0;  //estimated velocity at previous time step in [m/s]
    for(uint16_t calibration=0; calibration <120; calibration++){
    P = calculatePressure(); 
    z[0] = 10.0 * ((P - P_0) / 1000.0);
    z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau);
    z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau);
    delay(25);
    }
    //
    //reset running parameters
    timeStep = 0;
    valuesToSave[0] = timeStep;
    valuesToSave[1] = z[0];
    valuesToSave[2] = z[1];
    valuesToSave[3] = z[2];
    valuesToSave[4] = angle;
    valuesToSave[5] = alpha;
    valuesToSave[6] = zd[0];
    valuesToSave[7] = d_hat;
    //
    //store inital values
    writeArrayBufferLine(valuesToSave, iArray, dataArray);
    //
    //program loop
    overallTime = 0;
    looptime = 0;
    functionTime = 0;
    ////////////////////////////////////////////////////////////////////////// loop
    while (overallTime < (iSMC_timeMax + 1) && P < 1400) {
      // check battery
      if (batteryloop >= 400) {
        uint8_t Vcrit = voltageBattery();
        if (Vcrit == 1) {
          break;
        }
        batteryloop = 0;
      }
      P = calculatePressure();
      z[0] = 10.0 * ((P - P_0) / 1000.0);
      z[1] = get_xhat2(z[0], 0, SMO_rho, SMO_phi, SMO_tau);
      z[2] = get_xhat2(z[1], 1, SMO_rho, SMO_phi, SMO_tau);
      if(SinWave == 1){
      zd[0] =   zd0 + SinAmp*sin(Per*(overallTime));
      zd[1] =         SinAmp*cos(Per*(overallTime)) * Per *1000;
      zd[2] =       - SinAmp*sin(Per*(overallTime)) * Per * 1000 * Per * 1000;
      zd[3] =       - SinAmp*cos(Per*(overallTime)) * Per * 1000 * Per * 1000 * Per * 1000;
      Serial.print("zd[0] = ");
      Serial.println(zd[0],4);
      Serial.print("zd[1] = ");
      Serial.println(zd[1],4);
      Serial.print("zd[2] = ");
      Serial.println(zd[2],4);
      }      
      if(StepWave == 1){
      if(functionTime/1000 < StepTime){
      zd[0] =   zd0 + StepAmp;
      }
      else{
      zd[0] = zd0 - StepAmp;
      }
      if(functionTime/1000 >= 2*StepTime){
         functionTime = 0;
      }
      Serial.print("zd[0] = ");
      Serial.println(zd[0]);
      }      
      alpha = intelligentSlidingModeControler(z, zd, angle, iSMC_lambda, iSMC_eta, iSMC_phi, iSMC_a, iSMC_gamma, iSMC_neuron, Vw, Vp, d_hat);
      angle = step(alpha, angle, stepCounter);
      //
      //store vaules every storeValueCounter timeStep
      if (saveLoop == storeValueCounter) {
        valuesToSave[0] = timeStep;
        //valuesToSave[1] = P;
    valuesToSave[0] = timeStep;
    valuesToSave[1] = z[0];
    valuesToSave[2] = z[1];
    valuesToSave[3] = z[2];
    valuesToSave[4] = angle;
    valuesToSave[5] = alpha;
    valuesToSave[6] = zd[0];
    valuesToSave[7] = d_hat;
        writeArrayBufferLine(valuesToSave, iArray, dataArray);
        saveLoop = 0;
      }
      looptime = 0;
      batteryloop++;
      saveLoop++;
      timeStep++;
    }
    //
    //end programm
    emerge(angle, stepCounter);
    writeLastPageToEEPROM(dataArray, device_quarter, data_address1);
    digitalWrite(motorSleep_Pin, LOW);
    ESP8266WakeUp();
  }


  
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //if selected, the motor is driven to neutral position
  if (MotorReset == 1) {
    digitalWrite(motorSleep_Pin, HIGH);
    motorResetPosition(alpha_max / 2, angle, stepCounter);
    Serial.println("Motor position reseted");
    digitalWrite(motorSleep_Pin, LOW);
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //if selected, the motor is driven to position alpha
  if (MotorPosition == 1) {
    digitalWrite(motorSleep_Pin, HIGH);
    double  alpha = programParameter[15];
    Serial.print("alpha read = ");
    Serial.println(alpha);
    delay(1000);
    looptime = 0;
    while (abs(angle - alpha) > 0.08) {
      angle = step(alpha, angle, stepCounter);
      looptime = 0;
    }
    digitalWrite(motorSleep_Pin, LOW);
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //if selected, the last trial is read from EEPROM
  if (readEEPROM == 1) {
    readTrialFromEEPROM();
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //if selected, the last the RL matrices are stored to EEPROM
  if (RLstoreMatrices == 1) {
    storeRLMatrizesToEEPROM(transitionCounts1, transitionCounts2, value, rewardAmounts, rewardCounts, numFailures);
    LEDBlink(3, 100, 100);
  }
  //////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  //if selected, the motor is driven to closing position
  if (closeCylinder == 1) {
    digitalWrite(motorSleep_Pin, HIGH);
    delay(1000);
    while (abs(angle - alpha_max) > 0.08) {
      angle = step(alpha_max, angle, stepCounter);
      looptime = 0;
    }
    for (uint32_t i = 0; i < (50 * stepMode); i++) {
      digitalWrite(motorDir_Pin, HIGH);
      digitalWrite(motorStep_Pin, HIGH);
      delayMicroseconds(2);
      digitalWrite(motorStep_Pin, LOW);
      delayMicroseconds(stepTime);
      ++stepCounter;
    }
    digitalWrite(motorSleep_Pin, LOW);
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Functions of RL
///////////////////////////////////////////////////////////////////
//RL for diving cell
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////// selectNextAction
//This function calculates the sum of the expected values oof all possible successor states
//under action one and two and determines which action to take next.
//Additionally, exploration of the environment is undertaken with probability exploProb for failuresBeforeExploStop episodes
/////////////////////////////////////////////////////////////
int selectNextAction(uint8_t &randAction, const SparseMatrix<float> &transitionProbability1, const SparseMatrix<float> &transitionProbability2, const SparseMatrix<float> &reward,  const VectorXf &stateValue, const uint16_t numFailures, const uint16_t numFailuresBeforeExploStop) {
  double exploProb;
  /////calculating values of actions
  VectorXf  stateValue1(terminalState + 1);
  VectorXf  stateValue2(terminalState + 1);
  stateValue1.setZero();
  stateValue2.setZero();
  stateValue1 = transitionProbability1 * (reward * rewardCol1 + discount * stateValue); //calculate value of action1
  stateValue2 = transitionProbability2 * (reward * rewardCol2 + discount * stateValue); //calculate value of action2
  /////determine next action through values, in case of draw use random action
  exploProb = exploProb0 - exploProb0*numFailures/numFailuresBeforeExploStop;
  //generating random number for exploration and draw in values
  float  randNumber = (rand() % 100);
  if (numFailures < numFailuresBeforeExploStop) {
    randNum = randNumber / 100;
  }
  else {
    randNum = 1;
  }
  //check which value is higher and choose action accordingly
  if (stateValue1(state) > stateValue2(state)) {
    if (randNum > exploProb) {
      action = 1;
      randAction = 0;
    }
    else {
      action = 2;
      randAction = 1;
    }
  }
  else if (stateValue2(state) > stateValue1(state)) {
    if (randNum > exploProb) {
      action = 2;
      randAction = 0;
    }
    else {
      action = 1;
      randAction = 1;
    }
  }
  else {
    if (randNum > 0.5) {
      action = 1;
      randAction = 1;
    }
    else {
      action = 2;
      randAction = 1;
    }
  }
  return action;
}
///////////////////////////////////////////////////////////// rewardAndTransitionUpdate
//This function counts the transitions between states dependend on the selected action.
//Besides that reward is given based on the outcome of the last action
/////////////////////////////////////////////////////////////
void rewardAndTransitionUpdate(double (&z)[4], double (&zd)[4], uint8_t &newState, uint8_t &action, SparseMatrix<int> &transitionCounts1, SparseMatrix<int> &transitionCounts2, SparseMatrix<int> &rewardCounts, SparseMatrix<float> &rewardAmounts) {
  float R = 0;                           //reward signal
  //giving reward
  if (newState == terminalState) {
    R = -(float)10;
  }
  else {
    R = -(float)(fabs(z[0] - zd[0]));
  }
  //updating transitionCounts and rewardCounts
  if (action == 1) {
    uint32_t tCount =  transitionCounts1.coeff(state, newState);
    tCount = tCount + 1;
    transitionCounts1.coeffRef(state, newState) = tCount;
    int16_t rCount = rewardCounts.coeff(newState, 0);
    rCount = rCount + 1;
    rewardCounts.coeffRef(newState, 0) = rCount;
    float rAmount = rewardAmounts.coeff(newState, 0);
    rAmount = rAmount + R;
    rewardAmounts.coeffRef(newState, 0) =  rAmount;
  }
  else if (action == 2) {
    uint32_t tCount =  transitionCounts2.coeff(state, newState);
    tCount = tCount + 1;
    transitionCounts2.coeffRef(state, newState) = tCount;
    int16_t rCount = rewardCounts.coeff(newState, 1);
    rCount = rCount + 1;
    rewardCounts.coeffRef(newState, 1) = rCount;
    float rAmount = rewardAmounts.coeff(newState, 1);
    rAmount = rAmount + R;
    rewardAmounts.coeffRef(newState, 1) =  rAmount;
  }
}
///////////////////////////////////////////////////////////// valueIteration
//This functions calculates the new transition probabilities of the environment.
//Besides that value iteration is performed to aquire a new value function for the next episode
/////////////////////////////////////////////////////////////
void valueIteration(const SparseMatrix<int> &transitionCounts1, const SparseMatrix<int> &transitionCounts2, SparseMatrix<float> &transitionProbs1, SparseMatrix<float> &transitionProbs2, SparseMatrix<int> &rewardCounts, SparseMatrix<float> &rewardAmounts, SparseMatrix<float> &reward,  VectorXf &value) {
  VectorXi totalCounts1(terminalState);                                       //transition counts between state and newstate under action 1
  VectorXi totalCounts2(terminalState);                                       //transition counts between state and newstate under action 2
  uint32_t Counts1 = 0;                                                       //counting parameter
  uint32_t Counts2 = 0;                                                       //counting parameter
 

  for (uint16_t i = 0; i <= terminalState; i++) {
    for (uint16_t j = 0; j <= terminalState; j++) {
      Counts1 = Counts1 + transitionCounts1.coeff(i, j);
      Counts2 = Counts2 + transitionCounts2.coeff(i, j);
    }
    totalCounts1.coeffRef(i) = Counts1;
    totalCounts2.coeffRef(i) = Counts2;
    Counts1 = 0;
    Counts2 = 0;
  }
  //update values of transitionProbs1
  for (uint16_t i = 0; i <= terminalState; i++) {
    for (uint16_t j = 0; j <= terminalState; j++) {
      if (transitionCounts1.coeff(i, j) != 0) {
        transitionProbs1.coeffRef(i, j) = (float)transitionCounts1.coeff(i, j) / (float)totalCounts1(i);
      }
    }
  }
  totalCounts1.setZero();                           //setting transitionCounts1 to zero for next episode
  transitionProbs1.makeCompressed();                //compress transitionPobs1 for calculations and to save memory
  //updating values of transitionProbs2
  for (uint16_t i = 0; i <= terminalState; i++) {
    for (uint16_t j = 0; j <= terminalState; j++) {
      if (transitionCounts2.coeff(i, j) != 0) {
        transitionProbs2.coeffRef(i, j) = (float)transitionCounts2.coeff(i, j) / (float)totalCounts2.coeff(i);
      }
    }
  }
  totalCounts2.setZero();                           //setting transitionCounts2 to zero for next episode
  transitionProbs2.makeCompressed();                //compress transitionPobs2 for calculations and to save memory
  //update rewardCounts
  for(uint8_t j = 0; j<2; j++){
  for (uint16_t i = 0; i <= terminalState; i++) {
    if (rewardCounts.coeff(i,j) != 0) {
      reward.coeffRef(i,j) = (float)rewardAmounts.coeff(i,j) / (float)rewardCounts.coeff(i,j);
    }
  }
  }
  //policy evaluation to get new value function
  VectorXf newValue(terminalState + 1);
  uint16_t iterations = 0;
  while (true) {
    iterations = iterations + 1;
    VectorXf newValue1 = transitionProbs1 * (reward * rewardCol1 + (float)discount * value);  //calculating newValue under action 1 of all states
    VectorXf newValue2 = transitionProbs2 * (reward * rewardCol2 + (float)discount * value);  //calculating newValue under action 2 of all states
    newValue = newValue1.cwiseMax(newValue2);       //take maximum of both values of state s under action 1 and action 2
    //newValue(terminalState) = -10;
    VectorXf diff = value - newValue;               //determine change in value function
    diff = diff.cwiseAbs();                         //use absolut values
    value = newValue;
    //policy evaluation stops when change in value function is small enough
    float diffMax = diff(0);
    for (uint16_t i = 1; i < terminalState; i++) {
      if (diff(i) > diffMax) {
        diffMax = diff(i);
      }
    }
    if (diffMax < tolerance) {
      break;
    }
  }
}
///////////////////////////////////////////////////////////// updateRunningParameters
//This function keeps track of the running parameters like the total running time, the
//number of trials and the time steps per trial.
//For simulation purposses it also reinitiates the episode when DICE leaves the learning
//area.
/////////////////////////////////////////////////////////////
void updateRunningParameter(uint32_t &timeStepsTrial,
                            uint16_t &numFailures, VectorXi &timeStepsBeforeFailure) {
  /////update running parameters
  timeStepsBeforeFailure.conservativeResize(numFailures + 1); //resize timeStepsToFailures to save timeSteps
  timeStepsBeforeFailure(numFailures) = timeStepsTrial;        //save time steps of current trial
  timeStepsTrial = 0;                                          //reset timeStepsTrial for the next trial
  numFailures = numFailures + 1;                               //count number of failures up
}

///////////////////////////////////////////////////////////// get newState
//This functions discretizes the properties of the environment
//A state is created on basis of position and velocity of DICE
//The state is used for creating a model of the environment and in the value function
/////////////////////////////////////////////////////////////
int get_state(const double(&z)[4]) {
  if ((z[0] < (zd[0] - posTol)) || (z[0] > (zd[0] + posTol))) {
    newState = terminalState;
  }
  else {
    //position discretization
    {
      if ((z[0] >= (zd[0] - (posTol * 0.2))) && (z[0] <= zd[0])) {
        newState = 0;
      }
      else if ((z[0] <= (zd[0] + (posTol * 0.2))) && (z[0] > zd[0])) {
        newState = 1;
      }
      else if ((z[0] >= (zd[0] - (posTol * 0.4))) && (z[0] < zd[0] - (posTol * 0.2))) {
        newState = 2;
      }
      else if ((z[0] <= (zd[0] + (posTol * 0.4))) && (z[0] > zd[0] + (posTol * 0.2))) {
        newState = 3;
      }
      else if ((z[0] >= (zd[0] - (posTol * 0.6))) && (z[0] < zd[0] - (posTol * 0.4))) {
        newState = 4;
      }
      else if ((z[0] <= (zd[0] + (posTol * 0.6))) && (z[0] > zd[0] + (posTol * 0.4))) {
        newState = 5;
      }
      else if ((z[0] >= (zd[0] - (posTol * 0.8))) && (z[0] < zd[0] - (posTol * 0.6))) {
        newState = 6;
      }
      else if ((z[0] <= (zd[0] + (posTol * 0.8))) && (z[0] > zd[0] + (posTol * 0.6))) {
        newState = 7;
      }
      else if ((z[0] >= (zd[0] - posTol)) && (z[0] < zd[0] - (posTol * 0.8))) {
        newState = 8;
      }
      else {
        newState = 9;
      }
    }
    //velocity discretization
    {
      if ((z[1] >= -0.25 * velTol) && (z[1] <= 0)) {
        newState = newState + 10;
      }
      else if ((z[1] <= 0.25 * velTol) && (z[1] > 0)) {
        newState = newState + 20;
      }
      else if ((z[1] >= -0.5 * velTol) && (z[1] < -0.25 * velTol)) {
        newState = newState + 30;
      }
      else if ((z[1] <= 0.5 * velTol) && (z[1] > 0.25 * velTol)) {
        newState = newState + 40;
      }
      else if ((z[1] >= -0.75 * velTol) && (z[1] < -0.5 * velTol)) {
        newState = newState + 50;
      }
      else if ((z[1] <= 0.75 * velTol) && (z[1] > 0.5 * velTol)) {
        newState = newState + 60;
      }
      else if ((z[1] >= -velTol) && (z[1] < -0.75 * velTol)) {
        newState = newState + 70;
      }
      else if ((z[1] <= velTol) && (z[1] > 0.75 * velTol)) {
        newState = newState + 80;
      }
      else if (z[1] < -velTol) {
        newState = newState + 90;
      }
      else {
        newState = newState + 100;
      }
    }
  }
  return newState;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////SMC
///////////////////////////////////////////////////////////////////
//SMC for diving cell
///////////////////////////////////////////////////////////////////
double slidingModeControler(double (&z)[4], double (&zd)[4], double &z_prev, double angle, double lambda, double eta, double phi, double a) {
  double m_c = 0.3793; //mass in kg
  double rho_w = 1000; //density of water in kg/m^3
  double g = 9.81; //gravity in m/s^2
  double k = 1; //viscosian damping parameter in kg/m
  //double V_0 = 0.000379; //DIVE volume in m^3
  double A_dia = 0.000707; //membrane area in m^2
  double h_max = 0.02042; //maximum stroke from  null position
  double s;
  double V_var;
  double z_err[3];
  //
  //calculating error values
  z_err[0] = z[0] - zd[0];
  z_err[1] = z[1] - zd[1];
  z_err[2] = z[2] - zd[2];
  z_err[3] = z[3] - zd[3];
  //
  //calculate actual varialbe volume
  double V_act = A_dia * h_max * ((angle - (alpha_max / 2)) / alpha_max);
  //volume through controller
  //SMC 3rd order
  s = z_err[2] + 2 * lambda * z_err[1] + lambda * lambda * z_err[0];
  V_var = (1.0 / (a * rho_w * g)) * (a * rho_w * g * V_act - 2.0 * k * z[2] * abs(z[1]) - m_c * (zd[3] - 2.0 * lambda * z_err[2] - lambda * lambda * z_err[1]) + eta * sat(s, phi));
  //
  //robust Feedback SMC
  //s = z_err[2] + 2 * lambda * z_err[1] + lambda * lambda * z_err[0];
  //V_var = (1.0 / (a * rho_w * g)) * (a * rho_w * g * V_act - 2.0 * k * z[2] * abs(z[1]) - m_c * (zd[3] - 2.0 * lambda * z_err[2] -  pow(lambda, 2) * z_err[1]) + eta * sat(s, phi));
  //
  //3rd order SMC Integral
  //double z_integ = 0.5*(z_prev + z[0])*iterationtime;
  //z_prev = z[0];
  //s=z_err[2]+3*lambda*z_err[1]+3*pow(lambda,2)*z_err[0]+ pow(lambda,3)*z_integ;
  //V_var = (1.0 / (a * rho_w * g)) * (a * rho_w * g * V_act - 2.0 * k * z[2] * abs(z[1]) - m_c * (zd[3] - 3.0 * lambda * z_err[2] - 3.0 * pow(lambda, 2) * z_err[1] - pow(lambda, 3) * z_err[0]) + eta * sat(s, phi));
  //determine if control volume breaches max volume
  double alpha_c = alpha_max / 2 + (V_var / (A_dia * h_max / 2)) * alpha_max / 2;
  //ensure alpha is bounded
  if (alpha_c > alpha_max) {
    alpha_c = alpha_max;
  }
  else if (alpha_c < 0) {
    alpha_c = 0;
  }
  else {
    alpha_c = alpha_c;
  }
  return alpha_c;
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////iSMC
///////////////////////////////////////////////////////////////////
//iSMC for diving cell
///////////////////////////////////////////////////////////////////
double intelligentSlidingModeControler(double (&z)[4], double (&zd)[4], double angle, double lambda, double eta, double phi, double a, double gamma, uint8_t neuron, double (&Vw)[7], double (&Vp)[14], double &d_hat) {
  double m_c = 0.9*0.3793; //mass in kg
  double rho_w = 1000; //density of water in kg/m^3
  double g = 9.81; //gravity in m/s^2
  double k = 1; //viscosian damping parameter in kg/m
  //double V_0 = 0.000379; //DIVE volume in m^3
  double A_dia = 0.000707; //membrane area in m^2
  double h_max = 0.02042; //maximum stroke from  null position
  double s;
  double V_var;
  double z_err[3];
  //iSMC adaptions
  double Vr[7];
  for (uint8_t i=0; i<7; i++){
    Vr[i] = 0;
  }
  //
  //calculating error values
  z_err[0] = z[0] - zd[0];
  z_err[1] = z[1] - zd[1];
  z_err[2] = z[2] - zd[2];
  z_err[3] = z[3] - zd[3];
  //
  //calculate actual varialbe volume
  double V_act = A_dia * h_max * ((angle - (alpha_max / 2)) / alpha_max);
  //////////SMC 3rd order
  //slidung surface
  s = z_err[2] + 2 * lambda * z_err[1] + lambda * lambda * z_err[0];
  //neuronal net
  // Computes the regressor vector
  if (neuron == 0)  { // Triangular type
    if (s <= Vp[1]) {
      Vr[0] = 1.0;
    }
    else if (s < Vp[7]){
      for (uint8_t i = 0; i <= 6; i++)  {
        if ( s < Vp[i+2]){
          Vr[i] = max(min((s-Vp[i])/(Vp[i+1]-Vp[i]), (Vp[i+2]-s)/(Vp[i+2]-Vp[i+1])), 0.0);
          Vr[i+1] = max(min((s-Vp[i+1])/(Vp[i+2]-Vp[i+1]), (Vp[i+3]-s)/(Vp[i+3]-Vp[i+2])), 0.0);
        }
      }
    }
    else  {
      Vr[6] = 1.0;
    }
  }
  else if (neuron == 1) { // Gaussian type
    for (uint8_t i = 0; i <= 6; i++){  
    Vr[i] = exp(-0.5*(s-Vp[i])*(s-Vp[i])/(Vp[i+7]*Vp[i+7]));
    }
  }
  else{ 
  Serial.println("Not a neuron");
  }
  // Computes the intelligent compensation by means of a dot product
  d_hat = 0.0;
  for (uint8_t i = 0; i <= 6; i++)  { 
  d_hat = d_hat + Vr[i]*Vw[i];
  }
  // Updates the weight vector by means of the Euler method: y = y + f(x)*dt
  if (abs(s) < phi) for (uint8_t i=0; i<=6; i++)  {
    Vw[i] = Vw[i] + gamma*s*Vr[i]*iterationtimeSC*1e-6;
  }
  //control law
  V_var = (1.0 / (a * rho_w * g)) * (a * rho_w * g * V_act  - m_c * (zd[3] - 2.0 * lambda * z_err[2] - lambda * lambda * z_err[1]) + d_hat + (eta+abs(d_hat)) * sat(s, phi));

  Serial.print("V_var = ");
  Serial.println(V_var, 6);
  Serial.print("d_hat = ");
  Serial.println(d_hat, 6);
  //determine if control volume breaches max volume
  double alpha_c = alpha_max / 2 + (V_var / (A_dia * h_max / 2)) * alpha_max / 2;
  //ensure alpha is bounded
  if (alpha_c > alpha_max) {
    alpha_c = alpha_max;
  }
  else if (alpha_c < 0) {
    alpha_c = 0;
  }
  else {
    alpha_c = alpha_c;
  }
  return alpha_c;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// SMO
///////////////////////////////////////////////////////////////////
//The sliding mode observer to estimate the velocity of the diving
//cell from position information
///////////////////////////////////////////////////////////////////
// Esitmate derivative of x1
double get_xhat2(double x1, int n, double rho, double phi, double tau) {
  xhat1 = get_xhat1(x1, n, rho, phi, tau);
  xhat2 = xhat2_prev[n] + (double(iterationtime) / (double(1000) * tau)) * (-xhat2_prev[n] - rho * sat(xhat1 - x1, phi));
  xhat2_prev[n] = xhat2;
  return xhat2;
}
//
// Estimate ADA depth
double get_xhat1(double x1, int n, double rho, double phi, double tau) {
  xhat1 = xhat1_prev[n] - (double(iterationtime) / double(1000)) * rho * sat(xhat1_prev[n] - x1, phi);
  xhat1_prev[n] = xhat1;
  return xhat1;
}
//
// sat function
double sat(double x, double gamma) {
  double y = max(min(1.0, x / gamma), -1.0);
  return y;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Function of motor
///////////////////////////////////////////////////////////////////
//This function lets the diving cell emerge
//////////////////////////////////////////////////////////////////
void emerge(double &angle, int32_t &stepCounter) {
  allLEDHigh();
  while (abs(angle - alpha_max) > 0.16) {
    angle = step(alpha_max, angle, stepCounter);
    Serial.println("Emerge");
    Serial.print("angle = ");
    Serial.println(angle);
    Serial.print("alpha = ");
    Serial.println(alpha);
    looptime = 0;
    allLEDLow();
  }
}
///////////////////////////////////////////////////////////////////
//This function resets the position of the motor after start.
//The position is calibrated by stepping against the push switch,
//then the step counter is set onto -20 and the motor is driven to
//the desired alpha
//////////////////////////////////////////////////////////////////
void motorResetPosition(const double &alpha, double &angle, int32_t &stepCounter) {
  uint8_t resetOk = 0;                                //1 when reset successful
  //
  //reset loop
  while (resetOk != 1) {
    //
    //if push button is reached set stepCounter to -Switch_Steps
    if (digitalRead(Switch_Pin) == LOW) {
      stepCounter = -Switch_Steps;
      angle = (double(stepCounter) * double(anglePerStep)) / double(iplanet);
      //drive stepCounter to zero
      delay(200);
      while (stepCounter < 0) {
        digitalWrite(motorDir_Pin, HIGH);
        digitalWrite(motorStep_Pin, HIGH);
        delayMicroseconds(4);
        digitalWrite(motorStep_Pin, LOW);
        delayMicroseconds(stepTime - 4);
        ++stepCounter;
      }
      //drive motor to desired alpha
      while (abs(angle - alpha) > 0.08) {
        angle = step(alpha, angle, stepCounter) ;
        looptime = 0;
      }
      Serial.println("position reset ok");
      resetOk = 1;
    }
    //drive in the direction of the push button
    else {
      digitalWrite(motorDir_Pin, LOW);
      digitalWrite(motorStep_Pin, HIGH);
      delayMicroseconds(4);
      digitalWrite(motorStep_Pin, LOW);
      --stepCounter;
      delayMicroseconds(stepTime - 4);
    }
     //calculate acutal angle
    angle = (double(stepCounter) * double(anglePerStep)) / double(iplanet);
  }
}
///////////////////////////////////////////////////////////////////
//This function drives the motor to a desired alpha
//Thereby the whole looptime is used
//A ramp function for the step frequency ia used for better dynamics
//////////////////////////////////////////////////////////////////
double step(const double alpha, double angle, int32_t &stepCounter) {
  angle = (double(stepCounter) * double(anglePerStep)) / double(iplanet);
  double dalpha = alpha - angle;
  uint32_t requiredSteps = (int)abs((dalpha) * iplanet / anglePerStep);
  uint32_t localStepCounter = 0;
  uint32_t accelerationSteps = 15 * stepMode;
  //
  while (looptime < (iterationtime - 1)) {
    if (requiredSteps > 1) {
      if (dalpha > 0) {
        digitalWriteFast(motorDir_Pin, HIGH);
        while ((localStepCounter <= requiredSteps) && (looptime < (iterationtime - 1)) &&  angle < alpha) {
          digitalWrite(motorStep_Pin, HIGH);
          delayMicroseconds(4);
          digitalWrite(motorStep_Pin, LOW);
          if (localStepCounter < accelerationSteps) {
            delayMicroseconds(stepTime - 4);
          }
          else {
            delayMicroseconds(stepTime - 4);
          }
          ++stepCounter;
          ++localStepCounter;
          angle = (double(stepCounter) * double(anglePerStep)) / double(iplanet);
        }
      }
      else if (dalpha < 0) {
        digitalWrite(motorDir_Pin, LOW);
        while ((localStepCounter <= requiredSteps) && (looptime < (iterationtime - 1)) && angle >= 0 && angle > alpha) {
          digitalWrite(motorStep_Pin, HIGH);
          delayMicroseconds(4);
          digitalWrite(motorStep_Pin, LOW);
          if (localStepCounter < accelerationSteps) { //minimal 13 steps
            delayMicroseconds(stepTime - 4);
          }
          else {
            delayMicroseconds(stepTime - 4); // Halbe Schrittzeit warten
          }
          --stepCounter;
          //reset calibration if push botton is pressed
          if (digitalRead(Switch_Pin) == LOW) {
            stepCounter = -Switch_Steps;
          }
          ++localStepCounter;
          angle = (double(stepCounter) * double(anglePerStep)) / double(iplanet);
        }
      }
    }
    else {
    }
  }
  return angle;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Functions of ESP8266
///////////////////////////////////////////////////////////////////
//This functions configures the ESP8266 to act as an accesspoint
///////////////////////////////////////////////////////////////////
void ESPserverConfiguration() {
  sendData("AT+RST\r\n", 2000, DEBUG);              // reset module
  digitalWrite(ACTIVE_LED, HIGH);
  sendData("AT+CWMODE=2\r\n", 1000, DEBUG);         // configure as access point
  digitalWrite(RL_LED, HIGH);
  sendData("AT+CIFSR\r\n", 1000, DEBUG);            // get ip address
  digitalWrite(SMC_LED, HIGH);
  sendData("AT+CIPMUX=1\r\n", 1000, DEBUG);         // configure for multiple connections
  digitalWrite(DATA_LED, HIGH);
  sendData("AT+CIPSERVER=1,80\r\n", 1000, DEBUG);   // turn on server on port 80
  digitalWrite(ACTIVE_LED, LOW);
  digitalWrite(RL_LED, LOW);
  digitalWrite(SMC_LED, LOW);
  digitalWrite(DATA_LED, LOW);
}
///////////////////////////////////////////////////////////////////
//This function sents the ESP8266 into deep sleep for saving energy
///////////////////////////////////////////////////////////////////
void ESP8266DeepSleep() {
  sendData("AT+GSLP=0\r\n", 1000, DEBUG); // versetze Teensy in Deep Sleep
}
///////////////////////////////////////////////////////////////////
//Wake up function for the deep sleep modus
///////////////////////////////////////////////////////////////////
void ESP8266WakeUp() {
  digitalWriteFast(ESP8266Reset_Pin, LOW);
  delay(10);
  digitalWriteFast(ESP8266Reset_Pin, HIGH);
  ESPserverConfiguration();
}
///////////////////////////////////////////////////////////////////
//This functions starts the diving cell via the website
//It checks for get requests and reads the incoming data
//The incoming data is used to determine which program to start next
//Additionally, when choosing SMC, the parameters of the Controler
//can be updated or old parameters (stored in the Teensys EEPROM) are used.
///////////////////////////////////////////////////////////////////
void ESP8266programStart(float (&parameterArray)[25]) {
  uint16_t batteryloop = 0;                               //loop for batterycheck
  uint8_t programCode = 0;                                //program code sent from website
  uint8_t programStart = 0;                               //reset the while loop for selecting program mode
  uint8_t SMC = 0;                                        //variable for SMC program
  uint8_t iSMC = 0;                                       //variable for iSMC program
  uint8_t readEEPROM = 0;                                 //variable for readEEPROM program
  uint8_t MotorPosition = 0;                              //variable for readEEPROM program
  uint8_t MotorReset = 0;                                 //variable for reseting the motor to neutral position
  uint8_t closeCylinder = 0;                              //variable for diving cell closing position
  uint8_t RL = 0;                                         //variable for Machine Learning algorithm
  uint8_t RLstore = 0;                                    //variable to store matrices of RL
  uint8_t SMC_updated = 0;                                //determines if the SMC parameters are updated of default
  uint8_t SMO_updated = 0;
  uint8_t iSMC_updated = 0;
  uint8_t SinWave = 0;
  uint8_t StepWave = 0;
  //read data from EEPROM
  for (uint8_t i = 0; i <= 25; i++) {
    parameterArray[i] = 0;
  }
  //Program selection via website
  while (programStart != 1) {
    //check battery every 10s
    if (batteryloop == 25) {
      voltageBattery();
      batteryloop = 0;
    }
    ++batteryloop;
    //visualize readines for next command
    digitalWrite(ACTIVE_LED, LOW);
    delay(200);
    digitalWrite(ACTIVE_LED, HIGH);
    delay(200);
    //
    //process incomming get request
    if (ESP8266.available()) {                            //check if ESP is sending a message
      LEDBlink(1, 200, 200);                              //confirm incomming get request to user
      if (ESP8266.find("+IPD,")) {                        //find the part +IPD in the get request
        delay(1000);                                      //wait for the serial buffer to fill up (read all the serial data)
        //save connection id to close communication
        int connectionId = ESP8266.read() - 48;           //read connectionId for terminating connection
        ESP8266.find("modus=");                           //advance cursor to "modus="
        programCode = (ESP8266.read() - 48) * 10;         //get first number of the program code
        programCode += (ESP8266.read() - 48);             //get second number of program code
        //
        // select the program according to the programCode
        if (programCode == 1) {
          programStart = 1;                               //breaks the while loop to start program
          ESP8266.find("time=");                          //advance cursor to "SMC_maxTime="
          int runTime = (ESP8266.read() - 48) * 100;      //get first number of SMC_maxTime
          runTime += (ESP8266.read() - 48) * 10;          //get second number of SMC_maxTime
          runTime += (ESP8266.read() - 48);               //get second number of SMC_maxTime
          //
          ESP8266.find("depth=");                         //advance cursor to "depth="
          int targetDepth = (ESP8266.read() - 48) * 100;  //get first number of targetDepth
          targetDepth += (ESP8266.read() - 48) * 10;      //get second number of targetDepth
          targetDepth += (ESP8266.read() - 48);           //get third number of targetDepth
          //save runtime and target depth into array
          parameterArray[10] = (float)runTime * 1000;
          parameterArray[11] = (float)targetDepth / 100;
          Serial.println("DICE is started");
        }
        else if (programCode == 02) {
          SMC = 1;                                        //variable for SMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
          allLEDLow();
          digitalWrite(SMC_LED, HIGH);
          Serial.println("SMC is selected");
        }
        else if (programCode == 03) {
          SMC = 0;                                        //variable for SMC program
          readEEPROM = 1;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
          allLEDLow();
          digitalWrite(DATA_LED, HIGH);
          Serial.println("Read EEPROM is selected");
        }
        //
        //Udate of the SMC contollers values
        else if (programCode == 04) {
          SMC = 1;                                        //variable for SMC program
          iSMC = 0;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
          allLEDLow();
          digitalWrite(SMC_LED, HIGH);
          digitalWrite(DATA_LED, HIGH);
          //
          //read SMC parameters from get request
          ESP8266.find("lam=");                               //advance cursor to "SMC_lambda="
          int SMC_lambda = (ESP8266.read() - 48) * 100;       //get first number of SMC_lambda
          SMC_lambda += (ESP8266.read() - 48) * 10;           //get second number of SMC_lambda
          SMC_lambda += (ESP8266.read() - 48);                //get third number of SMC_lambda
          //
          ESP8266.find("eta=");                               //advance cursor to "SMC_eta="
          int SMC_eta = (ESP8266.read() - 48) * 100;          //get first number of SMC_eta
          SMC_eta += (ESP8266.read() - 48) * 10;              //get second number of SMC_eta
          SMC_eta += (ESP8266.read() - 48);                   //get third number of SMC_eta
          //
          ESP8266.find("phi=");                               //advance cursor to "SMC_eta="
          int SMC_phi = (ESP8266.read() - 48) * 100;          //get first number of SMC_eta
          SMC_phi += (ESP8266.read() - 48) * 10;              //get second number of SMC_eta
          SMC_phi += (ESP8266.read() - 48);                   //get third number of SMC_eta
          //save parameters to array
          parameterArray[12] = (float)SMC_lambda / 100;
          parameterArray[13] = (float)SMC_eta / 100;
          parameterArray[14] = (float)SMC_phi / 100;
          SMC_updated = 1;
        }
        else if (programCode == 05) {
          allLEDLow();
          SMC = 0;                                        //variable for SMC program
          iSMC = 0;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 1;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
        }
        else if (programCode == 06) {
          allLEDLow();
          SMC = 0;                                        //variable for SMC program
          iSMC = 0;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 1;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
          //
          //read alpha from get request for motor positioning
          ESP8266.find("Alpha=");                             //advance cursor to "SMC_eta="
          int Motor_alpha = (ESP8266.read() - 48) * 100;      //get first number of SMC_maxTime
          Motor_alpha += (ESP8266.read() - 48) * 10;          //get first number of SMC_maxTime
          Motor_alpha += (ESP8266.read() - 48);               //get second number of SMC_maxTime
          //
          //ensure that alpha lies in borders
          if (Motor_alpha > alpha_max) {
            Motor_alpha = alpha_max;
          }
          else if (Motor_alpha < 0) {
            Motor_alpha = 0;
          }
          parameterArray[15] = Motor_alpha;
          Serial.print("Selected Alpha: ");
          Serial.println(parameterArray[15]);
        }
        else if (programCode == 07) {
          allLEDLow();
          SMC = 0;                                        //variable for SMC program
          iSMC = 0;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 1;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
        }
        else if (programCode == 8) {
          allLEDLow();
          SMC = 0;                                        //variable for SMC program
          iSMC = 1;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
          digitalWrite(SMC_LED, HIGH);
          digitalWrite(DATA_LED, HIGH);
          //
          //read SMC parameters from get request
          ESP8266.find("gam=");                               //advance cursor to "SMC_lambda="
          int SMC_gam = (ESP8266.read() - 48) * 100;          //get first number of SMC_eta
          SMC_gam += (ESP8266.read() - 48) * 10;              //get second number of SMC_eta
          SMC_gam += (ESP8266.read() - 48);                   //get third number of SMC_eta
          ESP8266.find("neu=");                               //advance cursor to "SMC_lambda="
          int SMC_neu = (ESP8266.read() - 48);          //get first number of SMC_eta
          parameterArray[20] = (float)SMC_gam / 10;
          parameterArray[21] = (float)SMC_neu;      
          iSMC_updated = 1; 
        }
        else if (programCode == 9) {
          allLEDLow();
          SMC = 0;                                        //variable for SMC program
          iSMC = 0;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 1;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
          digitalWrite(RL_LED, HIGH);
          //
          //read SMC parameters from get request
          ESP8266.find("use=");                               //advance cursor to "SMC_lambda="
          int RL_useExistingParameters = (ESP8266.read() - 48);       //get first number of SMC_lambda
          //
          ESP8266.find("explo=");                             //advance cursor to "SMC_eta="
          int RL_numExplo = (ESP8266.read() - 48) * 100;      //get first number of SMC_maxTime
          RL_numExplo += (ESP8266.read() - 48) * 10;          //get first number of SMC_maxTime
          RL_numExplo += (ESP8266.read() - 48);               //get second number of SMC_maxTime
          //
          //save parameters to array
          if (RL_useExistingParameters == 1) {
            digitalWrite(DATA_LED, HIGH);
          }
          parameterArray[15] = (float)RL_useExistingParameters;
          parameterArray[16] = (float)RL_numExplo;
        }
        else if (programCode == 10) {
          allLEDLow();
          SMC = 0;                                        //variable for SMC program
          iSMC = 0;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 1;                                    //variable to store matrices of RL
          digitalWrite(RL_LED, HIGH);
          digitalWrite(DATA_LED, HIGH);
          digitalWrite(SMC_LED, HIGH);
        }
        else if (programCode == 11) {
          allLEDLow();
          SMC = 1;                                        //variable for SMC program
          iSMC = 0;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
          SinWave = 1;
          StepWave = 0;
          allLEDHigh();
          digitalWrite(RL_LED, HIGH);
          digitalWrite(DATA_LED, HIGH);
          digitalWrite(SMC_LED, HIGH);
          Serial.println("Sin wave program selected");
          //
          //read alpha from get request for motor positioning
          ESP8266.find("SinA=");                             
          int SinAmp = (ESP8266.read() - 48) * 10;      
          SinAmp += (ESP8266.read() - 48);    
          //
          ESP8266.find("SinB=");                             
          int SinPeriod = (ESP8266.read() - 48) * 10;      
          SinPeriod += (ESP8266.read() - 48);    
          parameterArray[15] = (float)SinAmp/100;
          parameterArray[16] = (float)SinPeriod;                
        }
        else if (programCode == 12) {
          allLEDLow();
          SMC = 1;                                        //variable for SMC program
          iSMC = 0;                                       //variable for iSMC program
          readEEPROM = 0;                                 //variable for readEEPROM program
          MotorPosition = 0;                              //variable for readEEPROM program
          MotorReset = 0;                                 //variable for reseting the motor to neutral position
          closeCylinder = 0;                              //variable for diving cell closing position
          RL = 0;                                         //variable for Machine Learning algorithm
          RLstore = 0;                                    //variable to store matrices of RL
          SinWave = 0;
          StepWave = 1;
          digitalWrite(RL_LED, HIGH);
          digitalWrite(DATA_LED, HIGH);
          digitalWrite(SMC_LED, HIGH);
          Serial.println("Step wave program selected");
          //
          //read alpha from get request for motor positioning
          ESP8266.find("StA=");                             
          int StepTime = (ESP8266.read() - 48) * 10;      
          StepTime += (ESP8266.read() - 48);    
          //
          ESP8266.find("StB=");                             
          int StepAmp = (ESP8266.read() - 48) * 10;      
          StepAmp += (ESP8266.read() - 48);    
          parameterArray[15] = (float)StepTime;
          parameterArray[16] = (float)StepAmp/100;  
        }
        else if (programCode == 13) {
          allLEDLow();
          digitalWrite(RL_LED, HIGH);
          digitalWrite(DATA_LED, HIGH);
          digitalWrite(SMC_LED, HIGH);
          Serial.println("SMO updated");
          //
          //read alpha from get request for motor positioning
          ESP8266.find("rho=");                             
          int SMOrho = (ESP8266.read() - 48) * 100;      
          SMOrho += (ESP8266.read() - 48) * 10;   
          SMOrho += (ESP8266.read() - 48); 
          //
          ESP8266.find("phi=");                             
          int SMOphi = (ESP8266.read() - 48) * 100;      
          SMOphi += (ESP8266.read() - 48 )* 10;    
          SMOphi += (ESP8266.read() - 48 ); 

          ESP8266.find("tau=");                             
          int SMOtau = (ESP8266.read() - 48) * 100;      
          SMOtau += (ESP8266.read() - 48) * 10;  
          SMOtau += (ESP8266.read() - 48);
            
          parameterArray[17] = (float)SMOrho/10;
          parameterArray[18] = (float)SMOphi/100;  
          parameterArray[19] = (float)SMOtau/100;
          SMO_updated = 1;
          Serial.print("SMO updated = ");
          Serial.println(SMO_updated);
        }
        //
        // Close connection to webside
        String closeCommand = "AT+CIPCLOSE=";
        closeCommand += connectionId;                         // append connection id
        closeCommand += "\r\n";
        sendData(closeCommand, 1000, DEBUG);                  // close connection
        Serial.println("Connection closed");
      }
    }
  }
  //store program parameters in parameterArray
  parameterArray[0] = SMC;
  parameterArray[1] = readEEPROM;
  parameterArray[2] = MotorReset;
  parameterArray[3] = MotorPosition;
  parameterArray[4] = closeCylinder;
  parameterArray[5] = RL;
  parameterArray[6] = RLstore;
  parameterArray[7] = iSMC;
  parameterArray[8] = SinWave;
  parameterArray[9] = StepWave;
  //if SMC parameters are not updated, use default values
  if (SMC_updated == 0) {
    parameterArray[12] = float(SMC_Lambda_default);
    parameterArray[13] = float(SMC_eta_default);
    parameterArray[14] = float(SMC_phi_default);
  }
  if (SMO_updated == 0) {
    parameterArray[17] = float(SMO_rho_default);
    parameterArray[18] = float(SMO_phi_default);
    parameterArray[19] = float(SMO_tau_default);
  }
  if (iSMC_updated == 0) {
    parameterArray[20] = float(iSMC_gam_default);
    parameterArray[21] = float(iSMC_neu_default);
  }
}
///////////////////////////////////////////////////////////////////
//This function initiates the connection to the website and sents
//the data
///////////////////////////////////////////////////////////////////
String sendData(String command, const int timeout, boolean debug) {
  String response = "";
  ESP8266.print(command);                           //send the command to the ESP8266
  long int time = millis();
  while ((time + timeout) > millis()) {             //wait for an answer until timeout is reached
    while (ESP8266.available()) {                   //the esp has data so display its output to the serial window
      char c = ESP8266.read();                      //read the next character of the anser
      response += c;                                //and attach it to the response
    }
  }
  if (debug) {
    Serial.print(response);                         //print response
  }
  return response;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Functions of EEPROM
///////////////////////////////////////////////////////////////////
//This function organizes the data written to EEPROM
//Data is written into rows representing a certain time step
//If the maximum number of rows per EEPROM page is reached,
//the array is written to EEPROM
//////////////////////////////////////////////////////////////////
void writeArrayBufferLine(float (&valuesToSave)[valuesPerRow], uint8_t &iArray, long (&dataArray)[rowPageMax][valuesPerRow]) {
  uint16_t scaleFactor = 10000;                                       //scalefactor to save values as long int
  if (iArray < rowPageMax) {                                          //as long as iArray does not reach the page max of the EEPROM
    for (uint8_t i = 0; i < valuesPerRow; i++) {                      //organize the data in rows
      dataArray[iArray][i] = (long)(scaleFactor * valuesToSave[i]);
    }
    ++iArray;                                                         //raise iArray for next row
    if (iArray == rowPageMax) {                                       //if the maximum ammount of rows per page is reached
      writePageToEEPROM(dataArray, device_quarter, data_address1);    //write the page to the EEPROM
      iArray = 0;                                                     //reset iArray for next page
    }
  }
}
///////////////////////////////////////////////////////////////////
//This function writes a page of data to the EEPROM
//The EEPROM consits of 2 chips with 1024 pages each
//The pages are organized in 4 quarters with 255 pages each
//Data is orgaized in rows with values to save
//////////////////////////////////////////////////////////////////
void writePageToEEPROM(long (&dataArray)[rowPageMax][valuesPerRow], uint8_t &device_quarter, uint16_t &data_address1) {
  uint8_t addressByte = 0;                                                          //address of a byte after conversion of long to bytes
  uint8_t data[sizeof(dataArray)];                                                  //vector for storing all bytes to be written on page
  uint8_t iData = 0;                                                                //row variable of dataArray
  while (iData < rowPageMax) {                                                      //Following block converts the data from long integer to byte format
    for (uint8_t n = 0; n < valuesPerRow; n++) {                                    //
      for (uint8_t a = 0; a <= 3; a++) {                                            //
        data[addressByte + a] = (dataArray[iData][n] >> ((3 - a) * 8)) & 0x000000ff;//
      }
      addressByte = addressByte + 4;                                                //reset addressByte for next long int
    }
    ++iData;                                                                        //raise row of dataArray to convert next row
  }
  //
  //warning if chip is full
  if (device_quarter >= 4) {
    Serial.println("EEPROM chip full");
  }
  //else write page
  else {
    uint8_t writeAddress = (M24M02DRC_1_DATA_ADDRESS | device_quarter);             //set address of quarter to write to
    Wire.beginTransmission(writeAddress);                                           //establish I2C connection; Set quarter address
    Wire.write(data_address1);                                                      //set page address
    Wire.write(data_address2);                                                      //set byte address
    //write all bytes of data into EEPROM
    for (uint8_t i = 0; i < (rowPageMax * valuesPerRow * 4); i++) {
      Wire.write(data[i]);
    }
    delay(10);                                                                      //delay of 10ms per write operation
    Wire.endTransmission();                                                         //end transmission to EEPROM
    Serial.print("Page ");
    Serial.print(data_address1);
    Serial.print(" in quarter ");
    Serial.print(device_quarter);
    Serial.println(" is written");
    data_address2 = 0;                                                              //reset byte address
    ++data_address1;                                                                //raise page number
    if (data_address1 == 256) {
      ++device_quarter;                                                             //if last page in quarter is reached, change quarter
      data_address1 = 0;                                                            //reset page in new quarter
    }
  }
}
///////////////////////////////////////////////////////////////////
//This function writes the last page of data to the EEPROM
//The EEPROM consits of 2 chips with 1024 pages each
//The pages are organized in 4 quarters with 255 pages each
//Data is orgaized in rows with values to save
//////////////////////////////////////////////////////////////////
void writeLastPageToEEPROM(long (&dataArray)[rowPageMax][valuesPerRow], uint8_t &device_quarter, uint16_t &data_address1) {
  float valuesToSave[valuesPerRow];
  for (uint8_t i = 0; i < valuesPerRow; i++) {
    valuesToSave[i] = 0;
  }
  uint8_t rowsToSave = rowPageMax - iArray;
  while (rowsToSave > 0) {
    writeArrayBufferLine(valuesToSave, iArray, dataArray);
    rowsToSave--;
  }
  //set the whole WIFI buffer in the reading process to zero
  for (uint8_t i = 0; i < 2 * rowPageMax; i++) {
    writeArrayBufferLine(valuesToSave, iArray, dataArray);
  }
  //memorize writing parameters in EEPROM for recovery when it should be read out
  long EEPROMdata[30];                  //array consiting of all relevant information
  uint16_t scaleFactor = 10000;
  byte dataByte;
  int address = 0;
  EEPROMdata[25] = (long)(scaleFactor * (device_quarter * 256 + data_address1));
  EEPROMdata[26] = (long)(scaleFactor * P_0);
  EEPROMdata[27] = (long)(scaleFactor * P_target);
  //
  //save parameterArray in EEPROM
  for (uint8_t i = 0; i < 25; i++) {                                           
    EEPROMdata[i] = (long)(scaleFactor * programParameter[i]);
  }
  //
  //write data on Teensys EEPROM
  for (uint8_t n = 0; n <= 30; ++n) {
    for (uint8_t a = 0; a <= 3; ++a) {
      dataByte = (EEPROMdata[n] >> ((3 - a) * 8)) & 0x000000ff;
      EEPROM.write(address + a, dataByte);
    }
    address = address + 4;
  }
  //set data_adress1 to zero for next writing operation
  data_address1 = 0;
  device_quarter = 0;
}
///////////////////////////////////////////////////////////////////
//This function reads the ammount of given pages from the EEPROM
//The amount is stored in the EEPROM of the Teensy by
//writeLastPageToEEPROM();
//The EEPROM consits of 2 chips with 1024 pages each
//The pages are organized in 4 quarters with 255 pages each
//////////////////////////////////////////////////////////////////
void readTrialFromEEPROM() {
  uint16_t batteryloop = 0;                                               //loop for batterycheck
  uint8_t addressByte = 0;                                                //address of byte in vector or read data
  long dataValue;                                                         //value of read data
  long dataVector[(rowPageMax * valuesPerRow)][10];                      //vector of all values on an EEPROM page
  uint8_t bytesToRead = rowPageMax * valuesPerRow * 4;                    //number of bytes to read from EEPROM page
  uint8_t dataByte = 0;                                                   //data byte read from EEPROM
  uint8_t data[256];                                                      //data vector of all read bytes
  uint8_t quarter = 0;                                                    //current quarter to read from (4 quarters per chip)
  uint16_t pageQuarter = 0;                                               //page to read on current quarter (255 pages per quarter, 4 quarters per chip)
  uint8_t pagesInWifiBuffer = 0;                                          //actual number of pages in the Wifi buffer
  uint8_t pagesInWifiBufferMax = 2;                                       //number of pages read from EEPROM before sent via Wifi
  uint8_t EEPROMread = 0;                                                 //variable determines when reading stops (1 all pages read, stop function)
  float EEPROMdata[30];                                                   //array of Information about the last Trial
  uint16_t scaleFactor = 10000;                                           //scaling factor of the data
  long valueRead;                                                         //value to be read from EEPROM
  int address = 0;                                                        //address of byte to be read from EEPROM
  //
  //read last trials parameters from Teensy EEPROM
  for (uint8_t n = 0; n < 30; n++) {
    valueRead = 0;
    for (int i = 0; i < 3; i++) {
      valueRead += EEPROM.read(address + i);
      valueRead = valueRead << 8;
    }
    valueRead += EEPROM.read(address + 3);
    address = address + 4;
    //convert data read from EEPROM
    EEPROMdata[n] = (float)valueRead / (float)scaleFactor;
  }
  //
  uint16_t  pagesToRead = int(EEPROMdata[25]);                             //ammount of pages used during the last trial
  digitalWrite(ACTIVE_LED, LOW);
  //
  //stay in loop until data is read from EEPROM
  while (EEPROMread != 1) {
    //check battery
    if (batteryloop == 30) {
      voltageBattery();
      batteryloop = 0;
    }
    ++batteryloop;
    //
    //visualize readiness for next command
    digitalWrite(DATA_LED, LOW);
    delay(200);
    digitalWrite(DATA_LED, HIGH);
    delay(100);
    //
    //wait for get request
    if (ESP8266.available()) {                                            //ESP8266 check if website is asking for data
      if (ESP8266.find("+IPD,")) {                                        //If website is asking search for +IPD in get request
        delay(500);
        //read get request for clising connection
        int connectionId = ESP8266.read() - 48;                           //The next number determines the connection ID
        //Build Header
        String header = "<h1>Hydra measured values</h1>";                 //prepare header text
        header += "<br/>";
        header += "Battery voltage: ";
        header += String(vBat);//("%f", vBat[1]);
        Serial.print("header = ");
        Serial.println(header);
        header += " V";
        header += "<br/>";
        header += "Total program time: ";
        header += String(EEPROMdata[10] / 1000);
        header += " s";
        header += "<br/>";
        header += "Target depth: ";
        header += String(EEPROMdata[11]);
        header += " m ";
        header += "<br/>";
        header += "Initial pressure: ";
        header += String(EEPROMdata[26]);
        header += " mPa";
        header += "<br/>";
        header += "Target pressure: ";
        header += String(EEPROMdata[27]);
        header += " mPa";
        header += "<br/>";
        header += "<br/>";
        header += "SC";
        header += "<br/>";
        header += "lambda = ";
        header += String(EEPROMdata[12]);
        header += "<br/>";
        header += "eta = ";
        header += String(EEPROMdata[13]);
        header += "<br/>";
        header += "phi = ";
        header += String(EEPROMdata[14]);
        header += "<br/>";
        header += "<br/>";
        if (EEPROMdata[7] == 1) {
          header += "iSC";
          header += "<br/>";
          header += "gamma = ";
          header += String(EEPROMdata[20]);
          header += "<br/>";
          header += "neuron= ";
          header += String(EEPROMdata[21]);
          header += "<br/>";
          header += "<br/>";
        }
        header += "SMO";
        header += "<br/>";
        header += "rho = ";
        header += String(EEPROMdata[17]);
        header += "<br/>";
        header += "phi = ";
        header += String(EEPROMdata[18]);
        header += "<br/>";
        header += "tau = ";
        header += String(EEPROMdata[19]);
        header += "<br/>";
        header += "<br/>";
        header += "Scale factor of data: 10000";
        header += "<br/>";
        if (EEPROMdata[5] == 1) {
          header += "num Failures max= ";
          header += String(EEPROMdata[10]);
          header += "<br/>";
          header += "use existing properties= ";
          header += String(EEPROMdata[15]);
          header += "<br/>";
          header += "num Failures exploration stop= ";
          header += String(EEPROMdata[16]);
          header += "<br/>";
          header += "<br/>";
          header += "timeStep; z [m]; z_dot [m/s]; state; anlge [deg]; action; randAction; numFailures";
        }
        else if (EEPROMdata[7] == 1) {
          header += "timeStep; z [m]; z_dot [m/s]; z_ddot [m/s2]; anlge [deg]; alpha [deg]; zd[m]; d_hat";
        }
        else if (EEPROMdata[8] == 1) {
          header += "Sin Amp = ";
          header += String(EEPROMdata[15]);
          header += "<br/>";
          header += "Sin Time= ";
          header += String(EEPROMdata[16]);
          header += "<br/>";
          header += "<br/>";
          header += "timeStep; z [m]; z_dot [m/s]; z_ddot [m/s2]; anlge [deg]; alpha [deg]; zd[m]; zd_dot[m/s]; zd_ddot[m/s2]";
        }
        else if (EEPROMdata[9] == 1) {
          header += "Step Time= ";
          header += String(EEPROMdata[15]);
          header += "<br/>";
          header += "Step Amp= ";
          header += String(EEPROMdata[16]);
          header += "<br/>";
          header += "<br/>";
          header += "timeStep; z [m]; z_dot [m/s]; z_ddot [m/s2]; anlge [deg]; alpha [deg]; zd[m]";
        }
       else if (EEPROMdata[0] == 1) {
          header += "timeStep; z [m]; z_dot [m/s]; z_ddot [m2/s]; anlge [deg]; alpha [deg];";
        }
        header += "<br/>";
        header += "<br/>";
        //
        //send header 
        String cipSend = "AT+CIPSEND=";                                   //AT+CHIPSEND= begins transmission
        cipSend += connectionId;                                          //connection ID determines contact partner
        cipSend += ",";
        cipSend += header.length();                                       //put in information about the length of the information
        cipSend += "\r\n";
        sendData(cipSend, 500, DEBUG);                                    //initialize the connections
        sendData(header, 500, DEBUG);                                     //send header to website
        //
        //reading of data from EEPROM
        for (uint16_t page = 0; page < pagesToRead; page++) {             //pageToRead restored from Teensy EEPROM
          //
          //Termination requirement
          P = calculatePressure();
          if (P > 1200) {
            break;
          }
          //
          //Changes quarter when page limit is reached
          if (pageQuarter == 256) {
            quarter = quarter + 1;
            pageQuarter = 0;
          }
          //
          // Establish I2C connection and read page from EEPROM
          delay(10);
          Wire.beginTransmission((M24M02DRC_1_DATA_ADDRESS | quarter));   //Initialize the Tx buffer
          Wire.write(pageQuarter);                                        //Put slave register address (data_address1) in Tx buffer
          Wire.write(0);                                                  //Put slave register address (data_address2) in Tx buffer
          Wire.endTransmission(I2C_NOSTOP);                               //Send the Tx buffer, but send a restart to keep connection alive
          uint8_t readAddress = (0x50 | 0);                               //Set current quarter address
          Wire.requestFrom(readAddress, (size_t) bytesToRead);                  //Read the next bytesToRead bytes from EEPROM starting at data_address1, data_address2
          while (Wire.available()) {
            data[dataByte++] = Wire.read();                               //save bytes to data[i]
          }
          //
          // Convert bytes back to long int value
          for (uint8_t i = 0; i < (bytesToRead / 4); i++) {
            for (uint8_t j = 0; j < 3; j++) {
              dataValue += data[addressByte + j];
              dataValue = dataValue << 8;
            }
            dataValue += data[addressByte + 3];
            addressByte = addressByte + 4;                                            //Set address_byte to next value
            //
            //store values in wifi buffer
            dataVector[i][pagesInWifiBuffer] = dataValue;   //save converted long int as float value in dataValue[i]
            dataValue = 0;
          }
          //if Wifi Buffer is full, sent data to website
          if (pagesInWifiBuffer == pagesInWifiBufferMax) {
            String datenSatz = "";                                                    //Initiate the string of data to sent
            String zwischen = "; ";                                                   //operator to seperate values
            //
            //Build message to send to website
            uint8_t addressValue = 0;                                                 //address in datavector to read from
            for (uint8_t page = 0; page <= pagesInWifiBufferMax; page++) {
              for (uint8_t row = 0; row < rowPageMax; row++) {
                for (uint8_t val = 0; val < valuesPerRow; val++) {
                  datenSatz += String(dataVector[addressValue][page]);             //Put dataValue into datenSatz in float format
                  datenSatz += zwischen;                                              //place seperator between values
                  addressValue++;                                                     //raise addressValue for the next data value
                }
                datenSatz += "<br/>";                                                 //put in a wordwrap after each row
              }
              addressValue = 0;                                                       //reset adressValue fo the next pages data
            }
            //sent page buffer to website
            String cipSend = "AT+CIPSEND=";
            cipSend += connectionId;
            cipSend += ",";
            cipSend += datenSatz.length();
            cipSend += "\r\n";
            sendData(cipSend, 300, DEBUG);
            sendData(datenSatz, 300, DEBUG);
            pagesInWifiBuffer = -1;                                       //reset buffer of pages to -1
          }
          ++pageQuarter;                                                  //raise pageQuarter to read next page
          ++pagesInWifiBuffer;                                            //raise number in Wifi buffer
        }
        //close connection
        String closeCommand = "AT+CIPCLOSE=";                             //initialize the closing command
        closeCommand += connectionId;                                     //append connection id
        closeCommand += "\r\n";
        sendData(closeCommand, 3000, DEBUG);
        EEPROMread = 1;                                                   //change status of EEPROMread to 1 to terminate function
        digitalWrite(DATA_LED, LOW);
      }
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// RL Matrizes on EEPROM
///////////////////////////////////////////////////////////////////
//This functions reads sparse Matrizes from the external EEPROM.
//It can read float and integer matrices in the form of:
// row, col, value
//and inserts them into the new matrix form matrix.insert(row, col) = value.
//////////////////////////////////////////////////////////////////
void readRLMatrizesFromEEPROM(SparseMatrix<int> &integerMatrix1, SparseMatrix<int> &integerMatrix2, VectorXf &floatVector1, SparseMatrix<float> &floatMatrix1, SparseMatrix<int> &integerMatrix3, uint16_t &numFailures) {
  Serial.println("Read matrix parameters from EEPROM");
  uint16_t EEPROMdataRestored[15];                                         //The parameters of the 5 Matrizes to save are at the end of the Teensys EEPROM
  uint16_t address = 2048 - ((5 * 3 +1) * 4);                                   //starting address for parameter aquisation
  long valueRead;                                                          //value to be read from EEPROM
  //
  //read last trials parameters from Teensy EEPROM
  for (uint8_t n = 0; n <= 15; n++) {                                       //read 15 values of long int format from EEPROM (5 matrices a 3 values)
    valueRead = 0;
    for (int i = 0; i < 3; i++) {
      valueRead += EEPROM.read(address + i);
      valueRead = valueRead << 8;
    }
    valueRead += EEPROM.read(address + 3);
    address = address + 4;
    //convert data read from EEPROM
    EEPROMdataRestored[n] = valueRead;
    //Serial.println("EEPROM DATA RESTORED = ");                            //If active, the parameters of the matrizes are shown
    //Serial.println(EEPROMdataRestored[n]);
  }
  //
  //restore the form of saving with quarter, starting page and pages to read
  //
  //matrix 1
  uint8_t device_quarter_C2_MAT1 = EEPROMdataRestored[0];
  uint16_t data_address1_C2_MAT1 = EEPROMdataRestored[1];
  uint16_t pagesToRead_MAT1 = EEPROMdataRestored[2];
  Serial.println("Address and pages to read = ");
  Serial.println("Matrix 1");
  Serial.print(device_quarter_C2_MAT1);
  Serial.print("  ");
  Serial.print(data_address1_C2_MAT1);
  Serial.print("  ");
  Serial.print(pagesToRead_MAT1);
  Serial.println();
  //matrix 2
  uint8_t device_quarter_C2_MAT2 = EEPROMdataRestored[3];
  uint16_t data_address1_C2_MAT2 = EEPROMdataRestored[4];
  uint16_t pagesToRead_MAT2 = EEPROMdataRestored[5];
  Serial.println("Matrix 2");
  Serial.print(device_quarter_C2_MAT2);
  Serial.print("  ");
  Serial.print(data_address1_C2_MAT2);
  Serial.print("  ");
  Serial.print(pagesToRead_MAT2);
  Serial.println();
  //vector1
  uint8_t device_quarter_C2_MAT3 = EEPROMdataRestored[6];
  uint16_t data_address1_C2_MAT3 = EEPROMdataRestored[7];
  uint16_t pagesToRead_MAT3 = EEPROMdataRestored[8];
  Serial.println("Vector 1");
  Serial.print(device_quarter_C2_MAT3);
  Serial.print("  ");
  Serial.print(data_address1_C2_MAT3);
  Serial.print("  ");
  Serial.print(pagesToRead_MAT3);
  Serial.println();
  // float matrix 1
  uint8_t device_quarter_C2_MAT4 = EEPROMdataRestored[9];
  uint16_t data_address1_C2_MAT4 = EEPROMdataRestored[10];
  uint16_t pagesToRead_MAT4 = EEPROMdataRestored[11];
  Serial.println("Matrix 3");
  Serial.print(device_quarter_C2_MAT4);
  Serial.print("  ");
  Serial.print(data_address1_C2_MAT4);
  Serial.print("  ");
  Serial.print(pagesToRead_MAT4);
  Serial.println();
  // float matrix 2
  uint8_t device_quarter_C2_MAT5 = EEPROMdataRestored[12];
  uint16_t data_address1_C2_MAT5 = EEPROMdataRestored[13];
  uint16_t pagesToRead_MAT5 = EEPROMdataRestored[14];
  Serial.println("Matrix 4");
  Serial.print(device_quarter_C2_MAT5);
  Serial.print("  ");
  Serial.print(data_address1_C2_MAT5);
  Serial.print("  ");
  Serial.print(pagesToRead_MAT5);
  Serial.println();
  //
  //read the matrizes from the external EEPROM board Chip 2
  readIntegerMatrixFromEEPROM(integerMatrix1, device_quarter_C2_MAT1, data_address1_C2_MAT1, pagesToRead_MAT1);
  Serial.println("transitionCounts1");
  rSMatI(integerMatrix1);
  readIntegerMatrixFromEEPROM(integerMatrix2, device_quarter_C2_MAT2, data_address1_C2_MAT2, pagesToRead_MAT2);
  Serial.println("transitionCounts2");
  rSMatI(integerMatrix2);
  readFloatVectorFromEEPROM(floatVector1, device_quarter_C2_MAT3, data_address1_C2_MAT3, pagesToRead_MAT3);
  Serial.println("value");
  rVecF(floatVector1);
  readFloatMatrixFromEEPROM(floatMatrix1, device_quarter_C2_MAT4, data_address1_C2_MAT4, pagesToRead_MAT4);
  Serial.println("rewardAmounts");
  rSMatF(floatMatrix1);
  readIntegerMatrixFromEEPROM(integerMatrix3, device_quarter_C2_MAT5, data_address1_C2_MAT5, pagesToRead_MAT5);
  Serial.println("rewardCounts");
  rSMatI(integerMatrix3); 

  numFailures = EEPROMdataRestored[15];
  Serial.println();
  Serial.println("numFailures last Trial");
  Serial.println(numFailures);
}
///////////////////////////////////////////////////////////////////
//This functions saves the RL matrizes to the external EEPROM board.
//The matrizes are stored as non zero entries of the matrix in the form of
// row, col, value
//Thereby, the number of rows and colums is discovered automatically.
//The storage system works with a starting quarter and a starting page
//to restore the matrix it has to be read out up to the next starting page.
//////////////////////////////////////////////////////////////////
void storeRLMatrizesToEEPROM(SparseMatrix<int> &integerMatrix1, SparseMatrix<int> &integerMatrix2, VectorXf &floatVector1, SparseMatrix<float> &floatMatrix1, SparseMatrix<int> &integerMatrix3, uint16_t numFailures) {
  //
  //the first page is set to zero at the zeros quarter
  device_quarter_C2 = 0;
  data_address1_C2 = 0;
  Serial.print("device_quarter_C2 = ");
  Serial.println(device_quarter_C2);
  Serial.print("data_address1_C2 = ");
  Serial.println(data_address1_C2);

  Serial.println("Matrices to store:");
  Serial.println();
  Serial.println("transitionCounts1");
  rSMatI(integerMatrix1);
  Serial.println();
  Serial.println("transitionCounts2");
  rSMatI(integerMatrix2);
  Serial.println();
  Serial.println("value");
  rVecF(floatVector1);
  Serial.println();
  Serial.println("rewardAmounts");
  rSMatF(floatMatrix1);
  Serial.println();
  Serial.println("rewardCounts");
  rSMatI(integerMatrix3); 
  
  
  //
  //save first matrix transitionCount1
  uint8_t device_quarter_C2_MAT1 = device_quarter_C2;                             //save initial quarter
  uint16_t data_address1_C2_MAT1 = data_address1_C2;                              //save initial page
  //store an integer matix
  storeIntegerMatrixToEEPROM(integerMatrix1, device_quarter_C2, data_address1_C2);
  uint16_t pagesToRead_MAT1 = (device_quarter_C2) * 256 + (data_address1_C2);
  //
  //save first matrix transitionCounts2
  uint8_t device_quarter_C2_MAT2 = device_quarter_C2;                             //save initial quarter
  uint16_t data_address1_C2_MAT2 = data_address1_C2;                              //save initial page
  //store an integer matix
  storeIntegerMatrixToEEPROM(integerMatrix2, device_quarter_C2, data_address1_C2);
  uint16_t pagesToRead_MAT2 = (device_quarter_C2 - device_quarter_C2_MAT2) * 256 + (data_address1_C2 - data_address1_C2_MAT2);
  //
  //save first vector value
  uint8_t device_quarter_C2_MAT3 = device_quarter_C2;
  uint16_t data_address1_C2_MAT3 = data_address1_C2;
  storeFloatVectorToEEPROM(floatVector1, device_quarter_C2, data_address1_C2);
  uint16_t pagesToRead_MAT3 = (device_quarter_C2 - device_quarter_C2_MAT3) * 256 + (data_address1_C2 - data_address1_C2_MAT3);
  //
  //save float matrix rewardAmounts
  uint8_t device_quarter_C2_MAT4 = device_quarter_C2;
  uint16_t data_address1_C2_MAT4 = data_address1_C2;
  storeFloatMatrixToEEPROM(floatMatrix1, device_quarter_C2, data_address1_C2);
  uint16_t pagesToRead_MAT4 = (device_quarter_C2 - device_quarter_C2_MAT4) * 256 + (data_address1_C2 - data_address1_C2_MAT4);
  //
  //save integer matrix rewardCounts
  uint8_t device_quarter_C2_MAT5 = device_quarter_C2;
  uint16_t data_address1_C2_MAT5 = data_address1_C2;
  storeIntegerMatrixToEEPROM(integerMatrix3, device_quarter_C2, data_address1_C2);
  uint16_t pagesToRead_MAT5 = (device_quarter_C2 - device_quarter_C2_MAT5) * 256 + (data_address1_C2 - data_address1_C2_MAT5);
  Serial.print("device_quarter_C2_MAT1 = ");
  Serial.println(device_quarter_C2_MAT1);
  Serial.print("data_address1_C2_MAT1 = ");
  Serial.println(data_address1_C2_MAT1);
  Serial.print("pagesToRead = ");
  Serial.println(pagesToRead_MAT1);
  Serial.println();
  Serial.print("device_quarter_C2_MAT2 = ");
  Serial.println(device_quarter_C2_MAT2);
  Serial.print("data_address1_C2_MAT2 = ");
  Serial.println(data_address1_C2_MAT2);
  Serial.print("pagesToRead = ");
  Serial.println(pagesToRead_MAT2);
  Serial.println();
  Serial.print("device_quarter_C2_MAT3 = ");
  Serial.println(device_quarter_C2_MAT3);
  Serial.print("data_address1_C2_MAT3 = ");
  Serial.println(data_address1_C2_MAT3);
  Serial.print("pagesToRead = ");
  Serial.println(pagesToRead_MAT3);
  Serial.println();
  Serial.print("device_quarter_C2_MAT4 = ");
  Serial.println(device_quarter_C2_MAT4);
  Serial.print("data_address1_C2_MAT4 = ");
  Serial.println(data_address1_C2_MAT4);
  Serial.print("pagesToRead = ");
  Serial.println(pagesToRead_MAT4);
  Serial.println();
  Serial.print("device_quarter_C2_MAT5 = ");
  Serial.println(device_quarter_C2_MAT5);
  Serial.print("data_address1_C2_MAT5 = ");
  Serial.println(data_address1_C2_MAT5);
  Serial.print("pagesToRead = ");
  Serial.println(pagesToRead_MAT5);
  //
  //save information about the storage parameters in the Teensys EEPROM
  Serial.println("Save Parameters to EEPROM");
  long EEPROMdata[16];                                  //array consiting of all relevant information
  byte dataByte;
  uint16_t address = 2048 - ((5 * 3 +1) * 4);                //the information is stored at the end of the EEPROM
  EEPROMdata[0] = (long)(device_quarter_C2_MAT1);
  EEPROMdata[1] = (long)(data_address1_C2_MAT1);
  EEPROMdata[2] = (long)(pagesToRead_MAT1);
  EEPROMdata[3] = (long)(device_quarter_C2_MAT2);
  EEPROMdata[4] = (long)(data_address1_C2_MAT2);
  EEPROMdata[5] = (long)(pagesToRead_MAT2);
  EEPROMdata[6] = (long)(device_quarter_C2_MAT3);
  EEPROMdata[7] = (long)(data_address1_C2_MAT3);
  EEPROMdata[8] = (long)(pagesToRead_MAT3);
  EEPROMdata[9] = (long)(device_quarter_C2_MAT4);
  EEPROMdata[10] = (long)(data_address1_C2_MAT4);
  EEPROMdata[11] = (long)(pagesToRead_MAT4);
  EEPROMdata[12] = (long)(device_quarter_C2_MAT5);
  EEPROMdata[13] = (long)(data_address1_C2_MAT5);
  EEPROMdata[14] = (long)(pagesToRead_MAT5);
  EEPROMdata[15] = (long) numFailures;
  //
  //write data on Teensys EEPROM
  for (uint8_t n = 0; n <= 15; ++n) {
    for (uint8_t a = 0; a <= 3; ++a) {
      dataByte = (EEPROMdata[n] >> ((3 - a) * 8)) & 0x000000ff;
      EEPROM.write(address + a, dataByte);
    }
    address = address + 4;
  }
}
///////////////////////////////////////////////////////////////////
//This functions saves the entries of a integer sparse marix
//////////////////////////////////////////////////////////////////
void storeIntegerMatrixToEEPROM(SparseMatrix<int> &MatrixToStore, uint8_t &device_quarter_C2, uint16_t &data_address1_C2) {
  int rows = MatrixToStore.rows();                                //determine number of columns
  int cols = MatrixToStore.cols();                                //determine number of rows
  long MatrixValueBuffer[5][3];                                   //buffer of entries for one page
  uint8_t iBuffer = 0;                                            //counter of the buffer
  //
  //check matrix for non zero entries and save them
  for (uint8_t i = 0; i < rows; i++) {
    for (uint8_t j = 0; j < cols; j++) {
      if (MatrixToStore.coeff(i, j) != 0) {
        MatrixValueBuffer[iBuffer][0] = long(i);
        MatrixValueBuffer[iBuffer][1] = long(j);
        MatrixValueBuffer[iBuffer][2] = long(MatrixToStore.coeff(i, j));
        ++iBuffer;
      }
      if (iBuffer == 5) {
        savePageMatrix(MatrixValueBuffer, device_quarter_C2, data_address1_C2); //saves a page to the EEPROM board
        iBuffer = 0;
      }
    }
  }
  //
  //fill the last page with zero entries
  while (iBuffer < 5) {
    MatrixValueBuffer[iBuffer][0] = long(0);
    MatrixValueBuffer[iBuffer][1] = long(0);
    MatrixValueBuffer[iBuffer][2] = long(0);
    ++iBuffer;
  }
  savePageMatrix(MatrixValueBuffer, device_quarter_C2, data_address1_C2);
}
///////////////////////////////////////////////////////////////////
//This functions saves the entries of a float sparse marix
//////////////////////////////////////////////////////////////////
void storeFloatMatrixToEEPROM(SparseMatrix<float> &MatrixToStore, uint8_t &device_quarter_C2, uint16_t &data_address1_C2) {
  int rows = MatrixToStore.rows();                                //determine number of columns
  int cols = MatrixToStore.cols();                                //determine number of rows
  long MatrixValueBuffer[5][3];                                   //buffer of entries for one page
  uint8_t iBuffer = 0;                                            //counter of the buffer
  //
  //check matrix for non zero entries and save them
  for (uint8_t i = 0; i < rows; i++) {
    for (uint8_t j = 0; j < cols; j++) {
      if (MatrixToStore.coeff(i, j) != 0) {
        MatrixValueBuffer[iBuffer][0] = long(i);
        MatrixValueBuffer[iBuffer][1] = long(j);
        MatrixValueBuffer[iBuffer][2] = long(MatrixToStore.coeff(i, j)*10000);
        ++iBuffer;
        Serial.print("row = ");
        Serial.print(MatrixValueBuffer[iBuffer][0]);
        Serial.print(";    col = ");
        Serial.print(MatrixValueBuffer[iBuffer][1]);
        Serial.print(";    value = ");
        Serial.println(MatrixValueBuffer[iBuffer][2]);
      }
      if (iBuffer == 5) {
        savePageMatrix(MatrixValueBuffer, device_quarter_C2, data_address1_C2); //saves a page to the EEPROM board
        iBuffer = 0;
      }
    }
  }
  //
  //fill the last page with zero entries
  while (iBuffer < 5) {
    MatrixValueBuffer[iBuffer][0] = long(0);
    MatrixValueBuffer[iBuffer][1] = long(0);
    MatrixValueBuffer[iBuffer][2] = long(0);
    ++iBuffer;
  }
  savePageMatrix(MatrixValueBuffer, device_quarter_C2, data_address1_C2);
}
///////////////////////////////////////////////////////////////////
//This functions saves the entries of a float vector
//////////////////////////////////////////////////////////////////
void storeFloatVectorToEEPROM(VectorXf &VectorToStore, uint8_t &device_quarter_C2, uint16_t &data_address1_C2) {
  int rows = VectorToStore.rows();                               //determine number of columns
  long VectorValueBuffer[7][2];                                  //counter of the buffer
  uint8_t iBuffer = 0;                                           //buffer of entries for one page
  //
  //check matrix for non zero entries and save them
  for (uint16_t i = 0; i < rows; i++) {
    VectorValueBuffer[iBuffer][0] = i;
    VectorValueBuffer[iBuffer][1] = long(VectorToStore[i] * 10000); //saves a page to the EEPROM board
    ++iBuffer;
    if (iBuffer == 7) {
      savePageVector(VectorValueBuffer, device_quarter_C2, data_address1_C2);
      iBuffer = 0;
    }
  }
  //
  //fill the last page with zero entries
  while (iBuffer < 7) {
    VectorValueBuffer[iBuffer][0] = long(0);
    VectorValueBuffer[iBuffer][1] = long(0);
    ++iBuffer;
  }
  savePageVector(VectorValueBuffer, device_quarter_C2, data_address1_C2);
}
///////////////////////////////////////////////////////////////////
//This functions saves each page of an integer matrix to the EEPROM board
//////////////////////////////////////////////////////////////////
void savePageMatrix(long (&MatrixValueBuffer)[5][3], uint8_t &device_quarter_C2, uint16_t &data_address1_C2) {
  int data[60];
  uint8_t addressByte;
  uint8_t iArray = 0;
  addressByte = 0;
  //
  //Following block converts the data of the buffer from long integer to byte format
  while (iArray < 5) {
    for (uint8_t n = 0; n < 3; n++) {
      for (uint8_t a = 0; a <= 3; a++) {
        data[addressByte + a] = (MatrixValueBuffer[iArray][n] >> ((3 - a) * 8)) & 0x000000ff;
      }
      addressByte = addressByte + 4;                                                //reset addressByte for next long int
    }
    ++iArray;                                                                       //raise row of dataArray to convert next row
  }
  iArray = 0;
  //
  //warning if chip is full
  if (device_quarter_C2 >= 4) {
    Serial.println("EEPROM chip full");
  }
  //else write page
  else {
    uint8_t writeAddress = (M24M02DRC_2_DATA_ADDRESS | device_quarter_C2);             //set address of quarter to write to. The second EEPROM chip is used
    Wire.beginTransmission(writeAddress);                                              //establish I2C connection; Set quarter address
    Wire.write(data_address1_C2);                                                      //set page address
    Wire.write(data_address2_C2);                                                      //set byte address
    //write all bytes of data into EEPROM
    for (uint8_t i = 0; i < 60; i++) {
      Wire.write(data[i]);
    }
    delay(10);                                                                      //delay of 10ms per write operation
    Wire.endTransmission();                                                         //end transmission to EEPROM
    Serial.print("Page ");
    Serial.print(data_address1_C2);
    Serial.print(" in quarter ");
    Serial.print(device_quarter_C2);
    Serial.println(" of chip 2 is written");
    data_address2_C2 = 0;                                                              //reset byte address
    ++data_address1_C2;                                                                //raise page number
    if (data_address1_C2 == 256) {
      ++device_quarter_C2;                                                             //if last page in quarter is reached, change quarter
      data_address1_C2 = 0;                                                            //reset page in new quarter
    }
  }
}
///////////////////////////////////////////////////////////////////
//This functions saves each page of a float vector to the EEPROM board
//////////////////////////////////////////////////////////////////
void savePageVector(long (&VectorValueBuffer)[7][2], uint8_t &device_quarter_C2, uint16_t &data_address1_C2) {
  int data[60];
  uint8_t addressByte;
  uint8_t iArray = 0;
  addressByte = 0;
  //
  //Following block converts the data of the buffer from long integer to byte format
  while (iArray < 7) {
    for (uint8_t n = 0; n < 2; n++) {
      for (uint8_t a = 0; a <= 3; a++) {
        data[addressByte + a] = (VectorValueBuffer[iArray][n] >> ((3 - a) * 8)) & 0x000000ff;
      }
      addressByte = addressByte + 4;                                                //reset addressByte for next long int
    }
    ++iArray;                                                                       //raise row of dataArray to convert next row
  }
  iArray = 0;
  //
  //warning if chip is full
  if (device_quarter_C2 >= 4) {
    Serial.println("EEPROM chip full");
  }
  //else write page
  else {
    uint8_t writeAddress = (M24M02DRC_2_DATA_ADDRESS | device_quarter_C2);             //set address of quarter to write to. The second EEPROM chip is used
    Wire.beginTransmission(writeAddress);                                              //establish I2C connection; Set quarter address
    Wire.write(data_address1_C2);                                                      //set page address
    Wire.write(data_address2_C2);                                                      //set byte address
    //write all bytes of data into EEPROM
    for (uint8_t i = 0; i < 60; i++) {
      Wire.write(data[i]);
    }
    delay(10);                                                                      //delay of 10ms per write operation
    Wire.endTransmission();                                                         //end transmission to EEPROM
    Serial.print("Page ");
    Serial.print(data_address1_C2);
    Serial.print(" in quarter ");
    Serial.print(device_quarter_C2);
    Serial.println(" of chip 2 is written");
    data_address2_C2 = 0;                                                              //reset byte address
    ++data_address1_C2;                                                                //raise page number
    if (data_address1_C2 == 256) {
      ++device_quarter_C2;                                                             //if last page in quarter is reached, change quarter
      data_address1_C2 = 0;                                                            //reset page in new quarter
    }
  }
}
///////////////////////////////////////////////////////////////////
//This function reads an integer matrix from the EEPROm
//////////////////////////////////////////////////////////////////
void readIntegerMatrixFromEEPROM(SparseMatrix<int> &MatrixToRestore, const uint8_t &device_quarter_C2, const uint16_t &data_address1_C2, uint16_t pagesToRead) {
  uint8_t device_quarter_read = device_quarter_C2;
  uint16_t data_address1_read = data_address1_C2;
  int MatrixValueBuffer[15];
  uint8_t data[60];
  uint8_t dataByte = 0;
  long dataValue;
  uint8_t addressByte = 0;
  //reading of data from EEPROM
  for (uint16_t page = 0; page < pagesToRead; page++) {             //pageToRead restored from Teensy EEPROM
    //
    //Changes quarter when page limit is reached
    if (data_address1_read == 256) {
      device_quarter_read = device_quarter_read + 1;
      data_address1_read = 0;
    }
    //
    // Establish I2C connection and read page from EEPROM
    delay(10);
    Wire.beginTransmission((M24M02DRC_2_DATA_ADDRESS | device_quarter_read));   //Initialize the Tx buffer
    Wire.write(data_address1_read);                                             //Put slave register address (data_address1_C2) in Tx buffer
    Wire.write(0);                                                              //Put slave register address (data_address2_C2) in Tx buffer
    Wire.endTransmission(I2C_NOSTOP);                                           //Send the Tx buffer, but send a restart to keep connection alive
    uint8_t readAddress = (0x54 | 0);                                           //Set current quarter address
    Wire.requestFrom(readAddress, (size_t) 60);                                 //Read the next bytesToRead bytes from EEPROM starting oat data_address1_C2, data_address2_C2
    while (Wire.available()) {
      data[dataByte++] = Wire.read();                                           //save bytes to data[i]
    }
    //
    // Convert bytes back to long int value
    for (uint8_t i = 0; i < 15; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        dataValue += data[addressByte + j];
        dataValue = dataValue << 8;
      }
      dataValue += data[addressByte + 3];
      addressByte = addressByte + 4;                                            //Set address_byte to next value
      //
      MatrixValueBuffer[i] = (dataValue);                                       //save converted long int value in the matrix buffer
      dataValue = 0;
    }
    //
    //insert entry into the matrix
    for (uint8_t i = 0; i < 5; i++) {
      int row = MatrixValueBuffer[i * 3];
      int col = MatrixValueBuffer[i * 3 + 1];
      int value = MatrixValueBuffer[i * 3 + 2];
      MatrixToRestore.insert(row, col) = value;
    }
    MatrixToRestore.makeCompressed();
    addressByte = 0;                                                            //reset addressByte for the next page
    dataByte = 0;                                                               //reset dataByte for the next page
    ++data_address1_read;                                                       //raise page number
  }
}
///////////////////////////////////////////////////////////////////
//This function reads an integer matrix from the EEPROm
//////////////////////////////////////////////////////////////////
void readFloatMatrixFromEEPROM(SparseMatrix<float> &MatrixToRestore, const uint8_t &device_quarter_C2, const uint16_t &data_address1_C2, uint16_t pagesToRead) {
  uint8_t device_quarter_read = device_quarter_C2;
  uint16_t data_address1_read = data_address1_C2;
  long MatrixValueBuffer[15];
  uint8_t data[60];
  uint8_t dataByte = 0;
  long dataValue;
  uint8_t addressByte = 0;
  //reading of data from EEPROM
  for (uint16_t page = 0; page < pagesToRead; page++) {             //pageToRead restored from Teensy EEPROM
    //
    //Changes quarter when page limit is reached
    if (data_address1_read == 256) {
      device_quarter_read = device_quarter_read + 1;
      data_address1_read = 0;
    }
    //
    // Establish I2C connection and read page from EEPROM
    delay(10);
    Wire.beginTransmission((M24M02DRC_2_DATA_ADDRESS | device_quarter_read));   //Initialize the Tx buffer
    Wire.write(data_address1_read);                                             //Put slave register address (data_address1_C2) in Tx buffer
    Wire.write(0);                                                              //Put slave register address (data_address2_C2) in Tx buffer
    Wire.endTransmission(I2C_NOSTOP);                                           //Send the Tx buffer, but send a restart to keep connection alive
    uint8_t readAddress = (0x54 | 0);                                           //Set current quarter address
    Wire.requestFrom(readAddress, (size_t) 60);                                 //Read the next bytesToRead bytes from EEPROM starting oat data_address1_C2, data_address2_C2
    while (Wire.available()) {
      data[dataByte++] = Wire.read();                                           //save bytes to data[i]
    }
    //
    // Convert bytes back to long int value
    for (uint8_t i = 0; i < 15; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        dataValue += data[addressByte + j];
        dataValue = dataValue << 8;
      }
      dataValue += data[addressByte + 3];
      addressByte = addressByte + 4;                                            //Set address_byte to next value
      //
      MatrixValueBuffer[i] = (dataValue);                                       //save converted long int value in the matrix buffer
      dataValue = 0;
    }
    //
    //insert entry into the matrix
    for (uint8_t i = 0; i < 5; i++) {
      int row = (int) MatrixValueBuffer[i * 3];
      int col = (int) MatrixValueBuffer[i * 3 + 1];
      float value = (float)MatrixValueBuffer[i * 3 + 2]/10000;
      MatrixToRestore.insert(row, col) = value;
    }
    MatrixToRestore.makeCompressed();
    addressByte = 0;                                                            //reset addressByte for the next page
    dataByte = 0;                                                               //reset dataByte for the next page
    ++data_address1_read;                                                       //raise page number
  }
}
///////////////////////////////////////////////////////////////////
//This function reads an float vector from the EEPROM
//////////////////////////////////////////////////////////////////
void readFloatVectorFromEEPROM(VectorXf &VectorToRestore, const uint8_t &device_quarter_C2, const uint16_t &data_address1_C2, uint16_t pagesToRead) {
  uint8_t device_quarter_read = device_quarter_C2;
  uint16_t data_address1_read = data_address1_C2;
  float VectorValueBuffer[7];
  uint8_t data[60];
  uint8_t dataByte = 0;
  long dataValue;
  uint8_t addressByte = 0;
  //reading of data from EEPROM
  for (uint16_t page = 0; page < pagesToRead; page++) {             //pageToRead restored from Teensy EEPROM
    //
    //Changes quarter when page limit is reached
    if (data_address1_read == 256) {
      device_quarter_read = device_quarter_read + 1;
      data_address1_read = 0;
    }
    //
    // Establish I2C connection and read page from EEPROM
    delay(10);
    Wire.beginTransmission((M24M02DRC_2_DATA_ADDRESS | device_quarter_read));   //Initialize the Tx buffer
    Wire.write(data_address1_read);                                             //Put slave register address (data_address1_C2) in Tx buffer
    Wire.write(0);                                                              //Put slave register address (data_address2_C2) in Tx buffer
    Wire.endTransmission(I2C_NOSTOP);                                           //Send the Tx buffer, but send a restart to keep connection alive
    uint8_t readAddress = (0x54 | 0);                                           //Set current quarter address
    Wire.requestFrom(readAddress, (size_t) 60);                                 //Read the next bytesToRead bytes from EEPROM starting oat data_address1_C2, data_address2_C2
    while (Wire.available()) {
      data[dataByte++] = Wire.read();                                           //save bytes to data[i]
    }
    //
    // Convert bytes back to long int value
    for (uint8_t i = 0; i < 14; i++) {
      for (uint8_t j = 0; j < 3; j++) {
        dataValue += data[addressByte + j];
        dataValue = dataValue << 8;
      }
      dataValue += data[addressByte + 3];
      addressByte = addressByte + 4;                                            //Set address_byte to next value
      //
      VectorValueBuffer[i] = (dataValue);                                       //save converted long int value in the matrix buffer
      dataValue = 0;
    }
    for (uint8_t i = 0; i < 7; i++) {
      int row = VectorValueBuffer[i * 2];
      float value = (float)VectorValueBuffer[i * 2 + 1] / 10000;
      VectorToRestore.coeffRef(row) = value;
    }
    addressByte = 0;                                                            //reset addressByte for the next page
    dataByte = 0;                                                               //reset dataByte for the next page
    ++data_address1_read;                                                       //raise page number
  }
}
///////////////////////////////////////////////////////////////////
//This function prints a sparse matrix of type integer to the
//command window
//////////////////////////////////////////////////////////////////
void rSMatI(SparseMatrix<int> name) {                   // Auslesen einer spaerlichen Matrix vom Typ double
  int rows = name.rows();                               // Bestimmen der Zeilenanzahl
  int cols = name.cols();                               // Bestimmen der Reihenanzahl
  uint8_t i = 0;
  while (i < rows) {                                    // Solange i kleiner als Reihenanzahl
    for (uint8_t j = 0; j < cols; j++) {                        // Zaehle die Spalten einer Reihe hoch
      Serial.print(name.coeff(i, j));                   // und gib Eintrag (i,j) der Matrix aus
      Serial.print("  ");
    }
    i++;                                                // Zahle die Reihe hoch
    Serial.println();
  }
}
///////////////////////////////////////////////////////////////////
//This function prints a sparse matrix of type float to the
//command window
//////////////////////////////////////////////////////////////////
void rSMatF(SparseMatrix<float> name) {                   // Auslesen einer spaerlichen Matrix vom Typ double
  int rows = name.rows();                               // Bestimmen der Zeilenanzahl
  int cols = name.cols();                               // Bestimmen der Reihenanzahl
  uint8_t i = 0;
  while (i < rows) {                                    // Solange i kleiner als Reihenanzahl
    for (uint8_t j = 0; j < cols; j++) {                        // Zaehle die Spalten einer Reihe hoch
      Serial.print(name.coeff(i, j));                   // und gib Eintrag (i,j) der Matrix aus
      Serial.print("  ");
    }
    i++;                                                // Zahle die Reihe hoch
    Serial.println();
  }
}
///////////////////////////////////////////////////////////////////
//This function prints a vector of type float to the
//command window
//////////////////////////////////////////////////////////////////
void rVecF(VectorXf name) {
  int rows = name.size();
  int j = 0;
  while (j < rows) {
    Serial.println(name(j), 4);
    j++;
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Function for battery check
///////////////////////////////////////////////////////////////////
//This function meassures the voltage of the LiPo battery.
//If voltage drops below 3.4V a warning is given.
//For meauring the voltage, Pin 14 is uses as batteryPin.
//////////////////////////////////////////////////////////////////
uint8_t voltageBattery() {
  Serial.print("Battery voltage = ");
  Serial.println(vBat);
  uint16_t sensorValueBat;                        //measured value of analogous signal (0-1023)
  double vCorrectionFactor = 4.00 / 3.90;         //correction factor (3.64V/3.57V)
  double vBatMeasured = 0;                        //calculated voltage of measurement (R1=32.95kOhm, R2=33.07kOhm)
  double vBatSum = 0;                             //Sum of measurments for determing average
  uint8_t Vcrit = 0;
  //
  //store last values
  vBat_old = vBat;
  //average voltage of batteryPin of 5 measurments
  for (uint8_t i = 0; i < 5; i++) {
    sensorValueBat = (double) analogRead(batteryPin);
    vBatMeasured = (3.3 / 1023) * sensorValueBat * 2;
    vBatSum = vBatSum + vBatMeasured;
    vBat = (vBatSum / 5)  * vCorrectionFactor;
  }
  //
  //if battery voltage of two consecutive measurements is below 3.4 stop program
  if (vBat < 3.40 && vBat_old < 3.35) {
    Serial.println("Warning Battery Voltage To Tow");
    ESP8266DeepSleep();
    LEDBlink(5, 200, 800);
    //
    //if diving cell is operation, emerge
    if (programParameter[0] == 1 && programParameter[1] != 1 || programParameter[5] == 1) {
      emerge(angle, stepCounter);
    }
    LEDBlink(5, 200, 800);
    Vcrit = 1;
  }
  return Vcrit;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Functions of LEDs
///////////////////////////////////////////////////////////////////
//Following functions control the LEDs to visualize the input of
//user commands on the Teensy
///////////////////////////////////////////////////////////////////
void allLEDLow() {
  digitalWrite(ACTIVE_LED, LOW);
  digitalWrite(RL_LED, LOW);
  digitalWrite(SMC_LED, LOW);
  digitalWrite(DATA_LED, LOW);
}
void allLEDHigh() {
  digitalWrite(ACTIVE_LED, HIGH);
  digitalWrite(RL_LED, HIGH);
  digitalWrite(SMC_LED, HIGH);
  digitalWrite(DATA_LED, HIGH);
}
void LEDBlink(uint8_t number, uint16_t duration, uint16_t delaytime) {
  for (uint8_t i = 0; i <= number; i++) {
    digitalWrite(ACTIVE_LED, HIGH);
    digitalWrite(RL_LED, HIGH);
    digitalWrite(SMC_LED, HIGH);
    digitalWrite(DATA_LED, HIGH);
    delay(duration);
    digitalWrite(ACTIVE_LED, LOW);
    digitalWrite(RL_LED, LOW);
    digitalWrite(SMC_LED, LOW);
    digitalWrite(DATA_LED, LOW);
    delay(delaytime - duration);
  }
}
void SMC_LEBBlink(uint16_t Delay) {
  digitalWrite(ACTIVE_LED, HIGH);
  digitalWrite(SMC_LED, HIGH);
  delay(200);
  digitalWrite(ACTIVE_LED, LOW);
  digitalWrite(SMC_LED, LOW);
  delay(Delay - 200);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////// Functions of the pressure sensor
///////////////////////////////////////////////////////////////////
//This function is for calibration of the pressure sensor.
//The calibration coefficients are read from the sensor.
//////////////////////////////////////////////////////////////////
void PressureSensorCalibration() {
  // Declarations
  unsigned char n_crc;
  SPI.beginTransaction(settings_PS);
  cmd_reset();                                        //reset the module after powerup
  for (int i = 0; i < 8; ++i) {                       //read calibration coefficients
    C[i] = cmd_prom(i);
    // Serial.printf("C[%i] = %i\n", i, C[i]);        //list of calibration coefficients if active
  }
  n_crc = crc4(C);
  SPI.endTransaction();
}

///////////////////////////////////////////////////////////////////
// Declaration of Copyright
// Copyright (c) 2009 MEAS Switzerland
// Edited 2015 Johann Lange
// This C code is for starter reference only. It was based on the
// MEAS Switzerland MS56xx pressure sensor modules and Atmel Atmega644p
// microcontroller code and has been by translated Johann Lange
// to work with Teensy 3.1 microcontroller.
//
//////////////////////////////////////////////////////////////////
double calculatePressure() {
  // Declarations
  unsigned long D1;             //ADC value of the pressure conversion
  unsigned long D2;             //ADC value of the temperature conversion
  double dT;                    //difference between actual and measured temperature
  double OFF;                   //offset at actual temperature
  double SENS;                  //sensitivity at actual temperature
  double T2;                    //compensated pressure value, 2nd order
  double OFF2;                  //compensated pressure value, 2nd order
  double SENS2;                 //compensated pressure value, 2nd order
  SPI.beginTransaction(settings_PS);
  // delay required in µs: OSR_4096: 9100, OSR_2048: 4600, OSR_1024: 2300, OSR_512: 1200, OSR_256: 700
  D1 = cmd_adc(CMD_ADC_D1 + CMD_ADC_256, 700);      // read uncompensated pressure, Conversation Command + OSR, delay in µs
  D2 = cmd_adc(CMD_ADC_D2 + CMD_ADC_256, 700);      // read uncompensated temperature, Conversation Command + OSR, delay in µs
  //
  // calcualte 1st order temperature (MS5803_01b 1st order algorithm), base for 2nd order temperature and pressure
  dT = D2 - C[5] * pow(2, 8);                       //Serial.print("dT = "); Serial.println(dT);
  OFF = C[2] * pow(2, 16) + dT * C[4] / pow(2, 7);  //Serial.print("OFF = "); Serial.println(OFF);
  SENS = C[1] * pow(2, 15) + dT * C[3] / pow(2, 8); //Serial.print("SENS = "); Serial.println(SENS);
  //
  T = (2000 + (dT * C[6]) / pow(2, 23)) / 100;
  P = (((D1 * SENS) / pow(2, 21) - OFF) / pow(2, 15)) / 100;
  //
  // calcualte 2nd order pressure and temperature (MS5803_01b 2nd order algorithm)
  if (T > 20) {
    T2 = 0;
    OFF2 = 0;
    SENS2 = 0;
    //
    if (T > 45) {
      SENS2 -= pow(T - 4500, 2) / pow(2, 3);
    }
  }
  else {
    T2 = pow(dT, 2) / pow(2, 31);
    OFF2 = 3 * pow(100 * T - 2000, 2);
    SENS2 = 7 * pow(100 * T - 2000, 2) / pow(2, 3);
    //
    if (T < 15) {
      SENS2 += 2 * pow(100 * T + 1500, 2);
    }
  }
  //
  // Recalculate T, OFF, SENS based on T2, OFF2, SENS2
  T -= T2;
  OFF -= OFF2;
  SENS -= SENS2;
  //
  P = (((D1 * SENS) / pow(2, 21) - OFF) / pow(2, 15)) / 100;
  return P;
  SPI.endTransaction();
}
//
void cmd_reset(void) {
  digitalWrite(CS_Pin_PS, LOW);         // pull CSB low to start the command
  SPI.transfer(CMD_RESET);              // send reset sequence
  delay(3);                             // wait for the reset sequence timing
  digitalWrite(CS_Pin_PS, HIGH);        // pull CSB high to finish the command
}
//
//brief preform adc conversion
//return 24bit result
unsigned long cmd_adc(char cmd, int delaytime) {
  digitalWrite(CS_Pin_PS, LOW);
  unsigned long ret;
  unsigned long temp = 0;
  SPI.transfer(CMD_ADC_CONV + cmd);     // send conversion command;
  cmd = SPI.transfer(0x00);
  delayMicroseconds(delaytime);         // delay required in µs: OSR_4096: 9100, OSR_2048: 4600, OSR_1024: 2300, OSR_512: 1200, OSR_256: 700
  digitalWrite(CS_Pin_PS, HIGH);        // pull CSB high to finish the conversion
  digitalWrite(CS_Pin_PS, LOW);         // pull CSB low to start new command
  SPI.transfer(CMD_ADC_READ);           // send ADC read command
  ret = SPI.transfer(0x00);             // send 0 to read 1st byte (MSB)
  temp = 65536 * ret;
  ret = SPI.transfer(0x00);             // send 0 to read 2nd byte
  temp = temp + 256 * ret;
  ret = SPI.transfer(0x00);             // send 0 to read 3rd byte (LSB)
  temp = temp + ret;
  digitalWrite(CS_Pin_PS, HIGH);        // pull CSB high to finish the read command
  return temp;
}

//brief Read calibration coefficients
//return coefficient
unsigned int cmd_prom(char coef_num) {
  unsigned int ret;
  unsigned int rC = 0;

  digitalWrite(CS_Pin_PS, LOW);               // pull CSB low
  SPI.transfer(CMD_PROM_RD + coef_num * 2);   // send PROM READ command
  ret = SPI.transfer(0x00);                   // send 0 to read the MSB
  rC = 256 * ret;
  ret = SPI.transfer(0x00);                   // send 0 to read the LSB
  rC = rC + ret;
  digitalWrite(CS_Pin_PS, HIGH);              // pull CSB high
  return rC;
}

//brief calculate the CRC code for details look into CRC CODE NOTES
//return crc code
unsigned char crc4(unsigned int n_prom[]) {
  int cnt;                                  // simple counter
  unsigned int n_rem;                       // crc reminder
  unsigned int crc_read;                    // original value of the crc
  unsigned char n_bit;
  n_rem = 0x00;
  crc_read = n_prom[7];                     // save read CRC
  n_prom[7] = (0xFF00 & (n_prom[7]));       // CRC byte is replaced by 0
  for (cnt = 0; cnt < 16; cnt++)            // operation is performed on bytes
  { // choose LSB or MSB
    if (cnt % 2 == 1) n_rem ^= (unsigned short) ((n_prom[cnt >> 1]) & 0x00FF);
    else n_rem ^= (unsigned short) (n_prom[cnt >> 1] >> 8);
    for (n_bit = 8; n_bit > 0; n_bit--)
    {
      if (n_rem & (0x8000))
      {
        n_rem = (n_rem << 1) ^ 0x3000;
      }
      else
      {
        n_rem = (n_rem << 1);
      }
    }
  }
  n_rem = (0x000F & (n_rem >> 12));       // final 4-bit reminder is CRC code
  n_prom[7] = crc_read;                   // restore the crc_read to its original place
  return (n_rem ^ 0x00);
}
