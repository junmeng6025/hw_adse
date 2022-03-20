import os
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from scipy.stats import norm
from sympy import Symbol, symbols, Matrix, sin, cos
from sympy import init_printing



# Extended Kalman Filter Implementation for Constant Heading and Velocity (CHCV) Vehicle Model

numstates = 4 # States

vs, psis, dts, xs, ys, lats, lons = symbols('v \psi T x y lat lon')

gs = Matrix([[xs + vs * dts * cos(psis)],
             [ys + vs * dts * sin(psis)],
             [psis],
             [vs]])

state = Matrix([xs,ys,psis,vs])


# Dynamic function g
# This formulas calculate how the state is evolving from one to the next time step

# Calculate the Jacobian of the dynamic function g with respect to the state vector x
# It has to be computed on every filter step because it depends on the current state!
gs.jacobian(state)


# Initial uncertainty P_0

P = np.eye(numstates) * 1000.0

# Load measurements
cur_dir = os.path.dirname(os.path.abspath(__file__))
datafile = os.path.join(cur_dir, 'data.csv')

df = pd.read_csv(datafile)
millis = df['millis'].to_numpy()
course = df['course'].to_numpy()
altitude = df['altitude'].to_numpy()
latitude = df['latitude'].to_numpy()
longitude = df['longitude'].to_numpy()
speed = df['speed'].to_numpy()

print('Read \'%s\' successfully.' % datafile)

# A course of 0° means the Car is traveling north bound
# and 90° means it is traveling east bound.
# In the Calculation following, East is Zero and North is 90°
course = ( -course + 90.0)


# Calculate dt for Measurements

dt = np.hstack([0.02, np.diff(millis)])/1000.0 # s


# Measurement Function h
# 
# Matrix J_H is the Jacobian of the Measurement function h with respect to the state. Function h can be used to compute the predicted measurement from the predicted state.
# If a GPS measurement is available, the following function maps the state to the measurement.


hs = Matrix([[xs],
             [ys]])

JHs=hs.jacobian(state)

# If no GPS measurement is available, simply set the corresponding values in J_h to zero.
# Remember task2 ;)
# Measurement Noise Covariance R

varGPS = 6.0 # Standard Deviation of GPS Measurement
R = np.diag([varGPS ** 2.0, varGPS ** 2.0])


# Identity Matrix

EYE = np.eye(numstates)

# Approx. Lat/Lon to Meters to check Location

RadiusEarth = 6378388.0 # m
arc= 2.0 * np.pi * (RadiusEarth + altitude) / 360.0

dx = arc * np.cos(latitude * np.pi / 180.0) * np.hstack((0.0, np.diff(longitude)))
dy = arc * np.hstack((0.0, np.diff(latitude)))

mx = np.cumsum(dx)
my = np.cumsum(dy)

ds = np.sqrt(dx ** 2 + dy ** 2)

# Write to vector in which filtersteps new measurements arrive
GPS = (ds != 0.0).astype('bool')


# Initial State

x = np.matrix([[mx[0], my[0], 0.5 * np.pi, 0.0]]).T


# Put everything together as a measurement vector

measurements = np.vstack((mx, my))
m = measurements.shape[1]

# Preallocation for Plotting
x0, x1, x2, x3 = [], [], [], []
Zx, Zy = [], []
Px, Py, Pdx, Pdy = [], [], [], []
Kx, Ky, Kdx, Kdy = [], [], [], []

def savestates(x, Z, P, K):
    x0.append(float(x[0]))
    x1.append(float(x[1]))
    x2.append(float(x[2]))
    x3.append(float(x[3]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))    
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))

import EKF_equations
# Extended Kalman Filter
for filterstep in range(0, m):
    
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    # see "Dynamic Matrix"

    x = x.astype(np.float64)
    dt = dt.astype(np.float64)
    x[0] = x[0] + dt[filterstep] * x[3] * np.cos(x[2])
    x[1] = x[1] + dt[filterstep] * x[3] * np.sin(x[2])
    x[2] = (x[2] + np.pi) % (2.0 * np.pi) - np.pi
    x[3] = x[3]

    # Calculate the Jacobian of the Dynamic Matrix A

    a13 = (-dt[filterstep] * x[3] * np.sin(x[2]))[0, 0]
    a14 = (dt[filterstep] * np.cos(x[2]))[0, 0]
    a23 = (dt[filterstep] * x[3] * np.cos(x[2]))[0, 0]
    a24 = (dt[filterstep] * np.sin(x[2]))[0, 0]
    JA = np.matrix([[1.0, 0.0, a13, a14],
                    [0.0, 1.0, a23, a24],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]])
    
    
    # Calculate the Process Noise Covariance Matrix
    sGPS     = 0.5 * 8.8 * dt[filterstep] ** 2  # assume 8.8m/s2 as maximum acceleration
    sCourse  = 2.0 * dt[filterstep] # assume 0.5rad/s as maximum turn rate
    sVelocity= 35.0 * dt[filterstep] # assume 8.8m/s2 as maximum acceleration

    Q = np.diag([sGPS ** 2, sGPS ** 2, sCourse ** 2, sVelocity ** 2])
    

    # Project the error covariance ahead
    P = EKF_equations.ProjectErrorCovariance(JA, P, Q)
    
    # Measurement Update (Correction)
    # ===============================
    # Measurement Function
    hx = np.matrix([[float(x[0])],
                    [float(x[1])]])

    if GPS[filterstep]: # with 10Hz, every 5th step
        JH = np.matrix([[1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0]])
    else: # every other step
        JH = np.matrix([[0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0]])
    
    
    K = EKF_equations.ComputeKalmanGain(JH, P, R)

    # Update the estimate via
    Z = measurements[:, filterstep].reshape(JH.shape[0], 1)
    y = Z - (hx)
    x = x + (K * y)

    # Update the error covariance
    P = EKF_equations.UpdateErrorCovariance(P, EYE, K, JH)

    # Save states for Plotting
    savestates(x, Z, P, K)



# PLOTS

# State Vector

def plotx():

    fig, axs = plt.subplots(3)
    fig.suptitle('Extended Kalman Filter State Estimates (State Vector $x$)')

    axs[0].plot(range(len(measurements[0])), x0-mx[0], label = '$x$')
    axs[0].plot(range(len(measurements[0])), x1-my[0], label = '$y$')
    axs[0].legend(loc='best')
    axs[0].set_ylabel('Position in m')

    axs[1].plot(range(len(measurements[0])), x2, label='$\psi$')
    axs[1].plot(range(len(measurements[0])),
             (course / 180.0 * np.pi + np.pi) % (2.0 * np.pi) - np.pi,
             label = '$\psi$ (from GPS as reference)')
    axs[1].legend(loc='best')
    axs[1].set_ylabel('Course')

    axs[2].plot(range(len(measurements[0])), x3, label = '$v$')
    axs[2].plot(range(len(measurements[0])), speed / 3.6,
             label = '$v$ (from GPS as reference)')
    axs[2].legend(loc='best')
    axs[2].set_ylabel('Velocity')
    axs[2].set_xlabel('Filtersteps')

    plt.show()

plotx()


# Position x/y

def plotxy():

    fig = plt.figure()

    # EKF State
    #plt.quiver(x0, x1, np.cos(x2), np.sin(x2), color = '#94C600',
    #           units = 'xy', width = 0.05, scale = 0.5)
    plt.plot(x0, x1, label = 'EKF Position', c = 'k', lw = 2)

    # Measurements
    plt.scatter(mx[::5], my[::5], s = 50, label = 'GPS Measurements', marker = '+')

    # Start/Goal
    plt.scatter(x0[0], x1[0], s = 60, label = 'Start', c = 'g')
    plt.scatter(x0[-1], x1[-1], s = 60, label = 'Goal', c = 'r')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.title('Position')
    plt.legend(loc = 'best')
    plt.axis('equal')

    plt.show()

plotxy()

