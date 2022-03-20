import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pyproj import Proj
from sympy import Symbol

plt.rcParams['figure.figsize'] = [16, 9]
plt.rc('xtick', labelsize=20)
plt.rc('ytick', labelsize=20)

# Kalman Filter for Constant Acceleration Model

# Initial State

x = np.matrix([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]).T
n = x.size  # States

# Initial Uncertainty

P = np.matrix([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 10.0]])

# Dynamic Matrix

dt = 0.02 # Time Step between Filter Steps

A = np.matrix([[1.0, 0.0, dt,  0.0, 1/2.0*dt**2, 0.0        ],
               [0.0, 1.0, 0.0, dt,  0.0,         1/2.0*dt**2],
               [0.0, 0.0, 1.0, 0.0, dt,          0.0        ],
               [0.0, 0.0, 0.0, 1.0, 0.0,         dt         ],
               [0.0, 0.0, 0.0, 0.0, 1.0,         0.0        ],
               [0.0, 0.0, 0.0, 0.0, 0.0,         1.0        ]])

# Measurement Matrix

H = np.matrix([[0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
               [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]])

# Measurement Noise Covariance

ra = 100.0 ** 2
rp = 2.0 ** 2

R = np.matrix([[ra, 0.0, 0.0, 0.0],
               [0.0, ra, 0.0, 0.0],
               [0.0, 0.0, rp, 0.0],
               [0.0, 0.0, 0.0, rp]])

# Process Noise Covariance Matrix Q for CA Model

# Symbolic Calculation

dts = Symbol('\Delta t')

sj = 0.1

Q = np.matrix([[(dt**6)/36,  0,           (dt**5)/12,    0,           (dt**4)/6,  0         ],
               [0,           (dt**6)/36,  0,             (dt**5)/12,  0,          (dt**4)/6 ],
               [(dt**5)/12,  0,           (dt**4)/4,     0,           (dt**3)/2,  0         ],
               [0,           (dt**5)/12,  0,             (dt**4)/4,   0,          (dt**3)/2 ],
               [(dt**4)/6,   0,           (dt**3)/2,     0,           (dt**2),    0         ],
               [0,           (dt**4)/6,   0,             (dt**3)/2,   0,          (dt**2)   ]])  * sj ** 2

# Identity Matrix

I = np.eye(n)

# Measurement

# Read data

# Use every value to get IMU updates in every timestep
n_rows = 10800  # len(df)

cur_dir = os.path.dirname(os.path.abspath(__file__))
datafile = os.path.join(cur_dir, 'data.csv')
df = pd.read_csv(datafile)

# Extract values
ax = df['ax'].dropna()
ay = df['ay'].dropna()
px = df['latitude'].dropna()
py = df['longitude'].dropna()

m = len(df['ax']) # Measurements

# Lat Lon to UTM
utm_converter = Proj("+proj=utm +zone=33U, +south +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

for i in range(0, len(px)):
    py[i], px[i] = utm_converter(py[i], px[i])
    if(i % 5):
        px[i] = px[i - 1]
        py[i] = py[i - 1]
    else:
        px[i] = px[i] + np.random.normal(0, 2.0, 1)
        py[i] = py[i] + np.random.normal(0, 2.0, 1)

# Start from position (0 ,0)
py_offset = py[0]
px_offset = px[0]
px = px-px_offset
py = py-py_offset

# Stack measurement vector
measurements = np.vstack((ax,ay,px,py))

fig = plt.figure(figsize=(16,9))
plt.step(range(m),ax, label='$a_x$')
plt.step(range(m),ay, label='$a_y$')
plt.ylabel('Acceleration in m / $s^2$', fontsize=20)
plt.xlabel('Number of measurements', fontsize=20)
plt.title('IMU Measurements', fontsize=20)
plt.ylim([-20, 20])
plt.legend(loc='best',prop={'size': 18})

fig = plt.figure(figsize=(16,9))
plt.step(px, py, label='$GNSS$')
plt.xlabel('x in m', fontsize=20)
plt.ylabel('y in m', fontsize=20)
plt.title('GNSS Measurements', fontsize=20)
plt.xlim([min(px), max(px)])
plt.ylim([min(py), max(py)])
plt.legend(loc='best',prop={'size': 18})

plt.show()

# Preallocation for Plotting
xt = []
yt = []
dxt= []
dyt= []
ddxt=[]
ddyt=[]
Zx = []
Zy = []
Px = []
Py = []
Pdx= []
Pdy= []
Pddx=[]
Pddy=[]
Kx = []
Ky = []
Kdx= []
Kdy= []
Kddx=[]
Kddy=[]

# Kalman Filter

from task2 import adaptHmatrix

for n in range(0, m):
    
    # Time Update (Prediction)
    # ========================
    # Project the state ahead
    x = A*x
    
    # Project the error covariance ahead
    P = A*P*A.T + Q
    
    
    # Measurement Update (Correction)
    # ===============================
    # Compute the Kalman Gain
    H = adaptHmatrix(measurements[:,n], measurements[:,n-1])

    S = H*P*H.T + R
    K = (P*H.T) * np.linalg.pinv(S)

    
    # Update the estimate via z
    Z = measurements[:,n].reshape(H.shape[0],1)
    y = Z - (H*x)                            # Innovation or Residual
    x = x + (K*y)
    
    # Update the error covariance
    P = (I - (K*H))*P

    # Save states for Plotting
    xt.append(float(x[0]))
    yt.append(float(x[1]))
    dxt.append(float(x[2]))
    dyt.append(float(x[3]))
    ddxt.append(float(x[4]))
    ddyt.append(float(x[5]))
    Zx.append(float(Z[0]))
    Zy.append(float(Z[1]))
    Px.append(float(P[0,0]))
    Py.append(float(P[1,1]))
    Pdx.append(float(P[2,2]))
    Pdy.append(float(P[3,3]))
    Pddx.append(float(P[4,4]))
    Pddy.append(float(P[5,5]))
    Kx.append(float(K[0,0]))
    Ky.append(float(K[1,0]))
    Kdx.append(float(K[2,0]))
    Kdy.append(float(K[3,0]))
    Kddx.append(float(K[4,0]))
    Kddy.append(float(K[5,0]))

# Plots

# State Vector

fig, axs = plt.subplots(3)
fig.suptitle("State")

axs[0].plot(range(len(measurements[0])), ddxt, label = '$\ddot x$')
axs[0].plot(range(len(measurements[0])), ddyt, label = '$\ddot y$')
axs[0].legend(loc='best')
axs[0].set_ylabel("Acceleration in $m/s^2$")

axs[1].plot(range(len(measurements[0])), dxt, label = '$\dot x$')
axs[1].plot(range(len(measurements[0])), dyt, label = '$\dot y$')
axs[1].legend(loc='best')
axs[1].set_ylabel("Velocity in $m/s$")

axs[2].plot(range(len(measurements[0])), xt, label = '$x$')
axs[2].plot(range(len(measurements[0])), yt, label = '$y$')
axs[2].legend(loc='best')
axs[2].set_ylabel("Position in $m$")
axs[2].set_xlabel("Filtersteps")

plt.show()

# Position x/y

fig = plt.figure(figsize=(16,9))

plt.scatter(xt[0], yt[0], s = 100, label = 'Start', c = 'g')
plt.scatter(xt[-1], yt[-1], s=100, label = 'Goal', c = 'r')
plt.plot(xt,yt, label='State', alpha=0.5)
plt.xlabel('x in m', fontsize=20)
plt.ylabel('y in m', fontsize=20)
plt.title('Position', fontsize=20)
plt.legend(loc='best', fontsize=20)
plt.xlim(min(xt), max(xt))
plt.ylim(min(yt), max(yt))

plt.show()
