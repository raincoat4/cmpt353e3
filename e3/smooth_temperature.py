import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from pykalman import KalmanFilter

filename1 = sys.argv[1]

cpu_data = pd.read_csv(filename1, sep = ',', header = 1, index_col = False, names = ['timestamp', 'temperature', 'sys_load_1', 'cpu_percent', 'cpu_freq', 'fan_rpm'])

plt.figure(figsize=(12, 4))
plt.plot(cpu_data['timestamp'], cpu_data['temperature'], 'b.', alpha=0.5, label = 'Data Points')

#lowess experimentation
cpu_data['timestamp_numeric'] = pd.to_datetime(cpu_data['timestamp'])
cpu_data['timestamp_numeric'] = cpu_data['timestamp_numeric'].values.astype(float)
loess_smoothed = lowess(cpu_data['temperature'], cpu_data['timestamp_numeric'], frac = 0.01)
plt.plot(cpu_data['timestamp'], loess_smoothed[:,1], 'r-', label = 'LOESS-smoothed line')

#kalman experimentation
kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]
initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([0.9, 0.2, 0.9, 0.9]) ** 2 
transition_covariance = np.diag([0.2, 0.1, 0.2, 0.2]) ** 2 
transition = [[0.97,0.5,0.2,-0.001], [0.1,0.4,2.2,0], [0,0,0.95,0], [0,0,0,1]] 
kf = KalmanFilter(
    initial_state_mean = initial_state,
    initial_state_covariance = observation_covariance,
    observation_covariance = observation_covariance,
    transition_covariance = transition_covariance,
    transition_matrices = transition
)
kalman_smoothed, _ = kf.smooth(kalman_data)
plt.plot(cpu_data['timestamp'], kalman_smoothed[:, 0], 'g-', label = 'Kalman-smoothed line')
plt.legend()
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.xticks([])
plt.savefig('cpu.svg') # for final submission