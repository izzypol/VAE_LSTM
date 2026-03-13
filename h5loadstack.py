import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = 'L44E_Nov02_JCtest.h5'

# Open file
with h5py.File(filename, 'r') as f:
    # Get all station groups
    vector_data = f['/results/vectorData']
    station_groups = list(vector_data.keys())
    nStations = len(station_groups)
    nSamples = 1600

    # Pre-allocate arrays
    Vstack_value = np.zeros((nSamples, nStations))
    Vstack_error = np.zeros((nSamples, nStations))
    Istack_value = np.zeros((nSamples, nStations))
    Istack_error = np.zeros((nSamples, nStations))
    stationIDs   = np.zeros(nStations)

    # Load all stations
    for k, station_name in enumerate(station_groups):
        stationPath = f'/results/vectorData/{station_name}'
        stationIDs[k] = float(station_name)

        # Read datasets
        V = f[f'{stationPath}/Vstack']
        I = f[f'{stationPath}/Istack']

        Vstack_value[:, k] = V['value'][:]
        Vstack_error[:, k] = V['error'][:]
        Istack_value[:, k] = I['value'][:]
        Istack_error[:, k] = I['error'][:]

        if (k + 1) % 200 == 0:
            print(f'Loaded {k+1} / {nStations} stations...')

print(f'Done! Arrays are {nSamples} x {nStations} (samples x stations)')

# Sort by station ID
sortIdx = np.argsort(stationIDs)
stationIDs_sorted = stationIDs[sortIdx]

Vstack_value = Vstack_value[:, sortIdx]
Istack_value = Istack_value[:, sortIdx]

# Plot station 1000 (Python index 999 if matching MATLAB 1000)
#k = len(stationIDs_sorted) - 1  # MATLAB 1000 → Python 999

# verify if the curve is generally decreasing 
# do this by splitting the curve into 10 segments and checking if the mean of each segment is decreasing
#Vstack_value_neg = []
#for i in Vstack_value: 
#    i = np.array_split(i, 10)

"""plt.figure(1)
plt.clf()
plt.plot(Vstack_value[:,k])
plt.title(f'Vstack - Station {k} (ID: {int(stationIDs_sorted[k])})')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.pause(0.5)

plt.show()"""


names = np.arange(0,nStations)
df = pd.DataFrame(Vstack_value)
df.to_csv("test_L44E.csv", index=False, header=names)