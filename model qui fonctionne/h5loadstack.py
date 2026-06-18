import h5py
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

filename = '18QC025-PD.h5'

# Open file
with h5py.File(filename, 'r') as f:
    # Get all station groups
    vector_data = f['/results/vectorData']
    station_groups = list(vector_data.keys())
    nStations = len(station_groups)
    nSamples = 400

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
        try: 
            Vstack_value[:, k] = V['value'][:]
            Vstack_error[:, k] = V['error'][:]
            Istack_value[:, k] = I['value'][:]
            Istack_error[:, k] = I['error'][:]
        except Exception as e:
            pass
        
        if (k + 1) % 100 == 0:
            print(f'Loaded {k+1} / {nStations} stations...')
        
# enleve les valeurs qui sont égale partout (on veut l'inverse)
mask = (Vstack_value == 0).all(axis=0)
Vstack_value = Vstack_value[:, ~mask] # tild signifie l'inverse
Istack_value = Istack_value[:, ~mask]
Vstack_error = Vstack_error[:, ~mask] # tild signifie l'inverse
Istack_error = Istack_error[:, ~mask]
stationIDs = stationIDs[~mask]

print(f'Done! Arrays are {nSamples} x {nStations} (samples x stations)')

# Sort by station ID
sortIdx = np.argsort(stationIDs)
stationIDs_sorted = stationIDs[sortIdx]

Vstack_value = Vstack_value[:, sortIdx]
Istack_value = Istack_value[:, sortIdx]
Vstack_error = Vstack_error[:, sortIdx]
Istack_error = Istack_error[:, sortIdx]

# Plot station 1000 (Python index 999 if matching MATLAB 1000)
k = len(stationIDs_sorted) - 1  # MATLAB 1000 → Python 999

# verify if the curve is generally decreasing 
# do this by splitting the curve into 10 segments and checking if the mean of each segment is decreasing
#Vstack_value_neg = []
#for i in Vstack_value: 
#    i = np.array_split(i, 10)

plt.figure(1)
plt.clf()
plt.plot(Vstack_value[:,k])
# plt.title(f'Vstack - Station {75}')
plt.xlabel('Sample (100 Hz)')
plt.ylabel('Potential (mV)')
plt.pause(0.5)

plt.show()


names = np.arange(0, len(stationIDs_sorted))
df = pd.DataFrame(Vstack_value)
df.to_csv("test_18QC025-PD.csv", index=False, header=names)