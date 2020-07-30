import numpy as np
import h5py
import matplotlib.pyplot as plt

with h5py.File('memory_1.h5', 'r') as hdf:
    state = hdf.get('reward')
    state = np.array(state)
print(len(state))


# with h5py.File('log/CMAPSS/CAC-0-final/CAC_theta.h5', 'r') as hdf:
#     data = hdf.get('Data')
#     theta = np.array(data)
# with h5py.File('log/CMAPSS/CAC-0-final/Normal_theta_ground.h5', 'r') as hdf:
#     data = hdf.get('Data')
#     gt = np.array(data)
#
#
# x = np.linspace(0,np.shape(theta)[0]-1,np.shape(theta)[0])
#
#
#
# plt.plot(x, theta, color='blue', label='Tracking')
# plt.plot(x, gt, color='black',linestyle='--',label='Ground truth')
# plt.show()