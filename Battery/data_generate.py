import math
import matplotlib.pyplot as plt
import h5py

import numpy as np
from BatteryCalib import Battery
from tqdm import tqdm

SEED = 20
np.random.seed(SEED)

LOAD_MIN = 8
LOAD_MAX = 8 

qMOBILE_MIN = 7600
qMOBILE_MAX = 7000

def generate_data(num_trajectories, method = 'specialized'):
	b = Battery()
	X = []
	U = []
	Z = []
	theta = []
	for i in tqdm(range(num_trajectories)):
		b.reset()
		if method == "random":
			load = np.random.uniform(low = LOAD_MIN, high = LOAD_MAX)
			qMobile = np.random.uniform(low = qMOBILE_MIN, high = qMOBILE_MAX)
		else:
			load = LOAD_MIN + i*(LOAD_MAX - LOAD_MIN)/num_trajectories
			qMobile = qMOBILE_MIN + i*(qMOBILE_MAX - qMOBILE_MIN)/num_trajectories

		print("load is ", load, " and qMobile is ", qMobile)
		b.applyDegradation(qMobile = qMobile)

		Ti, Xi, Ui, Zi = b.simulateToThreshold(default_load = load)

		t = np.array([qMobile]*len(Ti))
		print("Initial Xi: ", Xi[:,0])
		# print("Ui: ", Ui.shape)
		# print("Zi: ", Zi.shape)

		X.append(Xi.T)
		U.append(Ui.T)
		Z.append(Zi.T)
		theta.append(t)

	data = dict()
	data['X'] = X
	data['U'] = U
	data['Z'] = Z
	data['theta'] = theta
 	# data['X'] = np.array(X,dtype=object)
	# data['U'] = np.array(U,dtype=object)
	# data['Z'] = np.array(Z,dtype=object)
	# data['theta'] = np.array(theta,dtype=object)

	np.savez("data_50_trajectories_constant_load_8_uniform_q_7000_7600.npz", 
		X = np.array(X, dtype = object), U = np.array(U, dtype = object),
		Z = np.array(Z, dtype = object), theta = np.array(theta, dtype = object))
	# f = h5py.File('battery.h5','w')
	# print(len(X))
	# f['X'] = np.array(X)
	# f['U'] = np.array(U, dtype = object)
	# f['Z'] = np.array(Z, dtype = object)
	# f['theta'] = np.array(theta)

if __name__ == '__main__':
	# from multiprocessing import Pool
	
	# pool = Pool(processes = 2)
	# inputs = [2,2]
	# pool.map(generate_data, inputs)
	# generate_data(50)

	def get_data():
	    process = 'data_50_trajectories_constant_load_8_uniform_q_7000_7600.npz'
	    data = np.load(process, allow_pickle = True)
	    state_orig = data['X']
	    X_orig = data['Z']
	    W_orig = data['U']
	    T_orig = data['theta']

	    state = [s[:-1,:] for s in state_orig]
	    X = [x[:-1,:] for x in X_orig]
	    X_ = [x[1:, :] for x in X_orig]
	    W = [w[:-1,:] for w in W_orig]
	    T = [t.reshape(-1,1)[:-1,:] for t in T_orig]

	    data_stack = [np.concatenate((X[i], W[i], X_[i], T[i], state[i]), axis = -1) for i in range(len(X))]
	    # print(np.shape(data_stack))
	    return data_stack

	stacked_data = get_data()

	for d in stacked_data:
		plt.plot(list(range(len(d[:,1]))), d[:, 1], color = 'red', linestyle = '-')
	plt.show()
	# print(stacked_data[0].shape)