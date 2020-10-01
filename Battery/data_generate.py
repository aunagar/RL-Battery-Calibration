import math
import matplotlib.pyplot as plt
import h5py

import numpy as np
from BatteryCalib import Battery
from tqdm import tqdm

SEED = 20
np.random.seed(SEED)

LOAD_MIN = 14
LOAD_MAX = 16
load_breaks = 3

qMOBILE_MIN = 4000
qMOBILE_MAX = 7000
q_breaks = 101

Ro_min = 0.1
Ro_max = 0.2
r_breaks = 101
def generate_data():
	b = Battery()
	X = []
	U = []
	Z = []
	theta = []
	num_traj = 0
	for l in range(load_breaks):
		load = LOAD_MIN + l*(LOAD_MAX - LOAD_MIN)/(load_breaks-1)
		for q in range(q_breaks):
			qMobile = qMOBILE_MIN + q*(qMOBILE_MAX - qMOBILE_MIN)/(q_breaks-1)
			for r in range(r_breaks):
				num_traj += 1
				Ro = Ro_min + r*(Ro_max - Ro_min)/(r_breaks-1)
				print("load is ", load, " and qMobile is ", qMobile, " and Ro is ", Ro)
				b.reset()
				b.applyDegradation(qMobile = qMobile, Ro = Ro)

				Ti, Xi, Ui, Zi = b.simulateToThreshold(default_load = load)

				# t = np.array([qMobile]*len(Ti))
				t = np.array([[qMobile,Ro]]*len(Ti))
				# print("Initial Xi: ", Xi[:,0])
				# print("Xi", Xi.shape)
				# print("Ui", Ui.shape)
				# print("Zi", Zi.shape)

				l, h = int(0.05*Xi.shape[1]), int(0.95*Xi.shape[1])
				print("traj length = ",h-l)
				X.append(Xi[:,l:h].T)
				U.append(Ui[:,l:h].T)
				Z.append(Zi[:,l:h].T)
				theta.append(t[l:h])
	# for i in tqdm(range(num_trajectories+1)):
	# 	b.reset()
	# 	if method == "random":
	# 		load = np.random.uniform(low = LOAD_MIN, high = LOAD_MAX)
	# 		qMobile = np.random.uniform(low = qMOBILE_MIN, high = qMOBILE_MAX)
	# 		Ro = np.random.uniform(low = Ro_min, high = Ro_max)
	# 	elif method == "specialized":
	# 		load = LOAD_MIN + i*(LOAD_MAX - LOAD_MIN)/num_trajectories
	# 		qMobile = qMOBILE_MIN + i*(qMOBILE_MAX - qMOBILE_MIN)/num_trajectories
	# 		Ro = Ro_min + i*(Ro_max - Ro_min)/num_trajectories
	# 	elif method == "mixed":
	# 		load = LOAD_MIN + (i%3)*(LOAD_MAX - LOAD_MIN)/2
	# 		qMobile = qMOBILE_MIN + (i//3)*(qMOBILE_MAX - qMOBILE_MIN)
	# 		Ro = Ro_min + (i//3)*(Ro_max - Ro_min)/4


	data = dict()
	data['X'] = X
	data['U'] = U
	data['Z'] = Z
	data['theta'] = theta
 	# data['X'] = np.array(X,dtype=object)
	# data['U'] = np.array(U,dtype=object)
	# data['Z'] = np.array(Z,dtype=object)
	# data['theta'] = np.array(theta,dtype=object)
	print("Total number of trajectories = ", num_traj)
	np.savez("/cluster/scratch/aunagar/data_{}_trajectories_load_{}_{}_q_{}_{}_R_{}_{}_dt_1_short.npz".format(num_traj,
			LOAD_MIN, LOAD_MAX, qMOBILE_MIN, qMOBILE_MAX, Ro_min, Ro_max), 
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
	generate_data()

	# def get_data():
	#     process = 'data_50_trajectories_constant_load_8_uniform_q_7000_7600.npz'
	#     data = np.load(process, allow_pickle = True)
	#     state_orig = data['X']
	#     X_orig = data['Z']
	#     W_orig = data['U']
	#     T_orig = data['theta']

	#     state = [s[:-1,:] for s in state_orig]
	#     X = [x[:-1,:] for x in X_orig]
	#     X_ = [x[1:, :] for x in X_orig]
	#     W = [w[:-1,:] for w in W_orig]
	#     T = [t.reshape(-1,1)[:-1,:] for t in T_orig]

	#     data_stack = [np.concatenate((X[i], W[i], X_[i], T[i], state[i]), axis = -1) for i in range(len(X))]
	#     # print(np.shape(data_stack))
	#     return data_stack

	# stacked_data = get_data()

	# for d in stacked_data:
	# 	plt.plot(list(range(len(d[:,1]))), d[:, 1], color = 'red', linestyle = '-')
	# plt.show()
	# print(stacked_data[0].shape)