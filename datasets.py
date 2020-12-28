import random
import numpy as np

def two_sin(N, rotation=None, scale_noise=0.3):
	if rotation is not None:
		rot = [[ np.cos(rotation), np.sin(rotation) ],
				[-np.sin(rotation), np.cos(rotation)]]

	X0_ = np.transpose([np.linspace(0,4,N), np.sin(np.linspace(0,4,N))]) + np.random.normal(scale=scale_noise, size=(N, 2))+[[0,1]]
	X1_ = np.transpose([np.linspace(0,4,N), np.sin(np.linspace(0,4,N))]) + np.random.normal(scale=scale_noise, size=(N, 2))

	X0 = np.matmul(rot, X0_.T).T
	X1 = np.matmul(rot, X1_.T).T

	y0 = np.zeros(shape=N)
	y1 = np.ones (shape=N)

	X = np.concatenate([X0, X1], axis=0)
	y = np.concatenate([y0, y1], axis=0)

	perm = np.random.permutation(len(y))
	X = X[perm]
	y = y[perm]

	return X, y