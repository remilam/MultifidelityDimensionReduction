import numpy as np
import matplotlib.pyplot as plt
import pandas as pn

def compute_H(M):
	dimension, n_eval = np.shape(M)
	H = np.zeros((dimension,dimension))

	for i in range(0,n_eval):
		H += np.outer(M[:,i], M[:,i])
	H = H/(1.0*n_eval)

	return H

def compute_normE(M):
	normE2 = 0.0
	dimension, n_eval = np.shape(M)
	for i in range(0,n_eval):
		normE2 += pow(np.linalg.norm(M[:,i]), 2.0)
	normE2 = normE2/(1.0*n_eval)

	return normE2

def compute_normE_MF(MHF1, MLF1, MLF2):
	normE2 = 0.0
	dimension, n_eval1 = np.shape(MHF1)
	for i in range(0,n_eval1):
		normE2 += pow(np.linalg.norm(MHF1[:,i]), 2.0) - pow(np.linalg.norm(MLF1[:,i]), 2.0)
	normE2 = normE2/(1.0*n_eval1)

	normE2LF2 = 0.0
	dimension, n_eval2 = np.shape(MLF2)
	for i in range(0,n_eval2):
		normE2LF2 += pow(np.linalg.norm(MLF2[:,i]), 2.0)
	normE2LF2 = normE2LF2/(1.0*n_eval2)


	return normE2 + normE2LF2

def compute_normH(H):
	return np.linalg.norm(H, ord=2)


def compute_Hhat(M, n_sample):
	index = np.random.randint(0,n_eval, n_sample)

	M_resampled =  M[:,index]
	H_hat = compute_H(M_resampled)
	return H_hat, index

def compute_norm1_deviation(dimension, n_eval):
	D = np.random.uniform(-1.0, 1.0, (dimension, n_eval))
	for i in range(0, n_eval):
		D[:,i] = D[:,i] / np.linalg.norm(D[:,i], ord=2)
	return D

def compute_Ghat(M, index, extra_sample, theta2, identic_sample, normE):

	dimension, n_eval = np.shape(M)
	if identic_sample:
		D = theta2 * normE*  compute_norm1_deviation(dimension, extra_sample)
		G_hat = compute_H(M[:,index] + D)
	else:
		
		index = np.random.randint(0,n_eval, extra_sample)

		M_resampled =  M[:,index]
		D = theta2 * normE * compute_norm1_deviation(dimension, extra_sample)

		G_hat = compute_H(M_resampled + D)
	return G_hat

def compute_theoretical_error(var, L, delta):
	# from Tropp 2015, last equation of 7.7.4 (proof of theorem for expectation bound)
	return np.sqrt(2*var*np.log(1+delta)) + 2/3*L*np.log(1+delta)+4*np.sqrt(var)+8/3*L


def compute_SF_variance(beta2, normE, normH, m):
	return beta2*normE*normH/m


def compute_SF_L(beta2, normE, m):
	return (1+beta2)*normE/m


def compute_beta2(M):
	maxi = 0.0
	dimension, n_eval = np.shape(M)
	for i in range(0,n_eval):
		maxi = max(maxi, pow(np.linalg.norm(M[:,i]), 2.0) )

	normE = compute_normE(M)

	return maxi/normE

def compute_MF_variance(theta2, beta2, delta, normH, m1, m2):
	return ((pow(2+np.sqrt(theta2), 2.0))*theta2/m1  + pow(np.sqrt(theta2)+np.sqrt(beta2), 2.0)*pow(1+np.sqrt(theta2), 2.0)/m2)*pow(delta, 2.0)*pow(normH, 2.0)

def compute_MF_L(theta2, beta2, delta, normH, m1, m2):
	return delta* normH * max(2*np.sqrt(theta2)*(2*np.sqrt(beta2)+np.sqrt(theta2))/m1, 2*pow(np.sqrt(beta2)+np.sqrt(theta2), 2.0)/m2)
