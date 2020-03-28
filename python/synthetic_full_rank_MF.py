import numpy.matlib
import matplotlib
import matplotlib.pyplot as plt
from dimension_reduction_common import *
import cPickle as pickle


def gradF(X, param):
	dimension, N_sample = np.shape(X)
	grad = np.multiply(X,np.matlib.repmat(param,1,N_sample)) * np.sqrt(3)
	return grad

def gradG(X, param, b, T, coef):
	dimension, N_sample = np.shape(X)
	vect = np.zeros((dimension,1))
	vect[-1] = np.linalg.norm(param)
	grad = np.multiply(X,np.matlib.repmat(param,1,N_sample)) * np.sqrt(3) + np.multiply(b*np.sin(X/T),np.matlib.repmat(vect,1,N_sample))
	return grad

if __name__ == "__main__":

	np.set_printoptions(precision=2)

	file = "synthetic_full_rank_MF"

	n_repeat = 100

	dim = 100

	int_vect = [1, 5, 10, 50, 100] # vector of target intrinsic dimension for parametric study
	sample_vect = np.array([10, 100, 1000])

	alpha = [0.65, 0.22,  0.1, 0.016,0]
	for i in range(np.size(alpha)):
		int_vect[i] = np.sum(np.exp(-alpha[i]*(np.arange(0,100))))

	
	T = 0.1
	b = np.sqrt(0.05)

	all_computation = True

	if all_computation:
		error_SF_matrix = np.zeros((n_repeat, np.size(sample_vect), np.size(int_vect)))
		error_matrix = np.zeros((n_repeat, np.size(sample_vect), np.size(int_vect)))
	error_theoric_matrix = np.zeros((np.size(sample_vect), np.size(int_vect)))

	for k, tar in enumerate(int_vect):

		target_rel_error = 0.1
		beta2_true = 3.0 # closed form for that example
		
		# Parameters of H_true (eigen values in this case)
		param = np.zeros((dim,1))
		# param[0] = 1.0
		# param[0:tar] = 1.0
		param[:,0] = np.exp(-alpha[k]*(np.arange(0,100))).transpose()

		# We know this from closed form expression
		H_true = np.diag(np.power(param[:,0],2.0))
		H_true_norm = np.linalg.norm(H_true, ord=2)
		int_dim_true = np.trace(H_true)/H_true_norm
		if T<np.pi/2.0:
			theta2_true = 	np.power(b, 2.0)
		else:
			theta2_true = np.power(b*np.sin(T), 2.0) # to compute in closed form
		print "Theta2 ", theta2_true
		
		mult = pow(np.sqrt(theta2_true) + np.sqrt(beta2_true), 2.0) *max(pow(1+np.sqrt(theta2_true), 2.0)/(theta2_true * pow(2+np.sqrt(theta2_true), 2.0)),  1/(np.sqrt(theta2_true)*(2*np.sqrt(beta2_true)+np.sqrt(theta2_true)))    )
		lf_sample_size_vect = np.ceil(sample_vect*mult)
		lf_sample_size_vect = lf_sample_size_vect.astype(int)
		
		for index, n in enumerate(sample_vect):

			for i in range(0,n_repeat):

				if all_computation:
					X = np.random.uniform(-1.0, 1.0, (dim,n+lf_sample_size_vect[index]))
					GradF1 = gradF(X[:,:n], param)
					GradG1 = gradG(X[:,:n], param, b, T, param[::-1])
					GradG2 = gradG(X[:,n+1:-1], param, b, T, param[::-1])

					H1 = compute_H(GradF1)
					G1 = compute_H(GradG1)
					G2 = compute_H(GradG2)
					H = H1-G1+G2

					print np.linalg.norm(H_true-H, ord=2)/H_true_norm
					print np.linalg.norm(H_true-H1, ord=2)/H_true_norm

				L_th = compute_MF_L(theta2_true, beta2_true, int_dim_true, H_true_norm, n, lf_sample_size_vect[index])
				var_th = compute_MF_variance(theta2_true, beta2_true, int_dim_true, H_true_norm, n, lf_sample_size_vect[index])
				error_theoric_matrix[index, k] = compute_theoretical_error(var_th, L_th, dim)/ H_true_norm

				if all_computation:
					################################################
					# Computing the ``low fidelity'' H
					################################################
					
					# MF estimator
					error = np.linalg.norm(H_true-H, ord=2)
					rel_error = error/H_true_norm
					error_matrix[i,index,k] = rel_error

					# SF estimator
					error = np.linalg.norm(H_true-H1, ord=2)
					rel_error = error/H_true_norm
					error_SF_matrix[i,index,k] = rel_error

					print "\nThe error (in operator norm) is ", error
					print "The relative error (in operator norm) is ", rel_error
					print "True intrinsic dimension: ", int_dim_true


			if all_computation:
				print "-----------------------------"
				print "Used %d samples" % (n)
				print "Intrinsic dimension: ", int_dim_true
				print "Average relative error ", np.mean(error_matrix, axis = 0)
				print "Variance relative error ", np.var(error_matrix, axis = 0)


	if all_computation:
		pickle.dump(error_matrix, open(file + "/error_matrix.p", "wb"))
		pickle.dump(error_SF_matrix, open(file + "/error_SF_matrix.p", "wb"))
	pickle.dump(error_theoric_matrix, open(file + "/error_theoric_matrix.p", "wb"))
	pickle.dump(int_vect, open(file + "/int_vect.p", "wb"))
	pickle.dump(sample_vect, open(file + "/sample_vect.p", "wb"))


	symb = ['o', 's', 'v', '+']

	if all_computation:
		error_matrix = pickle.load(open(file + "/error_matrix.p", "rb"))
		error_SF_matrix = pickle.load(open(file + "/error_SF_matrix.p", "rb"))
	error_theoric_matrix = pickle.load(open(file + "/error_theoric_matrix.p", "rb"))
	int_vect = pickle.load(open(file + "/int_vect.p", "rb"))
	sample_vect = pickle.load(open(file + "/sample_vect.p", "rb"))


	mycolor = ['red', 'black', 'blue', 'magenta', 'green','cyan','black','black','black','black','black','black','black']

	plt.rc('text', usetex = True)
	plt.rc('font', family = 'serif')
	plt.rc('font', serif = 'Computer Modern')
	plt.rcParams.update({'font.size':20})

	fig, ax = plt.subplots()
	fig.set_size_inches(10,8)

	if all_computation:
		mean_error = np.mean(error_matrix, axis = 0)
		mean_SF_error = np.mean(error_SF_matrix, axis = 0)

	print mean_SF_error/mean_error
	print mean_error
	print mean_SF_error

	

	for i, intdim in enumerate(int_vect[::2]):
		print i
		if all_computation:
			plt.semilogy(sample_vect, mean_error[:,i], '-'+symb[i], ms=10, color = mycolor[i], label='$\delta_{H}= {intdim}$'.format(H="H", intdim=int(np.floor(intdim))), linewidth=2)
		plt.semilogy(sample_vect, error_theoric_matrix[:,i], '--'+symb[i], ms=10, color = mycolor[i], linewidth=2)

	ax.set_xscale('log')
	plt.ylabel(r'Relative error $\Vert H - \hat{H}\Vert/\Vert H\Vert$', fontsize = 28)
	plt.xlabel(r'Number of samples $m_{1}$', fontsize = 28)
	plt.xticks(fontsize = 28)
	plt.yticks(fontsize = 28)
	ax.set_xticks(sample_vect)
	ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	plt.xlim((sample_vect[0], sample_vect[-1]))
	plt.ylim([pow(10.0,-3 ), pow(10.0, 3.0)])
	plt.legend(loc='best')
	plt.savefig(file + '/numerical_validation_sample_size.pdf' , bbox_inches="tight")	
	plt.close()



	fig, ax = plt.subplots()
	fig.set_size_inches(10,8)
	for i, sample_size in enumerate(sample_vect):
		print i
		if all_computation:
			plt.semilogy(int_vect, mean_error[i,:], '-'+symb[i], ms=10, color = mycolor[i], label='$m_1= {sample_size}$'.format(sample_size=sample_size), linewidth=2)
		plt.semilogy(int_vect, error_theoric_matrix[i,:], '--'+symb[i], ms=10, color = mycolor[i],  linewidth=2)

	ax.set_xscale('log')
	plt.ylabel(r'Relative error $\Vert H - \hat{H}\Vert/\Vert H\Vert$', fontsize = 28)
	plt.xlabel(r'Intrinsic dimension $\delta_{H}$', fontsize = 28)
	plt.xticks(fontsize = 28)
	plt.yticks(fontsize = 28)
	ax.set_xticks(np.floor(int_vect))
	ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
	plt.xlim((1, int_vect[-1]))
	plt.ylim([pow(10.0,-3 ), pow(10.0, 3.0)])
	plt.legend(loc='upper left')
	plt.savefig(file + '/numerical_validation_int_dim.pdf' , bbox_inches="tight")	
	plt.close()


