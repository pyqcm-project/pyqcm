import numpy as np
import pyqcm
import os

first_time=True
####################################################################################################
# Internal functions

def _predict_bad(x, y, p):
	for i in range(len(x)):
		if abs(x[i] - y[i]) / (abs(x[i]) + 0.01) > p:
			return True
	return False


#---------------------------------------------------------------------------------------------------
def _predict_good(x, y, p):
	for i in range(len(x)):
		if abs(x[i] - y[i]) / abs((x[i]) + 0.01) > p:
			return False
	return True

#---------------------------------------------------------------------------------------------------
def _poly_predictor(X,Y,i,n):
	x = X[i-n-1:i]  # the n+1 previous values of X
	y = Y[i-n-1:i]  # the n+1 previous values of Y
	pol = np.polyfit(x, y, n)
	return np.polyval(pol, X[i])



####################################################################################################
# Additional methods of the model class

#---------------------------------------------------------------------------------------------------
def loop_from_file(self, task, file):
	"""Performs a task 'task' over a set of model parameters defined in a file. The definition of each
	model instance must be done in the task 'task'; it is not done by this looping function.

	:param task: function (task) to perform
	:param str file: name of the data file specifying the solutions, in the same tab-separated format usually used to write solutions

	"""
	param = self.parameter_set()
	data = np.genfromtxt(file, names=True, dtype=None, encoding='utf8')

	if len(data.shape) == 0 : data = np.expand_dims(data, axis=0)

	print('number of data sets: ', len(data))

	independent_params = []
	for key, val in param.items():
		if val[1] is None and key in data.dtype.names:
			independent_params.append(key)

	print('independent parameters: ', independent_params)

	N = len(data)
	for i in range(N):
		print('*'*30, ' set ', i+1, ' of ', N, '*'*30)
		for x in independent_params:
			print(x, ' --> ', data[x][i])
			self.set_parameter(x, data[x][i])
		task()


#---------------------------------------------------------------------------------------------------
def loop_from_table(self, task, data):
	"""Performs a task 'task' over a set of model parameters defined in a table. The definition of each
	model instance must be done in the task 'task'; it is not done by this looping function.

	:param task: function (task) to perform
	:param ndarray data: table of parameters, with named columns, as read or filtered from numpy.genfromtxt()

	"""
	param = self.parameter_set()
	print('number of data sets: ', len(data))

	independent_params = []
	for key, val in param.items():
		if val[1] is None and key in data.dtype.names:
			independent_params.append(key)

	print('independent parameters: ', independent_params)

	N = len(data)
	for i in range(N):
		print('*'*30, ' set ', i+1, ' of ', N, '*'*30)
		for x in independent_params:
			print(x, ' --> ', data[x][i])
			self.set_parameter(x, data[x][i])
		task()


#---------------------------------------------------------------------------------------------------
# performs a loop (a linear trajectory) between two subsets of parameters
def linear_loop(self, N, task, varia=None, params = None, predict=True):
	"""
	Performs a loop with a predictor. The definition of each
	model instance must be done in 'task'; it is not done by this looping function.

	:param N: number of intervals within the loop
	:param task: function called at each step of the loop
	:param [str] varia: names of the variational parameters
	:param {str (float,float)} P: dict of parameters to vary with a tuple of initial and final values 
	:param boolean predict: if True, uses a linear or quadratic predictor

	"""

	pyqcm.banner('linear loop over {:d} values'.format(N), '%', skip=1)
	if type(params) != dict:
		raise TypeError('the argument "P" of linear_loop must be a dict of 2-tuples')
	if varia is None:
		raise pyqcm.MissingArgError('variational parameters (varia) must be specified')
	if type(varia) != list:
		raise TypeError('the argument "varia" of linear_loop must be a list')

	current_value = {}

	nvar = len(varia)  # number of variational parameters
	sol = np.empty((4, nvar))  # current solution + 3 previous solutions
	start = [0]*nvar
	for iter in range(N+1):
		# sets the control parameters of the loop
		for x in params:
			current_value[x] = params[x][0]*(N-iter)/N + params[x][1]*iter/N
			self.set_parameter(x, current_value[x])
		
		sol = np.roll(sol, 1, 0)  # cycles the solutions
		# predicting the starting value from previous solutions (do nothing if loop_counter=0)
		fit_type = ''
		if iter == 1 or not predict:
			start = sol[1, :]  # the starting point is the previous value in the loop
		elif iter > 1:
			if iter == 2:
				fit_type = ' (linear fit)'
				x = [iter-1, iter-2]  # the two previous values of the loop parameter
				y = sol[1:3, :]  # the two previous values of the loop parameter
				pol = np.polyfit(x, y, 1)
			else:
				fit_type = ' (quadratic fit)'
				x = [iter-1, iter-2, iter-3]  # the three previous values of the loop parameter
				y = sol[1:4, :]  # the three previous values of the loop parameter
				pol = np.polyfit(x, y, 2)
			for i in range(nvar):
				start[i] = np.polyval(pol[:, i], iter)

		pyqcm.banner('loop index = {:d}'.format(iter + 1), '=')
		for x in params:
			print(x, ' ===> {:1.4f}'.format(current_value[x]))
		if iter > 0:
			print('predictor: ', start, fit_type)
		try:
			task()
		except: raise
			
		sol[0,:] = self.parameters(varia)

	pyqcm.banner('linear loop ended normally', '%')

#---------------------------------------------------------------------------------------------------
# performs a loop for VCA or CDMFT with an external loop parameter and an optional control on an order parameter

def controlled_loop(self, task, varia=None, loop_param=None, loop_range=None, control_func=None, retry=None, max_retry=4, predict=True):
	"""
	Performs a controlled loop for VCA or CDMFT with a predictor.  The definition of each
	model instance must be done and returned by 'task'; it is not done by this looping function.

	:param task: a function called at each step of the loop. Must return a model_instance.
	:param [str] varia: names of the variational parameters
	:param str loop_param: name of the parameter looped over
	:param (float, float, float) loop_range: range of the loop (start, end, step)
	:param control_func: (optional) name of the function that controls the loop (returns boolean). Takes a model instance as argument
	:param char retry: If None, stops on failure. If 'refine', adjusts the step (divide by 2) and retry. If 'skip', skip to next value.
	:param int max_retry: Maximum number of retries of either type
	:param boolean predict: if True, uses a linear or quadratic predictor

	"""
	if retry not in (None, 'skip', 'refine'): 
		raise ValueError('"retry" argument must be None, "refine" or "skip".')

	pyqcm.banner('controlled loop over '+loop_param, '%', skip=1)
	if loop_param is None:
		raise pyqcm.MissingArgError('The name of the control parameter (loop_param) is missing')
	if loop_range is None:
		raise pyqcm.MissingArgError('A loop range (min, max, step) must be provided')
	if (loop_range[1]-loop_range[0])*loop_range[2] < 0:
		raise ValueError('the loop range is incoherent: step has wrong sign')
	if varia is None:
		raise pyqcm.MissingArgError('variational parameters (varia) must be specified')
	if pyqcm.is_sequence(varia) == False:
		varia = (varia,)

	nvar = len(varia)  # number of variational parameters
	sol = np.empty((4, nvar))  # current solution + 3 previous solutions
	prm = np.empty(4)  # current loop parameter + 3 previous values

	# prm[0] = loop_range[0] - loop_range[2]
	prm[0] = loop_range[0]
	step = loop_range[2]
	loop_counter = 0
	try_again = False
	nretry = 0
	fac = 1
	start = [0]*nvar
	abort = False
	while True:  # main loop here
		self.set_parameter(loop_param, prm[0])
		pyqcm.banner('loop index = {:d}, {:s} = {:1.4f}'.format(loop_counter + 1, loop_param, prm[0]), '=', skip=1)
		try:
			I = task()

		except (pyqcm.OutOfBoundsError, pyqcm.TooManyIterationsError, Exception) as E:
			print(E)
			if loop_counter == 0 or not retry:
				raise ValueError('Out of bound on starting values in controlled_loop(), aborting')
			else:
				try_again = True
			
		if control_func is not None:
			if not control_func(I):
				if loop_counter < 2 and retry == None:
					print('Control failed on starting. Maybe try with different starting point')
					break
				elif retry == None:
					print('Control failed. Aborting loop.')
					abort = True
					break
				try_again = True
	
		if try_again:
			if retry == 'refine':
				step = step / 2
				pyqcm.banner('retrying with half the step', '=')
			elif retry == 'skip':
				fac += 1
				pyqcm.banner('skipping to next value', '=')
			try_again = False
			nretry += 1
			if nretry > max_retry:
				break
		else:
			prm = np.roll(prm, 1)  # cycles the values of the loop parameter
			sol = np.roll(sol, 1, 0)  # cycles the solutions
			nretry = 0
			fac = 1

		prm[0] = prm[1] + fac*step  # updates the loop parameter

		sol[0,:] = I.parameters(varia)

		if loop_range[1] >= loop_range[0] and prm[0] > loop_range[1]:
			break
		if loop_range[1] <= loop_range[0] and prm[0] < loop_range[1]:
			break
	
		# predicting the starting value from previous solutions (do nothing if loop_counter=0)
		fit_type = ''
		if loop_counter == 1 or not predict:
			start = sol[1, :]  # the starting point is the previous value in the loop
		elif loop_counter > 1:
			if loop_counter == 2:
				fit_type = ' (linear fit)'
				x = prm[1:3]  # the two previous values of the loop parameter
				y = sol[1:3, :]  # the two previous values of the loop parameter
				pol = np.polyfit(x, y, 1)
			else:
				fit_type = ' (quadratic fit)'
				x = prm[1:4]  # the three previous values of the loop parameter
				y = sol[1:4, :]  # the three previous values of the loop parameter
				pol = np.polyfit(x, y, 2)
			for i in range(nvar):
				start[i] = np.polyval(pol[:, i], prm[0])

		if loop_counter > 0:
			print('predictor : ', start, fit_type)
			for i in range(len(varia)):
				self.set_parameter(varia[i], start[i])
				print(' ---> ', varia[i], ' = ', start[i])

		if loop_counter > 2 and not retry and retry:
			if(_predict_good(sol[0, :], start, 0.01) and step <  loop_range[2]):
				step *= 1.5
				print('readjusting step to ', step)
			elif(_predict_bad(sol[0, :], start, 0.2) and step > loop_range[2] / 10):
				step /= 1.5
				print('readjusting step to ', step)

		loop_counter += 1

	if nretry > max_retry:
		pyqcm.banner('controlled loop ended on signal by control function', '%')
	elif abort:
		pyqcm.banner('controlled loop ended with error', '%')
	else:
		pyqcm.banner('controlled loop ended normally', '%')


#---------------------------------------------------------------------------------------------------
def fade(self, task, P, n):
	"""
	fades the model between two sets of parameters, in n steps

	:param task: task to perform wihtin the loop
	:param dict P: dict of parameters with a tuple of values (start and finish) for each
	:param n: number of steps

	"""

	lambda_array = np.linspace(0.0, 1.0, n)
	for L in lambda_array:
		for p in P:
			self.set_parameter(p, L*P[p][1] + (1-L)*P[p][0])
		task()

#---------------------------------------------------------------------------------------------------
# def controlled_fade(self, task, P, n, C, file='fade.tsv', tol=1e-4, method='Broyden'):
# # IN DEVELOPMENT
# 	"""
# 	fades the model between two sets of parameters, in n steps

# 	:param task: task to perform wihtin the loop
# 	:param dict P: dict of parameters with a tuple of values (start and finish) for each
# 	:param n: number of steps
# 	:param dict C: dict of adjustable parameters whose average (the value of C) should stay fixed during the fade
# 	:param str file: file to which the converged results are written
# 	:param float tol: precision on the averages

# 	"""
# 	assert type(P) is dict
# 	assert type(C) is dict
# 	for x in P:
# 		assert type(P[x]) is tuple

# 	print('Fading procedure in ', n, 'steps :')
# 	for x in P:
# 		print(x, ' from ', P[x][0], ' to ', P[x][1])

# 	A = np.zeros(len(C)) # allocating the array of fixed average values 
	
# 	def F(x):
# 		global I_current
# 		for y in C :
# 			self.set_parameter(y, x, pr=True)
# 		I = task()
# 		ave = I.averages()
# 		B = np.array([ave[y] for y in C]) # average values
# 		return B-A

# 	par = self.parameters()
# 	Cv = np.array([par[x] for x in C])
# 	Z = np.linspace(0.0, 1.0, n)
# 	for i, z in enumerate(Z):
# 		for p in P:
# 			self.set_parameter(p, z*P[p][1] + (1-z)*P[p][0], pr=True)
# 		if i==0:
# 			A = F(Cv)
# 			x0 = Cv
# 		else:
# 			try:
# 				alpha = X[2]
# 			except:
# 				alpha = 0.0
# 			if method == 'Broyden' : X = pyqcm.broyden(F, x0, iJ0 = alpha, xtol=tol, maxiter=32)
# 			else : X = pyqcm.fixed_point_iteration(F, x0, xtol=tol, maxiter=32,  alpha = 0.0)
# 			I = pyqcm.model_instance(self)
# 			I.averages()
# 			I.write_summary(file)
# 			x0 = X[0]
			

#---------------------------------------------------------------------------------------------------
def Hartree_procedure(self, task, couplings, maxiter=32, iteration='fixed_point', eps_algo=0, file='hartree.tsv', SEF=False, pr = False, alpha=0.0):
	"""
	Performs the Hartree approximation
	
	:param task: task to perform within the loop. Must return a model_instance
	:param [hartree] couplings: sequence of couplings (or single coupling)
	:param int maxiter: maximum number of iterations
    :param str iteration: method of iteration of parameters ('fixed_point' or 'Broyden')
	:param int eps_algo: number of elements in the epsilon algorithm convergence accelerator = 2*eps_algo + 1 (0 = no acceleration)
    :param float alpha: if iteration='fixed_point', damping parameter, otherwise trial inverse Jacobian (if a float, that is (1+alpha)*unit matrix)
	:returns: model instance, converged inverse Jacobian

	"""
	
	global first_time

	if pyqcm.is_sequence(couplings) is False:
		couplings = (couplings,)

	pyqcm.banner('Hartree procedure', c='*', skip=1)
	hartree_converged = False
	diff_tot = 1e6
	var_data = np.empty((len(couplings), maxiter+2))

	n = len(couplings)
	P = self.parameters()
	X = np.empty(n)
	for i,C in enumerate(couplings):
		X[i] = P[C.Vm]


	def F(x):
		x_new = np.copy(x)
		for i,C in enumerate(couplings):
			self.set_parameter(C.Vm, x[i])
		I = task()
		for i,C in enumerate(couplings):
			C.update(I, pr=pr)
			C.print()
			x_new[i] = C.vm
		return x - x_new
	
	def G():
		hartree_converged = True
		for i,C in enumerate(couplings):
			hartree_converged = hartree_converged and C.converged()
		return hartree_converged
	
	niter = 0
	if iteration == 'Broyden':
		hartree_params, niter, alpha = pyqcm.broyden(F, X, iJ0 = alpha, maxiter=maxiter, convergence_test=G)
	elif iteration == 'fixed_point':
		hartree_params, niter = pyqcm.fixed_point_iteration(F, X, xtol=1e-6, convergence_test=G, maxiter=maxiter, alpha=alpha, eps_algo=eps_algo)
	else:
		raise ValueError('type of iteration unknown in call to Hartree_procedure(...)')

	I = pyqcm.model_instance(self)
	I.props['method'] = iteration
	I.props['iterations'] = niter
	if SEF: I.Potthoff_functional(hartree=couplings)
	I.write_summary(file)

	pyqcm.banner('Hartree procedure has converged', c='*')
	return I, niter, alpha
	


#---------------------------------------------------------------------------------------------------
# performs a loop over mu with adjustable step to make the density step more constant
# IN DEVELOPMENT
def flexible_loop(self, task, initial_mu, final_n, initial_step, delta_n=0.005, varia=[]):
	"""
	Performs a controlled loop for VCA or CDMFT with a predictor.  The definition of each
	model instance must be done and returned by 'task'; it is not done by this looping function.

	:param task: a function called at each step of the loop. Must return a model_instance.
	:param (float) initial_mu: initial value of mu
	:param (float) initial_step: initial step in mu
	:param (float) final_n: final value of density
	:param (float) delta_n: approximate step in density to keep constant
	:param [str] varia: names of the variational parameters

	"""

	nvar = len(varia)  # number of variational parameters
	sol = np.empty((4, nvar))  # current solution + 3 previous solutions
	mu = np.empty(4)  # current mu + 3 previous values
	n = np.empty(4)  # current density + 3 previous values
	if initial_step*delta_n < 0 : delta_n *= -1

	mu[0] = initial_mu
	P = self.parameters()
	for i,v in enumerate(varia):
		sol[0] = P[v]
	loop_counter = 1
	
	while True:  # main loop here
		self.set_parameter('mu', mu[0])

		pyqcm.banner('loop index = {:d},  mu = {:1.4f}'.format(loop_counter + 1, mu[0]), '=', skip=1)
		try:
			I = task()

		except (pyqcm.OutOfBoundsError, pyqcm.TooManyIterationsError, Exception) as E:
			print(E)
			return
					
		n[0] = I.averages()['mu']
		print('density = ', n[0])
		mu = np.roll(mu, 1)  # cycles the values of the loop parameter
		n = np.roll(n, 1)  # cycles the values of the loop parameter
		sol = np.roll(sol, 1, 0)  # cycles the solutions


		n[0] = n[1] + delta_n
		if delta_n < 0 and n[0] < final_n: break
		if delta_n > 0 and n[0] > final_n: break

		# predicting the starting value from previous solutions (do nothing if loop_counter=0)
		fit_type = ''
		if loop_counter == 1:
			start = sol[1, :]  # the starting point is the previous value in the loop
			mu[0] = mu[1] + initial_step  # updates the loop parameter
		elif loop_counter == 2:
			fit_type = ' (linear fit)'
			x = mu[1:3]  # the two previous values of mu
			y = sol[1:3, :]  # the two previous values of the variational parameter
			z = n[1:3]  # the two previous values of the density
			pol_mu = np.polyfit(z, x, 1)
			pol = np.polyfit(x, y, 1)
			mu[0] = np.polyval(pol_mu, n[0])
			if abs(mu[0]-mu[1]) > 10*np.abs(initial_step):
				mu[0] = mu[1] + 10*initial_step
				loop_counter -= 1
			else:
				for i in range(nvar):
					start[i] = np.polyval(pol[:, i], mu[0])

		else:
			fit_type = ' (quadratic fit)'
			x = mu[1:4]  # the three previous values of the loop parameter
			y = sol[1:4, :]  # the three previous values of the loop parameter
			z = n[1:4]  # the three previous values of the density
			pol_mu = np.polyfit(z, x, 2)
			pol = np.polyfit(x, y, 2)
			mu[0] = np.polyval(pol_mu, n[0])
			for i in range(nvar):
				start[i] = np.polyval(pol[:, i], mu[0])

		if loop_counter > 1:
			print('flexible_loop step : mu = ', mu[0]-mu[1])
			print('predictor : ', start, fit_type)
			for i in range(len(varia)):
				self.set_parameter(varia[i], start[i])
				print(' ---> ', varia[i], ' = ', start[i])

		loop_counter += 1

	pyqcm.banner('flexible loop ended normally', '%')
