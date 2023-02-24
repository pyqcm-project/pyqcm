################################################################################
# This file contains functions implementing the Variational Cluster Approximation (VCA)
################################################################################

import numpy as np
from numpy.distutils.misc_util import is_sequence
import pyqcm
import time
from scipy.optimize import minimize

#-------------------------------------------------------------------------------
# MPI matter
comm = None
root = False
try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    if comm.Get_rank() == 0:
        root = True
        print('MPI available. Working on ', comm.Get_size(), ' concurrent processes')
except ImportError:
    root = True

first_time = True

################################################################################
# PRIVATE FUNCTIONS
################################################################################

################################################################################
# evaluation of many instances of a function in parallel

def __evalF(func, x):
    n = x.shape[0]
    y = np.empty(n)
    for i in range(n):
        y[i] = func(x[i,:])
    return y    

def __evalF_mpi(func, x):
    n = x.shape[0]
    nproc = comm.Get_size()
    rank = comm.Get_rank()
    y = np.zeros(n)
    y_tmp = np.zeros(n)
    k = 0  # index of points
    for i in range(n):
        if i % nproc == rank:
            y_tmp[i] = func(x[i,:])

    comm.Allreduce(y_tmp, y)
    return y


################################################################################
# quasi-Newton method function evaluation

def __QN_values(func, x, step):
    """Computes the function func of n variables at :math:`2n+1` points at x and at n pairs of points located at :math:`\pm` step from x in each direction. For use in the quasi-Newton method.

    :param func: function of n variables
    :param x: base point (n components NumPy array)
    :param float step: step to take in each direction in computing derivatives
    :returns: array of :math:`2n+1` values

    """
    n = len(x)
    N = 2 * n + 1
    xp = np.empty((N, n))
    for i in range(N):
        xp[i, :] = np.copy(x)
    k = 0
    for i in range(n):
        k += 1
        xp[k, i] += step[i]
        k += 1
        xp[k, i] -= step[i]

    if comm is None:
        return __evalF(func, xp)
    else:    
        return __evalF_mpi(func, xp)


################################################################################
# quasi-Newton method

def __quasi_newton(func=None, start=None, step=None, accur=None, max=10, gtol=1e-4, bfgs=False, max_iteration=30, max_iter_diff=None, hartree=None):
    """Performs the quasi_newton procedure
    
    :param func: a function of N variables
    :param [float] start: the starting values
    :param [float] step: the steps used to computed the numerical second derivatives
    :param [float] accur: the required accuracy for each variable
    :param [float] max: maximum absolute value of each parameter
    :param float gtol: the gradient tolerance (gradient must be smaller than gtol for convergence)
    :param boolean bfgs: True if the BFGS method is used, otherwise the symmetric rank-1 formula is used (default)
    :param int max_iterations: maximum number of iterations, beyond which an exception is raised
    :param float max_iter_diff: maximum step to make
    :param (class hartree) hartree: Hartree approximation couplings (see pyqcm/hartree.py)
    :return (float, [float], [[float]]): tuple of x (the solution), gradient (array, the value of the gradient), hessian (matrix, the Hessian matrix)

    """
    n = len(start)
    gradient = np.zeros(n)
    gradient0 = np.zeros(n)
    dx = np.zeros(n)
    y = np.zeros(n)
    ihessian = np.eye(n)  # inverse Hessian matrix
    iteration = 0
    x = start
    hartree_converged = True
    step0 = step

    while iteration < max_iteration:
        x0 = x
        x, dx, gradient, ihessian = __quasi_newton_step(iteration, func, x0, step, gradient, dx, bfgs)
        __vca_accuracy_warning(accur, ihessian)
        iteration += 1

        if hartree != None:
            hartree_converged = True
            for C in hartree:
                C.update()
                C.print()
                hartree_converged = hartree_converged and C.converged()

        if (np.linalg.norm(gradient) < gtol) and hartree_converged :
            if root:
                print('convergence on gradient after ', iteration, ' iterations')
            break

        converged = True
        for i in range(n):
            if np.abs(dx[i]) > accur[i]:
                converged = False
                break
        if converged and hartree_converged:
            if root:
                print('convergence on position after ', iteration, ' iterations')
            break

        #redefining the steps
        step_multiplier = 2.0
        for i in range(n):
            step[i] = np.abs(dx[i])
            if step[i] > step0[i]:
                step[i] = 2.0*step0[i]
            if step[i] < step_multiplier*accur[i]:
                step[i] = step_multiplier*accur[i]

    if iteration == max_iteration:
        raise pyqcm.TooManyIterationsError(max_iteration)

    return x, gradient, ihessian

################################################################################
# quasi-Newton method (step)

def __quasi_newton_step(iteration = 0, func=None, x=None, step=None, gradient=None, dx=None, bfgs=False, max_diff=None):
    """Performs a step of the quasi_newton procedure
    
    :param int iter: iteration number
    :param func: a function of N variables
    :param [float] x: the starting values
    :param [float] step: the steps used to computed the numerical second derivatives
    :param [float] dx: the previous difference
    :param boolean bfgs: True if the BFGS method is used, otherwise the symmetric rank-1 formula is used (default)
    :param float max_diff: maximum step to make
    :return (float, float, [float], [[float]]): tuple of x (the new point), dx (the difference between the current and previous point), gradient (array, the value of the gradient), hessian (matrix, the Hessian matrix)

    """
    n = len(x)
    y = np.zeros(n)
    ihessian = np.eye(n)  # inverse Hessian matrix

    F = __QN_values(func, x, step)
    gradient0 = np.copy(gradient)
    for i in range(n):
        gradient[i] = (F[2 * i + 1] - F[2 * i + 2]) / (2 * step[i])

    if iteration == 0:
        for i in range(n):
            ihessian[i, i] = step[i] * step[i] / (F[2 * i + 1] + F[2 * i + 2] - 2 * F[0])

    else:
        y = gradient - gradient0
        if bfgs:  # BFGS method
            z1 = 1.0e-14 + np.dot(y, dx)
            z1 = 1.0 / z1
            X = np.eye(n)
            X -= z1 * np.kron(y, dx).reshape((n, n))
            ihessian = np.dot(X.T, np.dot(ihessian, X))
            ihessian += z1 * np.kron(dx, dx).reshape((n, n))
        else:  # symmetric rank 1
            u = np.dot(ihessian, y)  # $u = H y_k$
            dx -= u  # the $\Delta x_k - H_k y_k$ of Wikipedia
            z1 = 1.0e-14 + np.dot(y, dx)
            z1 = 1.0 / z1
            ihessian += z1 * np.kron(dx, dx).reshape((n, n))

    dx = -np.dot(ihessian, gradient)

    if max_diff is not None:
        norm_dx = np.linalg.norm(dx)
        if norm_dx > max_diff:
            ratio = max_diff / norm_dx
            dx *= ratio
            if root:
                s1 = 'Norm of QN iteration variation dx is out of bounds: {} > {}'.format(norm_dx, max_diff)
                s2 = 'Readjusting: dx = dx * {:.4}%'.format(ratio * 100)
                print(s1, '\n', s2)

    x += dx
    if root:
        print('QN iteration no ', iteration+1, '\t x = ', x)

    return x, dx, gradient, ihessian


################################################################################
# Newton-Raphson method :  computation of the Hessian

def __NR_Hessian(func, x, step):
    """Computes the function func of n variables at :math:`1 + 2n + n(n-1)/2` points in order to fit a quadratic form x and n pairs of points located at :math:`\pm` step from x in each direction
    
    :param func: the function
    :param [float] x: the base point 
    :param float step: step taken in each direction
    :return (float, [float], [[float]]): the value of the function, the gradient, and the Hessian

    """
    n = len(x)
    N = 1 + 2*n + (n*(n-1))//2
    y = np.zeros(N)
    hess = np.zeros((n,n))

    xp = np.zeros((N, n))
    for k in range(N):
        xp[k, :] = np.copy(x)
 
    k = 0
    # points located at +/- step
    for i in range(n):
        k += 1
        xp[k, i] += step[i]
        k += 1
        xp[k, i] -= step[i]

    # points located off diagonal
    for i in range(n):
        for j in range(i):
            k += 1
            xp[k, i] += 0.707*(2*(i%2)-1)*step[i]
            xp[k, j] += 0.707*(2*(j%2)-1)*step[j]

    if comm is None:
        y = __evalF(func, xp)
    else:    
        y = __evalF_mpi(func, xp)

    # setting fit coefficients
    A = np.empty((N,N))
    A[:, 1:n+1] = xp
    for k in range(N):
        A[k,0] = 1.0
        m = n+1
        for i in range(n):
            for j in range(i+1):
                A[k, m] = xp[k, i]*xp[k,j]
                if i == j :
                    A[k, m] = xp[k, i]*xp[k,j]*0.5
                else:
                    A[k, m] = xp[k, i]*xp[k,j]        
                m += 1

    # inverting
    A1 = np.linalg.inv(A)
    y = np.dot(A1, y)
    val = y[0]
    grad = y[1:n+1]
    val += np.sum(grad*x)
    m = n+1
    for i in range(n):
        for j in range(i+1):
            hess[i,j] = y[m]
            hess[j,i] = y[m]
            if i == j :
                val += x[i] * x[j] * y[m] * 0.5
            else:
                val += x[i] * x[j] * y[m]
            m += 1
    grad += np.dot(hess,x)
    return val, grad, hess


################################################################################
# Newton-Raphson method

def __newton_raphson(func=None, start=None, step=None, accur=None, max=10, gtol=1e-4, max_iteration=30, max_iter_diff=None, hartree=None):
    """Performs the Newton-Raphson procedure
    
    :param func: a function of N variables
    :param [float] start: the starting values
    :param [float] step: the steps used to computed the numerical second derivatives
    :param [float] accur: the required accuracy for each variable
    :param [float] max: maximum absolute value of each parameter
    :param float gtol: the gradient tolerance (gradient must be smaller than gtol for convergence)
    :param int max_iterations:  maximum number of iterations, beyond which an exception is raised
    :param float max_iter_diff: optional maximum value of the maximum step
    :param (class hartree) hartree: Hartree approximation couplings (see pyqcm/hartree.py)
    :returns (float, [float], [[float]]): the value of the function, the gradient, and the Hessian

    """
    n = len(start)
    gradient = np.zeros(n)
    dx = np.zeros(n)
    y = np.zeros(n)
    iteration = 0
    x = start
    step0 = step
    hartree_converged = True

    while iteration < max_iteration:
        iteration += 1

        gradient0 = np.copy(gradient)
        dx, F, gradient, hessian = __newton_raphson_step(func, x, step)
        ihessian = np.linalg.inv(hessian)
        __vca_accuracy_warning(accur, ihessian)

        if hartree != None:
            hartree_converged = True
            for C in hartree:
                C.update(pr=True)
                hartree_converged  = hartree_converged and C.converged()

        if np.linalg.norm(gradient) < gtol and hartree_converged:
            if root:
                print('convergence on gradient after ', iteration, ' iterations')
            break


        if max_iter_diff is not None:
            norm_dx = np.linalg.norm(dx)
            if norm_dx > max_iter_diff:
                ratio = max_iter_diff / norm_dx
                dx *= ratio
                s1 = 'Norm of NR iteration variation dx is out of bounds: {} > {}'.format(norm_dx, max_iter_diff)
                s2 = 'Readjusting: dx = dx * {:.4}%'.format(ratio * 100)
                if root:
                    print(s1, '\n', s2)
                    
        x += dx

        #redefining the steps
        step_multiplier = 2.0
        for i in range(n):
            step[i] = np.abs(dx[i])
            if step[i] > step0[i]:
                step[i] = 2.0*step0[i]
            if step[i] < step_multiplier*accur[i]:
                step[i] = step_multiplier*accur[i]

        if root:
            print('NR iteration no ', iteration, '\t x = ', x, '\t steps = ', step, '\t dx = ', dx)

        converged = True
        for i in range(n):
            if np.abs(dx[i]) > accur[i]:
                converged = False
                break
        if converged and hartree_converged:
            if root:
                print('convergence on position after ', iteration, ' iterations')
            break

    if iteration == max_iteration:
        raise pyqcm.TooManyIterationsError(max_iteration)

    return x, gradient, ihessian

################################################################################
# Newton-Raphson step

def __newton_raphson_step(func=None, x=None, step=None):
    """Performs a Newton-Raphson step
    
    :param func: a function of N variables
    :param [float] x: the current value of the parameters (changed by the function)
    :param [float] step: the steps used to computed the numerical second derivatives
    :returns ([float], float, [float], [[float]]): the change in position, the value of the function, the gradient, and the Hessian

    """

    F, gradient, hessian = __NR_Hessian(func, x, step)
    ihessian = np.linalg.inv(hessian)
    dx = -np.dot(ihessian, gradient)

    return dx, F, gradient, ihessian


################################################################################
# checking consistency of accuracy with effective precision

def __vca_accuracy_warning(accur, ihessian):

    accur_SEF = pyqcm.get_global_parameter('accur_SEF')
    inv_der2 = np.diagonal(ihessian)
    inv_der2 = np.sqrt(accur_SEF*np.abs(inv_der2))
    print('effective VCA precision: ', inv_der2)
    for i in range(len(accur)):
        if inv_der2[i] > accur[i]:
            print('WARNING : effective accuracy of variable {:d} lower than what is required. Maybe lower required accuracy (i.e. increase "accur") or lower "accur_SEF"'.format(i+1))
    

################################################################################
# Newton-Raphson in 1D

def __altNR(func=None, start=None, step=None, accur=None, max=10, gtol=1e-4, max_iteration=30, max_iter_diff=None, hartree=None):
    """Performs a Newton-Raphson procedure in one variable by reusing old points as much as possible
    
    :param func: a function of 1 variable
    :param [float] start: the starting value
    :param [float] step: the first step 
    :param [float] accur: the required accuracy
    :param [float] max: maximum absolute value of the parameter
    :param float gtol: the gradient tolerance (gradient must be smaller than gtol for convergence)
    :param int max_iterations:  maximum number of iterations, beyond which an exception is raised
    :param float max_iter_diff: optional maximum value of the maximum step
    :param (class hartree) hartree: Hartree approximation couplings (see pyqcm/hartree.py)
    :returns (float, float, float): the value of the function, the gradient, and the 2nd derivative

    """

    # initial three points
    X = np.zeros(3)
    Y = np.zeros(3)
    X[0] = start[0]-step[0]
    X[1] = start[0]
    X[2] = start[0]+step[0]
    Y[0] = func([X[0]])
    Y[1] = func([X[1]])
    Y[2] = func([X[2]])

    iter = 0
    xp0 = 1e9
    accur_SEF = pyqcm.get_global_parameter('accur_SEF')
    while iter < max_iteration:
        iter += 1
        C = np.polyfit(X,Y,2)
        xp = -0.5*C[1]/C[0]

        der1 = 2*C[0]*xp0 + C[1]
        der2 = 2*C[0]
        dx = np.sqrt(accur_SEF/np.abs(der2))
        if iter > 1:
            print('---> X :', X, '\tder1 = {:.6g}\tder2 = {:.3g}'.format(der1, der2), '\teffective precision: {:.3g}'.format(dx))
            if dx > accur:
                print('WARNING : effective accuracy lower than what is required. Maybe lower required accuracy (i.e. increase "accur") or lower "accur_SEF"')
        if np.abs(der1) < gtol:
            print('VCA 1D : convergence on gradient')
            break
        I = np.argmax(np.abs(X-xp))
        X[I] = xp
        Y[I] = func([X[I]])
        if iter > 1 and np.abs(xp-xp0) < accur:
            print('VCA 1D : convergence on position')
            break
        xp0 = xp
    if iter == max_iteration:
        raise pyqcm.TooManyIterationsError(max_iteration)

    return np.array([xp]), np.array([der1]), np.array([[1.0/der2]])

################################################################################
# minimax method

def __minimax(names, var_max_start, start=None, step=None, accur=None, max=10,  max_iteration=30, hartree=None):
    """Applies a minimization of a subset of variables and a maximization on the remainder
    
    :param ['str'] names : names of variational parameters (minimal ones first, then maximal ones)
    :param int var_max_start : label of the first maximal variational parameter (= number of minimal variational parameters)
    :param [float] start: the starting values
    :param [float] step: the steps used to computed the numerical second derivatives
    :param [float] accur: the required accuracy for each variable
    :param [float] max: maximum absolute value of each parameter
    :param int max_iterations:  maximum number of iterations, beyond which an exception is raised
    :returns [float]: the value of the variables

    """

    ftol = 2*pyqcm.qcm.get_global_parameter('accur_SEF')
    nvar_max = len(names) - var_max_start
    nvar_min = var_max_start
    steps_min = step[0:nvar_min]
    steps_max = step[nvar_min:nvar_min+nvar_max]
    iter = 0
    nvar_tot = nvar_min+nvar_max

    def F_min(x):
        global SEF_eval
        for i in range(nvar_min): 
            pyqcm.set_parameter(names[i], x[i])
        print('(min) x = ', x) # new
        SEF_eval += 1
        pyqcm.new_model_instance()
        return pyqcm.Potthoff_functional(hartree)
    def F_max(x):
        global SEF_eval
        for i in range(nvar_max): 
            pyqcm.set_parameter(names[i+nvar_min], x[i])
        print('(max) x = ', x) # new
        SEF_eval += 1
        pyqcm.new_model_instance()
        return -pyqcm.Potthoff_functional(hartree)


    X0 = np.array(start)

    while iter < max_iteration:
        P = pyqcm.parameters()
        x_min = [P[names[i]] for i in range(nvar_min)]
        x_max = [P[names[i]] for i in range(nvar_min, nvar_tot)]

        if iter > 0:
            steps_min = np.abs(diff[0:nvar_min])
            steps_max = np.abs(diff[nvar_min:nvar_min+nvar_max])

        nvar = nvar_min
        steps = steps_min
        initial_simplex = np.zeros((nvar+1,nvar))
        for i in range(nvar+1):
            initial_simplex[i, :] = x_min
        for i in range(nvar):
            initial_simplex[i+1, i] += steps[i]
        if type(accur) == list:
            accur = accur[0]
        solution = minimize(F_min, x_min, method='Nelder-Mead', options={'maxfev':100, 'xatol': accur, 'fatol':ftol, 'initial_simplex': initial_simplex, 'adaptive': True, 'disp':True})
        iter_done = solution.nit
        sol_min = solution.x
        for i in range(nvar_min):
            pyqcm.set_parameter(names[i], sol_min[i])
        
        nvar = nvar_max
        steps = steps_max
        initial_simplex = np.zeros((nvar+1,nvar))
        for i in range(nvar+1):
            initial_simplex[i, :] = x_max
        for i in range(nvar):
            initial_simplex[i+1, i] += steps[i]
        if type(accur) == list:
            accur = accur[0]
        solution = minimize(F_max, x_max, method='Nelder-Mead', options={'maxfev':100, 'xatol': accur, 'fatol':ftol, 'initial_simplex': initial_simplex, 'adaptive': True, 'disp':True})
        iter_done = solution.nit
        sol_max = solution.x
        for i in range(nvar_max):
            pyqcm.set_parameter(names[i+nvar_min], sol_max[i])

        sol = np.concatenate((sol_min, sol_max))

        hartree_converged = True
        if hartree != None:
            hartree_converged = True
            for C in hartree:
                C.update(pr=True)
                hartree_converged  = hartree_converged and C.converged()

        diff = sol - X0
        X0 = sol
        iter += 1
        converged = True
        for i in range(nvar_tot):
            if np.abs(diff[i]) > accur: converged = False

        if converged and hartree_converged:
            return sol
    
    raise(pyqcm.TooManyIterationsError(max_iteration))
    return sol

################################################################################
# PUBLIC FUNCTIONS
################################################################################
# performs the VCA
SEF_eval = 0

def vca(
    var2sef=None,
    varia=None, 
    start=None, 
    steps=None, 
    accur=None, 
    max=None,
    file="vca.tsv",
    accur_grad=1e-6, 
    max_iter=30, 
    max_iter_diff=None, 
    method='NR', 
    hartree=None, 
    hartree_self_consistent=False,
    symmetrized_operator=None,
    var_max_start = None
):
    """Performs a VCA with the QN or NR method
    
    :param var2sef: function that converts variational parameters to model parameters
    :param str or [str] varia: variational parameters (list or tuple)
    :param float or [float] start: starting values
    :param float or [float] steps: initial steps
    :param float or [float] accur: accuracy of parameters (also step for 2nd derivatives)
    :param float or [float] max: maximum values that are tolerated
    :param float accur_grad: max value of gradient for convergence
    :param int max_iter: maximum number of iterations in the procedure
    :param float max_iter_diff: optional maximum value of the maximum step in the quasi-Newton method
    :param str method: method used to optimize ('SYMR1', 'NR', 'BFGS', 'altNR', 'Nelder-Mead', 'COBYLA', 'Powell', 'CG', 'minimax')
    :param (class hartree) hartree: Hartree approximation couplings (see pyqcm/hartree.py)
    :param boolean hartree_self_consistent: True if the Hartree approximation is treated in the self-consistent, rather than variational, way.
    :param str symmetrized_operator: name of an operator wrt which the functional must be symmetrized
    :param int var_max_start: label of the first variable for which the function is a maximum (minimal vars first, maximal vars last)
    :return: None
    
    """
    global SEF_eval
    # type and length checks
    if is_sequence(varia) == False:
        if type(varia) != str:
            raise ValueError('argument varia of vca() must be a string or a sequence of strings')
        single = True
        varia = [varia]
        if start != None:
            if type(start) != float and type(start) != int:
                raise ValueError('argument start of vca() must be a float or a sequence of float')
            start = [start]
        if type(steps) != float:
            raise ValueError('argument steps of vca() must be a float')
        if type(accur) != float:
            raise ValueError('argument accur of vca() must be a float')
        if type(max) != float and type(max) != int:
            raise ValueError('argument max of vca() must be a float')
        steps = [steps]
        accur = [accur]
        max = [max]
        nvar = 1
    else:
        nvar = len(varia)  # number of variational parameters
        if start != None:
            if len(start) != nvar:
                raise ValueError('argument max of vca() must contain {:d} elements'.format(nvar))
        if type(steps) != list or type(accur) != list or type(max) != list:
            raise ValueError('arguments steps, accur and max of vca() must be lists of {:d} elements each'.format(nvar))
        if len(steps) != nvar or len(accur) != nvar or len(max) != nvar:
            raise ValueError('arguments steps, accur and max of vca() must be lists of {:d} elements each'.format(nvar))

    if type(hartree) is not list and hartree is not None:
        hartree = [hartree] # fixes possible type error

    global first_time
    pyqcm.new_model_instance()
    L = pyqcm.model_size()[0]
    pyqcm.first_SEF = True

    hartree_self = None
    if hartree_self_consistent:
        hartree_self = hartree


    if varia is None:
        print('missing argument varia : variational parameters must be specified')
        raise pyqcm.MissingArgError('varia')


    if max is None:
        max = 10*np.ones(nvar)
    elif len(max) != nvar:
        print('the argument "max" should have ', nvar, ' components')
        raise pyqcm.MissingArgError('max')

    if start is None:
        if var2sef != None:
            raise ValueError('the start argument is missing in vca(). Should be specified if var2sef is not None')
        start = [0.0]*nvar
        P = pyqcm.parameters()
        for i,v in enumerate(varia):
            start[i] = P[v]

    elif len(start) != nvar:
        print('the argument "start" should have ', nvar, ' components')
        raise pyqcm.MissingArgError('start')

    if accur is None:
        accur = 1e-4*np.ones(nvar)

    if steps is None:
        steps = 10*np.array(accur)
    elif len(steps) != nvar:
        print('the argument "steps" should have ', nvar, ' components')
        raise pyqcm.MissingArgError('steps')

    SEF_eval = 0
    def var2x(x):
        for i in range(len(x)):
            if np.abs(x[i]) > max[i]:
                raise pyqcm.OutOfBoundsError(variable=varia[i])
        global SEF_eval
        if var2sef is None:
            for i in range(len(varia)): 
                pyqcm.set_parameter(varia[i], x[i])
            print('x = ', x) # new
        else:
            var2sef(x)    
        SEF_eval += 1
        pyqcm.new_model_instance()
        return pyqcm.Potthoff_functional(hartree, symmetrized_operator=symmetrized_operator)
        
    if hartree is None:
        pyqcm.banner('VCA procedure, method {:s}'.format(method), '*')
    else:
        pyqcm.banner('VCA procedure, (combined with Hartree procedure) method {:s}'.format(method), '*')
    var_val = pyqcm.__varia_table(varia,start)
    print(var_val)

    scipy_minimization = False
    if method == 'Nelder-Mead' or method == 'COBYLA' or method == 'Powell'  or method == 'CG'  or method == 'BFGS'  or method == 'minimax':
        scipy_minimization = True

    if scipy_minimization and hartree_self_consistent:
        raise ValueError('Hartree self-consistency not allowed when using SciPy minimization methods in VCA')

    ftol = 2*pyqcm.qcm.get_global_parameter('accur_SEF')
    iH = None

    #------------------------------------- SWITCH ACCORDING TO METHOD -----------------------------------------
    try:
        if method == 'altNR':
            if nvar != 1:
                raise ValueError('vca method altNR only works with one variable')
            sol, grad, iH = __altNR(var2x, start, steps, accur, max, accur_grad, max_iteration=max_iter, max_iter_diff=max_iter_diff, hartree=hartree_self)  # special Newton-Raphson process in 1 variable

        elif method == 'NR' :
            sol, grad, iH = __newton_raphson(var2x, start, steps, accur, max, accur_grad, max_iteration=max_iter, max_iter_diff=max_iter_diff, hartree=hartree_self)  # Newton-Raphson process

        elif method == 'SYMR1' :
            sol, grad, iH = __quasi_newton(var2x, start, steps, accur, max, accur_grad, False, max_iteration=max_iter, max_iter_diff=max_iter_diff, hartree=hartree_self)  # quasi-Newton process

        elif method == 'BFGS2' :
            sol, grad, iH = __quasi_newton(var2x, start, steps, accur, max, accur_grad, True, max_iteration=max_iter, max_iter_diff=max_iter_diff, hartree=hartree_self)  # quasi-Newton process

        elif method == 'Nelder-Mead':
            initial_simplex = np.zeros((nvar+1,nvar))
            for i in range(nvar+1):
                initial_simplex[i, :] = start
            for i in range(nvar):
                initial_simplex[i+1, i] += steps[i]
            if type(accur) == list:
                accur = accur[0]
            solution = minimize(var2x, start, method='Nelder-Mead', options={'maxfev':3*max_iter, 'xatol': accur, 'fatol':ftol, 'initial_simplex': initial_simplex, 'adaptive': True, 'disp':True})
            iter_done = solution.nit
            sol = solution.x

        elif method == 'Powell':
            accur = accur[0]
            solution = minimize(var2x, start, method='Powell', tol = ftol, options={'xtol': accur, 'ftol': ftol, 'disp':True})
            iter_done = solution.nit
            sol = solution.x
            if type(sol) != list:
                sol = [sol]

        elif method == 'CG':
            solution = minimize(var2x, start, method='CG', jac=False, tol = ftol, options={'eps':accur, 'maxiter':3*max_iter, 'disp':True})
            iter_done = solution.nit
            sol = solution.x

        elif method == 'BFGS':
            solution = minimize(var2x, start, method='BFGS', jac=False, tol = ftol, options={'eps':accur, 'disp':True})
            iter_done = solution.nit
            sol = solution.x
            iH = solution.hess_inv

        elif method == 'COBYLA':
            solution = minimize(var2x, start, method='COBYLA', options={'rhobeg':steps[0], 'maxiter':3*max_iter, 'tol': ftol, 'disp':True})
            iter_done = solution.nfev
            sol = solution.x

        elif method == 'minimax':
            sol = __minimax(varia, var_max_start, start, steps, accur, max, max_iteration=max_iter, hartree=hartree)
        else:
            raise ValueError('method {:s} unknown in VCA'.format(method))

    #----------------------------------------------------------------------------------------------------------


    except pyqcm.OutOfBoundsError as E:
        print(E)
        raise pyqcm.OutOfBoundsError(E.variable)

    except pyqcm.TooManyIterationsError as E:
        print('VCA method failed to converge after ', E.max_iteration, ' iterations')
        raise pyqcm.TooManyIterationsError(E.max_iteration)

    omega = var2x(sol)  # final, converged value
    if root:
        if iH is not None: 
            H = np.linalg.inv(iH)  # Hessian at the solution (inverse of iH)
        print('saddle point = ', sol)
        if scipy_minimization is False: 
            print('gradient = ', grad)
            print('second derivatives :', np.diag(H))
            print('eigenvalues of Hessian :', np.linalg.eigh(H)[0])
        print('computing properties of converged solution...')
        print('omega = ', omega)
    ave = pyqcm.averages()


    # writes the solution in the standard file
    if root:
        val = method + '\t{:d}\t'.format(SEF_eval)
        des = 'method\tSEF_eval\t'
        if iH is not None: 
            for i in range(nvar):
                val += '{:.4g}\t'.format(H[i, i])
            for i in range(nvar):
                des += '2der_' + varia[i] + '\t'
        if hartree != None:
            val += '{:.8g}\t'.format(omega)
            des += 'omegaH\t'
        # pyqcm.write_summary(file, first = first_time, suppl_descr = des, suppl_values = val)
        pyqcm.write_summary(file, suppl_descr = des, suppl_values = val)
        first_time = False

    if root:
        pyqcm.banner('VCA ended normally', '*')

    if scipy_minimization is False:
        return sol, 1.0/np.diag(iH)
    else:
        return sol, None

################################################################################
def plot_sef(param, prm, file="sef.tsv", accur_SEF=1e-4, hartree=None, show=True, symmetrized_operator=None):
    """Draws a plot of the Potthoff functional as a function of a parameter param taken from the list prm. The results are going to be appended to 'sef.tsv'
    
    :param str param: name of the parameter (independent variable)
    :param [float] prm: list of values of the parameter
    :param float accur_SEF: precision of the computation of the self-energy functional
    :param (class hartree) hartree: Hartree approximation couplings (see pyqcm/hartree.py)
    :param boolean show: if True, the plot is shown on the screen.
    :returns: None

    """
    L = pyqcm.model_size()[0]
    pyqcm.first_SEF = True

    if type(file) != str:
	    raise TypeError('the argument "file" of plot_sef() must be a string')
    if type(accur_SEF) != float:
	    raise TypeError('the argument "accur_SEF" of plot_sef() must be a float')
    if type(param) != str:
	    raise TypeError('the argument "param" of plot_sef() must be a string, the name of a parameter')

    if type(hartree) is not list and hartree is not None:
        hartree = [hartree]

    pyqcm.set_global_parameter('accur_SEF', accur_SEF)
    omega = np.empty(len(prm))
    for i in range(len(prm)):
        pyqcm.set_parameter(param, prm[i])
        pyqcm.new_model_instance()
        # print(pyqcm.parameter_set(opt='report'))
        # print('.'*80, '\n', pyqcm.cluster_parameters())
        omega[i] = pyqcm.Potthoff_functional(hartree, file=file, symmetrized_operator=symmetrized_operator)

        print("omega(", prm[i], ") = ", omega[i])
    
    if show:
        import matplotlib.pyplot as plt
        plt.xlim(prm[0], prm[-1])
        plt.plot(prm, omega, 'b-')
        plt.xlabel(param)
        plt.ylabel(r'$\Omega$')
        plt.axhline(omega[0], c='r', ls='solid', lw=0.5)
        plt.title(pyqcm.parameter_string())
        plt.show()




################################################################################
def plot_GS_energy(param, prm, clus=0, file=None, plt_ax=None, **kwargs):
    """Draws a plot of the ground state energy as a function of a parameter param taken from the list `prm`. The results are going to be appended to 'GS.tsv'
    
    :param str param: name of the parameter (independent variable)
    :param [float] prm: list of values of the parameter
    :param int clus: label of the cluster (starts at 0)
    :param str file: if not None, saves the plot in a file with that name
    :param plt_ax: optional matplotlib axis set, to be passed when one wants to collect a subplot of a larger set
    :param kwargs: keyword arguments passed to the matplotlib 'plot' function
    :returns: None

    """
    import matplotlib.pyplot as plt
    if plt_ax is None:
        plt.figure()
        plt.gcf().set_size_inches(13.5/2.54, 9/2.54)
        ax = plt.gca()
    else:
        ax = plt_ax

    omega = np.empty(len(prm))
    for i in range(len(prm)):
        pyqcm.set_parameter(param, prm[i])
        pyqcm.new_model_instance()
        omega[i] = pyqcm.ground_state()[clus][0]
        print("omega(", prm[i], ") = ", omega[i])

        # writing the parameters in a progress file
        f = open('GS.tsv', 'a')
        des, val = pyqcm.properties()
        if i == 0:
            f.write('\n\n')
            f.write(des + '\n')
        f.write(val + '\n')
        f.close()

    
    ax.axhline(omega[0], c='r', ls='solid', lw=0.5)
    ax.plot(prm, omega, 'b-')
    ax.set_xlim(prm[0], prm[-1])
    if plt_ax is None:
        ax.set_xlabel(param)
        ax.set_ylabel('GS energy')
        ax.set_title(pyqcm.parameter_string())

    if file is not None:
        plt.savefig(file)
        plt.close()
    elif plt_ax is None:
        plt.show()



################################################################################
# detects a continuous phase transition

def __transition(varia, P, bracket, step=0.001, verb=False, symmetrized_operator=None):
    """Detects a transition as a function of external parameter param by looking at the equality between 
    :math:`\Omega(h=s)` and :math:`\Omega(h=0)` where *h* is a single variational parameter (Weiss field)
    and *s* is a step. 
    
    :param str varia: name of the variational parameter
    :param str P: name of the control parameter (the parameter that controls the transition)
    :param (float,float) bracket: bracketing values for parameter P that enclose the transition
    :param float step: small, but finite value *s* of the Weiss field 
    :param boolean verb: If True, prints progress
    :returns: the value of P at the transition

    """

    from scipy.optimize import brentq

    def F(x):
        pyqcm.set_parameter(P, x)
        pyqcm.set_parameter(varia, step)
        pyqcm.new_model_instance()
        Om1 = pyqcm.Potthoff_functional(symmetrized_operator=symmetrized_operator)
        pyqcm.set_parameter(varia, 1e-8)
        pyqcm.new_model_instance()
        Om0 = pyqcm.Potthoff_functional(symmetrized_operator=symmetrized_operator)
        if verb:
            print(P, '= ', x, '\tdelta Omega = ', Om1-Om0)
        return Om1-Om0


    x0, r = brentq(F, bracket[0], bracket[1], maxiter=100, full_output=True, disp=True)

    print(r.flag)
    if not r.converged:
        raise RuntimeError('the root finding routine could not find a solution!')
        
    return x0

################################################################################
# detects a continuous phase transition (meta function with loop over control parameter)

def transition_line(varia, P1, P1_range, P2, P2_range, delta, verb=False):
    """Builds the second-order transition line as a function of a control parameter P1. The results are written in the file `transition.tsv`

    :param str varia: variational parameter
    :param str P1: control parameter
    :param [float] P1_range: an array of values of P1
    :param str P2: dependent parameter
    :param (float) P2_range: 2-uple of values of P2 that bracket the transition
    :param boolean verb: If True, prints progress
    :param float delta: at each step, the new bracket for P2 will be P2c +/- delta, P2c being the previous critical value 
    :return: None

    """

    assert type(varia) == str, 'the first argument of transition_line must be the name of a single variational parameter'
    assert type(P1) == str, 'the second argument of transition_line must be the name of a single control parameter'
    assert type(P2_range) == list, 'the argument P2_range of transition_line must be a list of 2 floats'

    trans = np.empty((len(P1_range), 2))
    for i, p1 in enumerate(P1_range):
        pyqcm.set_parameter(P1, p1)
        p2c = __transition(varia, P2, P2_range, step=0.001, verb=verb)
        trans[i,0] = p1
        trans[i,1] = p2c
        P2_range[0] = p2c - delta
        P2_range[1] = p2c + delta
        print('---> transition at ', P1, ' = ', p1, '\t', P2, ' = ', p2c)
        np.savetxt('transition.tsv', trans[0:i+1, :], delimiter='\t', header=P1+'\t'+P2)
