import pyqcm
from model_1D import model1D
from scipy.optimize import brentq

L = 4 # length of 1D system
model = model1D(L, sym=True)

model.set_target_sectors(['R0:N4:S0'])
model.set_parameters("""
t=1
U = 4
mu = 4
""")

target = 1.05
bracket = (2, 6)

def task(mu):
	model.set_parameter('mu', mu)
	I = pyqcm.model_instance(model)
	n = I.averages()['mu']
	print('mu = {:g}, n = {:g}'.format(I.parameters()['mu'], n))
	return n-target


x0, r = brentq(task, bracket[0], bracket[1], xtol=1e-4, maxiter=100, full_output=True, disp=True)

print(r.flag)
if not r.converged:
	raise RuntimeError('the root finding routine could not find a solution!')
