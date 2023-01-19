from pyqcm import *
set_global_parameter("nosym")
new_cluster_model('2x2', 4, 0, generators=None, bath_irrep=False)
add_cluster('2x2', [0, 0, 0], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], ref = 0)
add_cluster('2x2', [0, 0, 1], [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], ref = 0)
lattice_model('2C', [[2, 0, 0], [0, 2, 0]], None)
interaction_operator('U', orbitals=(1,1))
hopping_operator('t', [1, 0, 0], -1, orbitals=(1,1))
hopping_operator('t', [0, 1, 0], -1, orbitals=(1,1))
hopping_operator('tp', [1, 1, 0], -1, orbitals=(1,1))
hopping_operator('tp', [1, -1, 0], -1, orbitals=(1,1))
hopping_operator('tpp', [2, 0, 0], -1, orbitals=(1,1))
hopping_operator('tpp', [0, 2, 0], -1, orbitals=(1,1))
interaction_operator('U', orbitals=(2,2))
hopping_operator('t', [1, 0, 0], -1, orbitals=(2,2))
hopping_operator('t', [0, 1, 0], -1, orbitals=(2,2))
hopping_operator('tp', [1, 1, 0], -1, orbitals=(2,2))
hopping_operator('tp', [1, -1, 0], -1, orbitals=(2,2))
hopping_operator('tpp', [2, 0, 0], -1, orbitals=(2,2))
hopping_operator('tpp', [0, 2, 0], -1, orbitals=(2,2))
hopping_operator('tz', [0, 0, 1], -1, orbitals=(1,2))
hopping_operator('tzp', [1, 1, 1], -1, orbitals=(1,2))
hopping_operator('tzp', [1, -1, 1], -1, orbitals=(1,2))
hopping_operator('tzp', [-1, 1, 1], -1, orbitals=(1,2))
hopping_operator('tzp', [-1, -1, 1], -1, orbitals=(1,2))
hopping_operator('tzpp', [2, 0, 1], -1, orbitals=(1,2))
hopping_operator('tzpp', [0, 2, 1], -1, orbitals=(1,2))
hopping_operator('tzpp', [-2, 0, 1], -1, orbitals=(1,2))
hopping_operator('tzpp', [0, -2, 1], -1, orbitals=(1,2))
hopping_operator('mu1', [0, 0, 0], -1, tau=0, orbitals=(1,1))
hopping_operator('mu2', [0, 0, 0], -1, tau=0, orbitals=(2,2))
anomalous_operator('S1', [1, 0, 0], 1, orbitals=(1,1))
anomalous_operator('S1', [0, 1, 0], 1, orbitals=(1,1))
anomalous_operator('S2', [1, 0, 0], 1, orbitals=(2,2))
anomalous_operator('S2', [0, 1, 0], 1, orbitals=(2,2))
anomalous_operator('D1', [1, 0, 0], 1, orbitals=(1,1))
anomalous_operator('D1', [0, 1, 0], -1, orbitals=(1,1))
anomalous_operator('D2', [1, 0, 0], 1, orbitals=(2,2))
anomalous_operator('D2', [0, 1, 0], -1, orbitals=(2,2))
hopping_operator('lambda1', [0, 1, 0], 1, orbitals=(1,1), tau=2, sigma=1)
hopping_operator('lambda1', [1, 0, 0], -1, orbitals=(1,1), tau=2, sigma=2)
hopping_operator('lambda2', [0, 1, 0], 1, orbitals=(2,2), tau=2, sigma=1)
hopping_operator('lambda2', [1, 0, 0], -1, orbitals=(2,2), tau=2, sigma=2)
anomalous_operator('B1y1', [0, 1, 0], 1j, orbitals=(1,1), type='dx')
anomalous_operator('B1x1', [1, 0, 0], 1j, orbitals=(1,1), type='dy')
anomalous_operator('B1y2', [0, 1, 0], 1j, orbitals=(2,2), type='dx')
anomalous_operator('B1x2', [1, 0, 0], 1j, orbitals=(2,2), type='dy')

try:
    import model_extra
except:
    pass		
set_target_sectors(['R0:N2', 'R0:N2'])
set_parameters("""

    U=1e-9
    mu=-0.73
    t=1
    tp=-0.3
    tpp=0.1
    tz=-0.08
    tzp=0.04
    tzpp=-0.02
    lambda1=0.03
    lambda2=-0.03    
""")
set_parameter("U", 1e-09)
set_parameter("lambda1", 0.03)
set_parameter("lambda2", -0.03)
set_parameter("mu", -0.73)
set_parameter("t", 1.0)
set_parameter("tp", -0.3)
set_parameter("tpp", 0.1)
set_parameter("tz", -0.08)
set_parameter("tzp", 0.04)
set_parameter("tzpp", -0.02)

new_model_instance(0)

solution=[None]*2

#--------------------- cluster no 1 -----------------
solution[0] = """
U	1e-09
lambda1	0.03
lambda2	-0.03
mu	-0.73
t	1
tp	-0.3

GS_energy: -1.94 GS_sector: R0:N2:1
GF_format: bl
mixing	2
state
R0:N2	-1.94	1
w	8	8
-0.96999999975	(0,0)	(0,0)	(0,0)	(0,0)	(0.5,0)	(0.5,0)	(0.5,0)	(0.5,0)
-0.96999999975	(-0.5,0)	(-0.5,0)	(-0.5,0)	(-0.5,0)	(0,0)	(0,0)	(0,0)	(0,0)
0.43000000025	(-0.00015889842030443,0)	(-0.00016598879226066,0)	(0.00016598879226067,0)	(0.00015889842030442,0)	(-0.52967610577023,0)	(0.46844761732247,0)	(-0.46844761732247,0)	(0.52967610577023,0)
0.43000000025	(0.47783210190695,0)	(-0.5212259415674,0)	(0.5212259415674,0)	(-0.47783210190695,0)	(1.134088426758e-05,0)	(-9.7853053096905e-06,0)	(9.7853053096953e-06,0)	(-1.1340884267585e-05,0)
0.43000000025	(0.52122591752588,0)	(0.477832073275,0)	(-0.477832073275,0)	(-0.52122591752588,0)	(-0.00017739587298877,0)	(0.00014553257601821,0)	(-0.0001455325760182,0)	(0.00017739587298876,0)
0.43000000025	(-6.1470938719993e-06,0)	(-5.8839100467385e-06,0)	(5.8839100467317e-06,0)	(6.1470938720061e-06,0)	(-0.46844763995368,0)	(-0.52967613552946,0)	(0.52967613552946,0)	(0.46844763995368,0)
3.03000000025	(0.5,0)	(-0.5,0)	(-0.5,0)	(0.5,0)	(0,0)	(0,0)	(0,0)	(0,0)
3.03000000025	(0,0)	(0,0)	(0,0)	(0,0)	(0.5,0)	(-0.5,0)	(-0.5,0)	(0.5,0)

"""

#--------------------- cluster no 2 -----------------
solution[1] = """
U	1e-09
lambda1	0.03
lambda2	-0.03
mu	-0.73
t	1
tp	-0.3

GS_energy: -1.94 GS_sector: R0:N2:1
GF_format: bl
mixing	2
state
R0:N2	-1.94	1
w	8	8
-0.96999999975	(0,0)	(0,0)	(0,0)	(0,0)	(0.5,0)	(0.5,0)	(0.5,0)	(0.5,0)
-0.96999999975	(-0.5,0)	(-0.5,0)	(-0.5,0)	(-0.5,0)	(0,0)	(0,0)	(0,0)	(0,0)
0.43000000025	(-0.00015889842030443,0)	(-0.00016598879226066,0)	(0.00016598879226067,0)	(0.00015889842030442,0)	(-0.52967610577023,0)	(0.46844761732247,0)	(-0.46844761732247,0)	(0.52967610577023,0)
0.43000000025	(0.47783210190695,0)	(-0.5212259415674,0)	(0.5212259415674,0)	(-0.47783210190695,0)	(1.134088426758e-05,0)	(-9.7853053096905e-06,0)	(9.7853053096953e-06,0)	(-1.1340884267585e-05,0)
0.43000000025	(0.52122591752588,0)	(0.477832073275,0)	(-0.477832073275,0)	(-0.52122591752588,0)	(-0.00017739587298877,0)	(0.00014553257601821,0)	(-0.0001455325760182,0)	(0.00017739587298876,0)
0.43000000025	(-6.1470938719993e-06,0)	(-5.8839100467385e-06,0)	(5.8839100467317e-06,0)	(6.1470938720061e-06,0)	(-0.46844763995368,0)	(-0.52967613552946,0)	(0.52967613552946,0)	(0.46844763995368,0)
3.03000000025	(0.5,0)	(-0.5,0)	(-0.5,0)	(0.5,0)	(0,0)	(0,0)	(0,0)	(0,0)
3.03000000025	(0,0)	(0,0)	(0,0)	(0,0)	(0.5,0)	(-0.5,0)	(-0.5,0)	(0.5,0)

"""
read_cluster_model_instance(solution[0], 0)
read_cluster_model_instance(solution[1], 1)
