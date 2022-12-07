import pyqcm
from pyqcm.vca import *
# pyqcm.set_global_parameter("nosym")
import model_1D_4



class subsystem:
    def __init__(self, basis, A, B):
        self.basis = basis
        self.A = A
        self.B = B
        self.L = A.shape[0]+B.shape[0]  # total number of sites
        self.maskA = 0
        for i in range(A.shape[0]):
            self.maskA += 1 << A[i]
            self.maskA += 1 << (A[i]+self.L)
        self.maskB = 0
        for i in range(B.shape[0]):
            self.maskB += 1 << B[i]
            self.maskB += 1 << (B[i]+self.L)
        self.maskA = np.uint(self.maskA)
        self.maskB = np.uint(self.maskB)

    def transpose(self, I, Ip):
        As = self.basis[I] & self.maskA
        Bs = self.basis[I] & self.maskB
        Asp = self.basis[Ip] & self.maskA
        Bsp = self.basis[Ip] & self.maskB
        K = np.where(self.basis == As + Bsp)
        Kp = np.where(self.basis == Asp + Bs)
        return K[0][0], Kp[0][0]

    def transpose_matrix(self, A):
        At = np.empty(A.shape)
        for I in range(basis.shape[0]):
            for J in range(I+1):
                Ip, Jp = S.transpose(I,J)
                At[Ip, Jp] = A[I,J]
                At[Jp, Ip] = A[J,I]
        return At





pyqcm.set_global_parameter("verb_ED")

pyqcm.set_target_sectors(['R0:S0'])
pyqcm.set_parameters("""
t=1
U=4
mu = 2
D = 0.4
""")
pyqcm.new_model_instance()
sites = [0,1]
rho, basis = pyqcm.density_matrix(sites)
print(np.real(rho))
print('trace of rho : ', np.trace(rho))

L = len(sites)
print("basis of subsystem:\n")
for i in range(basis.shape[0]):
    print('|' + np.binary_repr(basis[i], 2*L)+'>')


S = subsystem(basis, np.array([0]), np.array([1]))

# for I in range(basis.shape[0]):
#     for J in range(basis.shape[0]):
#         Ip, Jp = S.transpose(I,J)
#         print('({:d}, {:d}) --> ({:d}, {:d})'.format(I,J,Ip,Jp))

rhoT = S.transpose_matrix(rho)

print('spectrum of rhoT:\n', np.linalg.eigh(rhoT)[0])

        


    