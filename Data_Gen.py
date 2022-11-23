import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.linalg import circulant, toeplitz


def coefficients(last_ind, frac_ord):
    # last_ind the last index of coefficients
    # alpha the fractional order must belong to (0,1)
    j = np.arange(0, 5)
    aux = (-1)**j*gamma(2*frac_ord+1)/(gamma(1+frac_ord+j)*gamma(1+frac_ord-j))
    a_before = aux[-1]
    for n in range(4, last_ind):
        a_next = a_before*(n-frac_ord)/(n+1+frac_ord)
        aux = np.append(aux,a_next)
        a_before = a_next
    return aux


num_potential = 4
sigma = np.random.rand(num_potential)*(0.05-0.0125)+0.0125
# center = np.random.rand(num_potential)*0.15+np.array([0.05, 0.3, 0.55, 0.8])
center = np.random.rand(num_potential)


def potential(x):
        aux = 0
        for i in np.arange(num_potential):
            aux += 100*np.exp(-0.5*((x-center[i])/sigma[i])**2)
        return aux


M1 = 10
M = 640
M_pad = int((M-M1+1)/2)
h = 1/(M+1)
alpha = 0.5
vert = np.linspace(-1, 1, 2*M+2)
node_bd = vert[1:M+1]
node_free = vert[M+1:-1]
stiff_row = coefficients(2*M-1, alpha)
stiff_mat = toeplitz(stiff_row)
q = potential(node_free)
q_diag = np.diag(q)
dirichlet_bd = (node_bd+1)**2*(node_bd+0.5)**2*200
source = node_free*(1-node_free)*400
stiff = np.eye(2*M)
stiff[M:, ] = stiff_mat[M:, ]
stiff[M:, M:] = stiff[M:, M:]+q_diag*h**(2*alpha)
rhs = np.zeros(2*M)
rhs[0:int(M/2)] = dirichlet_bd[0:int(M/2)]
rhs[M:] = source*h**(2*alpha)
u_sol = np.linalg.solve(stiff, rhs)
# DtN
stiff_row = coefficients(3*M-1, alpha)
stiff_mat = toeplitz(stiff_row)/(h**(2*alpha))
u = np.zeros(3*M)
u[0:2*M] = u_sol
plt.plot(u)
plt.show()
dtn = np.matmul(stiff_mat, u)
dtn = dtn[-M:-M+50]
plt.plot(dtn)
plt.show()




