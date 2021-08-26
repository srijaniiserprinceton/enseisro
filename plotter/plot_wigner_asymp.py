import numpy as np
from enseisro import misc_functions as FN
import py3nj
from scipy.special import lpmn
from math import factorial
from math import floor
from math import ceil
from scipy.integrate import simps
import matplotlib.pyplot as plt 
plt.ion()

fac_vec = np.vectorize(factorial)


# checking the orthonormality of legendre polynomials
z = np.linspace(-1,1,10000)

ell = 2
m = 1

leg1 = np.zeros_like(z)
for i in range(len(leg1)):
    leg1[i] = lpmn(m, ell, z[i])[0][-1,-1]

check_orth = simps(leg1 * leg1, x = z)
orth_fac = (2./(2*ell+1)) * (factorial(ell+m)/factorial(ell-m))

print("Orthonormality: ", check_orth/orth_fac)

# parameters
N = 20
s = 2
ell = np.linspace(1,N,N).astype(int)
ell_ = np.linspace(1,N,N).astype(int)

print(ell)
m = 1

# the 2 is multiplied since that is how py3nj is written
wigner_val = py3nj.wigner3j(2*s, 2*ell_, 2*ell, 0, 2*m, -2*m)

# asymptotic approx from legendre polynomials
asymp_val_arr = np.zeros(N)
for i in range(N):
    l = ell[i]
    l_ = ell_[i]
    legendre_value = lpmn(l_-l,s,1.0*m/l)[0][0,-1]
    legendre_coeff =  ((-1)**(l_ + m))/np.sqrt(2 * l) * np.sqrt(fac_vec(s - l_ + l) / fac_vec(s + l_ - l))
    asymp_val_arr[i] = legendre_coeff * legendre_value

error_percent = (wigner_val - asymp_val_arr)/wigner_val * 100

plt.figure()
plt.plot(ell, error_percent, 'k')
plt.plot(ell, error_percent, 'or')
plt.xticks(ell[::2])
plt.xlim([0,N+1])
plt.xlabel('$\ell$')
plt.grid(True)
plt.tight_layout()
plt.savefig('Exact_vs_asymptotic.pdf')

'''
##################################################################################################################
# this section checks the equations (A15) and (A17) in Vorontsov 2007
# these checks are important to see if the Wigners and Legendre polynomials
# have desired normalization.

# computes the Schou polynomials
def schou_poly(s, ell, m):
    coeff = ((-1)**(ell+m)) * np.sqrt(1. * factorial(2*ell-s) * factorial(2*ell + s + 1)) / (2. * factorial(2*ell - 1))
    wigner = py3nj.wigner3j(2*s, 2*ell, 2*ell, 0, 2*m, -2*m)
    
    return (coeff * wigner)

# checks the orthogonality of the Schou polynomials
def check_schou_P_orth(s, s_, ell):
    m_arr = np.arange(-ell, ell+1)
    
    # initializing the Schou polynomials
    schou_poly_s  = np.zeros(2 * ell + 1)
    schou_poly_s_ = np.zeros(2 * ell + 1)

    for i, m in enumerate(m_arr):
        schou_poly_s[i]  = schou_poly(s, ell, m)
        schou_poly_s_[i] = schou_poly(s_, ell, m)

    # summing the product to check orthogonality
    sum_prod = np.sum(schou_poly_s * schou_poly_s_)

    print("Schou orthogonality: ", sum_prod)

# checks if (A17) is true
s, ell, m = 1, 1, 1
schou_P = schou_poly(1,1,1)
print("Schou polynomial as in A17 of V07: ", schou_P)

# checks the orthogonality of Schou polynomials
check_schou_P_orth(5,5,10)
############################################################################################################
'''
