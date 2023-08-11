from sympy import *
import numpy as np

p_k, p_km1, p_kp1, p_kref, p_kp2= symbols('p_k p_km1 p_kp1 p_kref p_kp2')
m1, m2, m3 = symbols('m1 m2 m3')

expr_smooth = (2*p_k - p_km1 - p_kp1)**2
expr_deviation = (m1*p_k + m2*p_kp1 + m3*p_kp2 - p_kref)**2

sp_smooth = expand(expr_smooth)
sp_deviation = expand(expr_deviation)
sp_cost = expand(expr_smooth+expr_deviation)

print(sp_smooth)
# print('=')
# print('=')
# print(sp_cost)


x = np.array([p_km1, p_k, p_kp1]).reshape((1,3))
pp = np.array([[ 1, -2, 1]
               ,[-2, 4, -2]
               ,[1, -2, 1]])
y = x @ pp @ x.T
print(expand(y.item()))
# 4*p_k**2 - 4*p_k*p_km1 - 4*p_k*p_kp1 + p_km1**2 + 2*p_km1*p_kp1 + p_kp1**2

print('###############')
print(sp_deviation)

#   m1**2*p_k**2   + 2*m1*m2*p_k*p_kp1   + 2*m1*m3*p_k*p_kp2 - 2*m1*p_kref*p_k
# + m2**2*p_kp1**2 + 2*m2*m3*p_kp1*p_kp2 - 2*m2*p_kref*p_kp1
# + m3**2*p_kp2**2 - 2*m3*p_kref*p_kp2   + p_kref**2
