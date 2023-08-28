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

print('~~~~~~~~~~~~~~~~~~~')
p_k_x, p_km1_x, p_kp1_x, p_k_y, p_km1_y, p_kp1_y = symbols('p_k_x p_km1_x p_kp1_x p_k_y p_km1_y p_kp1_y')
v_p_k = np.array([p_k_x, p_k_y])
v_p_km1 = np.array([p_km1_x, p_km1_y])
v_p_kp1 = np.array([p_kp1_x, p_kp1_y])

expr_smooth_new = ((2*v_p_k - v_p_km1 - v_p_kp1).dot(v_p_kp1 - v_p_km1))**2
tmp1 = (2*v_p_k - v_p_km1 - v_p_kp1)
tmp2 = (v_p_kp1 - v_p_km1)
tmp3 = (2*v_p_k - v_p_km1 - v_p_kp1).dot(v_p_kp1 - v_p_km1)

print(expand(expr_smooth_new))
# -2*p_k_x*p_km1_x + 2*p_k_x*p_kp1_x - 2*p_k_y*p_km1_y + 2*p_k_y*p_kp1_y + p_km1_x**2 + p_km1_y**2 - p_kp1_x**2 - p_kp1_y**2
# 4*p_k_x**2*p_km1_x**2 - 8*p_k_x**2*p_km1_x*p_kp1_x + 4*p_k_x**2*p_kp1_x**2 + 8*p_k_x*p_k_y*p_km1_x*p_km1_y - 8*p_k_x*p_k_y*p_km1_x*p_kp1_y - 8*p_k_x*p_k_y*p_km1_y*p_kp1_x + 8*p_k_x*p_k_y*p_kp1_x*p_kp1_y - 4*p_k_x*p_km1_x**3 + 4*p_k_x*p_km1_x**2*p_kp1_x - 4*p_k_x*p_km1_x*p_km1_y**2 + 4*p_k_x*p_km1_x*p_kp1_x**2 + 4*p_k_x*p_km1_x*p_kp1_y**2 + 4*p_k_x*p_km1_y**2*p_kp1_x - 4*p_k_x*p_kp1_x**3 - 4*p_k_x*p_kp1_x*p_kp1_y**2 + 4*p_k_y**2*p_km1_y**2 - 8*p_k_y**2*p_km1_y*p_kp1_y + 4*p_k_y**2*p_kp1_y**2 - 4*p_k_y*p_km1_x**2*p_km1_y + 4*p_k_y*p_km1_x**2*p_kp1_y - 4*p_k_y*p_km1_y**3 + 4*p_k_y*p_km1_y**2*p_kp1_y + 4*p_k_y*p_km1_y*p_kp1_x**2 + 4*p_k_y*p_km1_y*p_kp1_y**2 - 4*p_k_y*p_kp1_x**2*p_kp1_y - 4*p_k_y*p_kp1_y**3 + p_km1_x**4 + 2*p_km1_x**2*p_km1_y**2 - 2*p_km1_x**2*p_kp1_x**2 - 2*p_km1_x**2*p_kp1_y**2 + p_km1_y**4 - 2*p_km1_y**2*p_kp1_x**2 - 2*p_km1_y**2*p_kp1_y**2 + p_kp1_x**4 + 2*p_kp1_x**2*p_kp1_y**2 + p_kp1_y**4
