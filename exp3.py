import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.stats import normaltest
data = np.loadtxt("data/data0.txt")
epsilon = 0

# use scipy.linprog to get reference result
c = [0,0,0]+[1 for _ in range(2*data.shape[0])]
A = []
b = []
for i in range(data.shape[0]):
    _A = [-2*data[i,0], -2*data[i,1], -1] + [0 for _ in range(2*data.shape[0])]
    _A[3+2*i] = -1
    A.append(_A)
    b.append(-data[i,0]**2-data[i,1]**2+epsilon)
    _A = [2*data[i,0], 2*data[i,1], 1] + [0 for _ in range(2*data.shape[0])]
    _A[3+2*i+1] = -1
    A.append(_A)
    b.append(data[i,0]**2+data[i,1]**2-epsilon)
bounds = [(None,None), (None,None), (None,None)]+ [(0,None) for _ in range(2*data.shape[0])]
res = linprog(np.array(c), A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method='interior-point')


a = res.x[0]
b = res.x[1]
r = np.sqrt(res.x[0]**2+res.x[1]**2+res.x[2])
delta_r_mean = np.mean(np.sqrt(np.sum((data-res.x[:2])**2,1))-r)
Guassian_p = normaltest(np.sqrt(np.sum((data-res.x[:2])**2,1))-r).pvalue
plt.scatter(data[:,0],data[:,1])
plt.plot(a,b,'xr')
plt.axis('equal')
plt.savefig("3.jpg")

_debug = np.array([2,3,3])