import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.stats import normaltest
data = np.loadtxt("data/data0.txt")
epsilon = 0

# use scipy.linprog to get reference result
c = [0,0,0,0,0,0]+[1 for _ in range(2*data.shape[0])]
A = []
b = []
for i in range(data.shape[0]):
    _A = [-2*data[i,0], -2*data[i,1], -1] + [2*data[i,0], 2*data[i,1], 1] + [0 for _ in range(2*data.shape[0])]
    _A[6+2*i] = -1
    A.append(_A)
    b.append(-data[i,0]**2-data[i,1]**2+epsilon)
    _A = [2*data[i,0], 2*data[i,1], 1] + [-2*data[i,0], -2*data[i,1], -1] + [0 for _ in range(2*data.shape[0])]
    _A[6+2*i+1] = -1
    A.append(_A)
    b.append(data[i,0]**2+data[i,1]**2-epsilon)
c = np.array(c)
A = np.array(A)
b = np.array(b)
#res = linprog(np.array(c), A_ub=np.array(A), b_ub=np.array(b), method='interior-point')

x = np.array(c*0)
m = np.array(b*0)
n = np.array(x*0)
learning_rate = 0.1
rou = 0.001
for e in range(50):
    t = np.sum(c*x)+np.sum(m*(np.matmul(A,x)-b))+rou/2*np.sum((np.matmul(A,x)-b)**2)+rou/2*np.sum(x**2)
    t0 = t
    counter = 0
    while t<=t0 and counter<10000:
        t0 = t
        counter = counter+1
        g = -c-np.matmul(A.T,(m+rou*(np.matmul(A,x)-b)))+n-rou*x
        g = np.clip(g,-1,1)
        x = x + learning_rate*(g)
        t = np.sum(c*x)+np.sum(m*(np.matmul(A,x)-b))+rou/2*np.sum((np.matmul(A,x)-b)**2)+rou/2*np.sum(x**2)
    m = m + rou*(np.matmul(A,x)-b)
    n = n + rou*(-x)

a = x[0]-x[3]
b = x[1]-x[4]
r = np.sqrt(a**2+b**2+x[2]-x[5])
# a = res.x[0]-res.x[3]
# b = res.x[1]-res.x[4]
# r = np.sqrt(a**2+b**2+res.x[2]-res.x[5])
delta_r_mean = np.mean(np.sqrt(np.sum((data-[a,b])**2,1))-r)
Guassian_p = normaltest(np.sqrt(np.sum((data-[a,b])**2,1))-r).pvalue
plt.cla()
plt.scatter(data[:,0],data[:,1])
plt.plot(a,b,'xr')
plt.axis('equal')
plt.savefig("3.jpg")

_debug = np.array([2,3,3])