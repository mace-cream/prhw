import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import normaltest


def SVR_dual_ascent(data):
    '''
    This fuction evaluate the circle parameters of given datasets, using SVR formulation and dual ascent opimization, and further use D'Agostino and Pearson's test to verify the result.
    Input:
        data: 2-d array of shape (N,2).
    output:
        a: center coordinates a.
        b: center coordinates b.
        r: radius r.
        pvalue: test p-value.
    '''
    # Reformulate our prolem into a standard LP.
    # min c^T@x
    # st. A@x <= b
    #     x >= 0
    c = [0,0,0,0,0,0]+[1 for _ in range(2*data.shape[0])]
    A = []
    b = []
    for i in range(data.shape[0]):
        _A = [-2*data[i,0], -2*data[i,1], -1] + [2*data[i,0], 2*data[i,1], 1] + [0 for _ in range(2*data.shape[0])]
        _A[6+2*i] = -1
        A.append(_A)
        b.append(-data[i,0]**2-data[i,1]**2)
        _A = [2*data[i,0], 2*data[i,1], 1] + [-2*data[i,0], -2*data[i,1], -1] + [0 for _ in range(2*data.shape[0])]
        _A[6+2*i+1] = -1
        A.append(_A)
        b.append(data[i,0]**2+data[i,1]**2)
    c = np.array(c)
    A = np.array(A)
    b = np.array(b)

    # Use a dual ascent procedure on Augmented Lagrangians (aka. method of multipliers)
    x = np.array(c*0)+1e-4
    m = np.array(b*0)
    n = np.array(x*0)
    learning_rate = 0.1
    rou = 0.0001
    for e in range(50):
        # step 1: argmin_x L(x,m,n)
        # Here we use a gradient descent to minimize over x.
        t = np.sum(c*x)+np.sum(m*(np.matmul(A,x)-b))+rou/2*np.sum((np.matmul(A,x)-b)**2)+rou/2*np.sum(x**2)
        t0 = t
        counter = 0
        x0 = x
        while t<=t0 and counter<10000:
            t0 = t
            counter = counter+1
            g = -c-np.matmul(A.T,(m+rou*(np.matmul(A,x)-b)))+n-rou*x
            g = np.clip(g,-1,1)
            x = x + learning_rate*(g)
            t = np.sum(c*x)+np.sum(m*(np.matmul(A,x)-b))+rou/2*np.sum((np.matmul(A,x)-b)**2)+rou/2*np.sum(x**2)
        
        # step 2: (m,n) += rou*dL(x,m,n)/d(m,n) 
        m = m + rou*(np.matmul(A,x)-b)
        n = n + rou*(-x)

        # Convergence check
        if np.sum((x-x0)**2)/np.sum((x0)**2)<0.01:
            break

    # recover orginal variables from solved LP
    a = x[0]-x[3]
    b = x[1]-x[4]
    r = np.sqrt(a**2+b**2+x[2]-x[5])

    # Use a t-test to verify the result
    error = np.sqrt(np.sum((data-[a,b])**2,1))-r
    pvalue = normaltest(error).pvalue

    return a,b,r,pvalue

if __name__=="__main__":
    plt.figure(figsize=(10,10))
    for i in range(5):
        plt.subplot(3,2,i+1)
        data = np.loadtxt("data/data"+str(i)+".txt")
        a,b,r,pvalue = SVR_dual_ascent(data)

        # print and plot
        print("data/data"+str(i)+".txt:")
        print('result (a, b, r): ',(a, b, r))
        print('p-value: ',pvalue)
        plt.scatter(data[:,0],data[:,1])
        plt.plot(a,b,'xr')
        plt.axis('equal')
        plt.title("data"+str(i)+".txt")
        plt.legend(['center','data'])
    
    plt.savefig("3.jpg")
