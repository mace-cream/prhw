import numpy as np
import mytf

a = mytf.tensor([2,2],'a')
b = mytf.tensor([2,1],'b')
c = mytf.tensor([1,2],'c')
a2 = mytf.sigmoid(mytf.matmul(a,a))
d = mytf.sigmoid(mytf.matmul(a2,b))
e = mytf.sigmoid(mytf.matmul(c,a2))
c = mytf.sigmoid(mytf.matmul(e,d))

feed = {'a':np.array([[1.,2],[3,4]]),'b':np.array([[1.],[2]]),'c':np.array([[1.,2]])}
print(a.back(c,feed))
import copy
for i in range(2):
    for j in range(2):
        feed2 = copy.deepcopy(feed)
        feed2['a'][i][j] = feed2['a'][i][j]+1e-5
        print((c.eval(feed2)-c.eval(feed))/1e-5)

_debug = np.array([2,3,3])