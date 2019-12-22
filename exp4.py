import numpy as np
import mytf

a = mytf.tensor([2,2],'a')
b = mytf.tensor([2,1],'b')
c = mytf.matmul(a,b)
c = mytf.sigmoid(c)
d = mytf.tensor([1,2],'d')
c = mytf.matmul(d,c)
c = mytf.sigmoid(c)

#feed = {'a':np.ones((1,2)),'b':np.ones((2,1))}
feed = {'a':np.array([[1,2],[3,4]]),'b':np.ones((2,1)),'d':np.array([[1,0]])}
feed2 = {'a':np.array([[1+1e-5,2],[3,4]]),'b':np.ones((2,1)),'d':np.array([[1,0]])}

print((c.eval(feed2)-c.eval(feed))/1e-5)
print(a.back(c,feed))

_debug = np.array([2,3,3])