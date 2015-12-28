import numpy as np
import theano
import theano.tensor as T

w1 = np.random.rand(4)
w2 = np.random.rand(6)

th_w1 = theano.shared(w1, borrow=True)
th_w2 = theano.shared(w2, borrow=True)

positions1 = T.ivector()
positions2 = T.ivector()
ind1 = T.ivector()
ind2 = T.ivector()

th_w1_ind = th_w1[ind1]
th_w2_ind = th_w2[ind2]
positions = T.concatenate([positions1, positions2])
x = T.concatenate([th_w1_ind, th_w2_ind])
x = T.set_subtensor(x[positions], x)
y = T.sum(x**2)

gy = T.grad(y, th_w1_ind)

f = theano.function([positions1, positions2, ind1, ind2], y)

print f([0,2], [1,3,4], [3,1], [2,0,5])
print w1[3]**2 + w1[1]**2 + w2[2]**2 + w2[0]**2 + w2[5]**2

g = theano.function([positions1, positions2, ind1, ind2], gy)

print g([0,2], [1,3,4], [3,1], [2,0,5])
print [2*w1[3], 2*w1[1]]






