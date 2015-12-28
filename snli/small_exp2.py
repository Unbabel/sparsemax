import numpy as np
import theano
import theano.tensor as T

w = np.random.rand(10)

th_w = theano.shared(w, borrow=True)

positions1 = T.ivector()
positions2 = T.ivector()
ind1 = T.ivector()
ind2 = T.ivector()

th_w_ind1 = th_w[ind1]
th_w_ind2 = th_w[ind2]
positions = T.concatenate([positions1, positions2])
x = T.concatenate([th_w_ind1, th_w_ind2])
x = T.set_subtensor(x[positions], x)
y = T.sum(x**2)

gy = T.grad(y, th_w_ind1)

f = theano.function([positions1, positions2, ind1, ind2], y)

print f([0,2], [1,3,4], [3,1], [6,4,9])
print w[3]**2 + w[1]**2 + w[6]**2 + w[4]**2 + w[9]**2

g = theano.function([positions1, positions2, ind1, ind2], gy)

print g([0,2], [1,3,4], [3,1], [6,4,9])
print [2*w[3], 2*w[1]]






