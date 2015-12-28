import numpy as np
import theano
import theano.tensor as T

w = np.random.rand(3, 10)

th_w = theano.shared(w, borrow=True)

ind = T.ivector()

positions1 = (ind < 4).nonzero()[0]
positions2 = (ind >= 4).nonzero()[0]
ind1 = ind[positions1]
ind2 = ind[positions2]

th_w_ind1 = th_w[:, ind1]
th_w_ind2 = th_w[:, ind2]
positions = T.concatenate([positions1, positions2])
x = T.concatenate([th_w_ind1, th_w_ind2], axis=1)
x = T.set_subtensor(x[:, positions], x)
y = T.sum(x**2)

gy = T.grad(y, th_w_ind1)

f = theano.function([ind], y)

print f([3,6,1,4,9])
print sum(w[:,3]**2 + w[:,1]**2 + w[:,6]**2 + w[:,4]**2 + w[:,9]**2)

g = theano.function([ind], gy)

print g([3,6,1,4,9])
print [2*w[:,3], 2*w[:,1]]






