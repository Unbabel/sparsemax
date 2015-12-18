import sparsemax_theano

u = T.matrix()
g = sparsemax_theano.sparsemax
cost = g(u)

X = np.random.randn(3,5)
print cost.eval({u: X})

# Not sure the gradient is correct if the index corresponds to something which is not in the support set!!!
# I think the above is fixed now.
h = T.grad(cost=cost[0,2], wrt=u)

print h.eval({u: X})

