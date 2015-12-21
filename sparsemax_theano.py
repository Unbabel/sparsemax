import numpy as np
import theano
from theano import gof
from theano.tensor import basic as tensor
from theano.gof import Apply

def project_onto_simplex(a, radius=1.0):
    '''Project point a to the probability simplex.
    Returns the projected point x and the residual value.'''
    x0 = a.copy()
    d = len(x0);
    ind_sort = np.argsort(-x0)
    y0 = x0[ind_sort]
    ycum = np.cumsum(y0)
    val = 1.0/np.arange(1,d+1) * (ycum - radius)
    ind = np.nonzero(y0 > val)[0]
    rho = ind[-1]
    tau = val[rho]
    y = y0 - tau
    ind = np.nonzero(y < 0)
    y[ind] = 0
    x = x0.copy()
    x[ind_sort] = y

    return x, tau, .5*np.dot(x-a, x-a)


def compute_support(probs):
    ind = probs.nonzero()[0]
    supp =  np.zeros_like(probs)
    supp[ind] = 1.
    return supp


class SoftmaxGrad(gof.Op):
    """Gradient wrt x of the Softmax Op"""
    nin = 2
    nout = 1

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    #def __hash__(self):
    #    return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, dy, sm, **kwargs):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        return Apply(self, [dy, sm], [sm.type.make_variable()])

    def perform(self, node, input_storage, output_storage):
        dy, sm = input_storage
        dx = np.zeros_like(sm)
        #dx[i,j] = - (\sum_k dy[i,k] sm[i,k]) sm[i,j] + dy[i,j] sm[i,j]
        for i in xrange(sm.shape[0]):
            dy_times_sm_i = dy[i] * sm[i]
            dx[i] = dy_times_sm_i - sum(dy_times_sm_i) * sm[i]
        output_storage[0][0] = dx

    def grad(self, *args):
        raise NotImplementedError()

    def infer_shape(self, node, shape):
        return [shape[1]]

softmax_grad = SoftmaxGrad()


class Softmax(gof.Op):
    """
    WRITEME
    """

    nin = 1
    nout = 1

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got ', x.type)
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)
        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        e_x = np.exp(x - x.max(axis=1)[:, None])
        sm = e_x / e_x.sum(axis=1)[:, None]
        output_storage[0][0] = sm

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = softmax(x)
        return [softmax_grad(g_sm, sm)]

    def R_op(self, inputs, eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.grad(inputs, eval_points)

    def infer_shape(self, node, shape):
        return shape

softmax = Softmax()


class SparsemaxGrad(gof.Op):
    """Gradient wrt x of the Sparsemax Op"""
    nin = 2
    nout = 1

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    #def __hash__(self):
    #    return tensor.hashtype(self)

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, dy, sm, **kwargs):
        dy = tensor.as_tensor_variable(dy)
        sm = tensor.as_tensor_variable(sm)
        return Apply(self, [dy, sm], [sm.type.make_variable()])

    def perform(self, node, input_storage, output_storage):
        dy, sm = input_storage
        dx = np.zeros_like(sm)
        #dx[i,j] = - (\sum_{k in supp[i]} dy[i,k]) / |supp[i]| + dy[i,j] 
        for i in xrange(sm.shape[0]):
            # Compute support set supp[i].
            supp_i = compute_support(sm[i])
            dy_times_supp_i = dy[i] * supp_i
            dx[i] = dy_times_supp_i - sum(dy_times_supp_i) * supp_i / sum(supp_i)
        output_storage[0][0] = dx

    def grad(self, *args):
        raise NotImplementedError()

    def infer_shape(self, node, shape):
        return [shape[1]]

sparsemax_grad = SparsemaxGrad()


class Sparsemax(gof.Op):
    """
    WRITEME
    """

    nin = 1
    nout = 1

    def __init__(self, **kwargs):
        gof.Op.__init__(self, **kwargs)

    def __eq__(self, other):
        return type(self) == type(other)

    def __hash__(self):
        return hash(type(self))

    def __str__(self):
        return self.__class__.__name__

    def make_node(self, x):
        x = tensor.as_tensor_variable(x)
        if x.type.ndim not in (1, 2) \
                or x.type.dtype not in tensor.float_dtypes:
            raise ValueError('x must be 1-d or 2-d tensor of floats. Got ', x.type)
        if x.ndim == 1:
            x = tensor.shape_padleft(x, n_ones=1)
        return Apply(self, [x], [x.type()])

    def perform(self, node, input_storage, output_storage):
        x, = input_storage
        sm = np.zeros_like(x)
        for i in xrange(x.shape[0]):
            sm[i, :], tau, _ = project_onto_simplex(x[i])
        output_storage[0][0] = sm

    def grad(self, inp, grads):
        x, = inp
        g_sm, = grads
        sm = sparsemax(x)
        return [sparsemax_grad(g_sm, sm)]

    def R_op(self, inputs, eval_points):
        # I think the Jacobian is symmetric so the R_op
        # is the same as the grad
        if None in eval_points:
            return [None]
        return self.grad(inputs, eval_points)

    def infer_shape(self, node, shape):
        return shape

sparsemax = Sparsemax()
