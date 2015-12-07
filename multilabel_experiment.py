import numpy as np
import matplotlib.pyplot as plt
from multilabel_dataset import make_multilabel_classification
from multilabel_dataset_reader import read_multilabel_dataset
#from sklearn.datasets import make_multilabel_classification
import sys
import pdb

def compute_support(probs):
    ind = probs.nonzero()[0]
    supp =  np.zeros_like(probs)
    supp[ind] = 1.
    return supp

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


loss_function = sys.argv[1] #'softmax' #'logistic' # 'sparsemax'
num_epochs = int(sys.argv[2]) #20
learning_rate = float(sys.argv[3]) #0.001
regularization_constant = float(sys.argv[4])

#sparsemax_scales = [1., 1.5,  2., 2.5]
sparsemax_scales = [1., 1.5,  2., 2.5, 3., 3.5, 4.]
softmax_thresholds = [.05, .06, .07, .08, .09, .1]

#probability_threshold = 0.05
#sparsemax_scale = 1.5

toy_data = False #True
sparse = True
if toy_data:
    num_documents_train = 1000
    num_documents_dev = 200
    num_documents_test = 200
    num_words = 20
    num_classes = 5
    num_average_labels = 1
    document_length = 500 #100 #500

    num_documents_total = \
                          num_documents_train + num_documents_dev + num_documents_test
    X, Y = make_multilabel_classification(n_samples=num_documents_total,
                                          n_features=num_words,
                                          n_classes=num_classes,
                                          n_labels=num_average_labels,
                                          length=document_length,
                                          allow_unlabeled=False,
                                          use_class_proportions=True,
                                          random_state=1)
    X = X.astype(float)
    Y = Y.astype(float)

    offset = 0
    X_train = X[offset:(offset+num_documents_train), :]
    Y_train = Y[offset:(offset+num_documents_train), :]

    offset += num_documents_train
    X_dev = X[offset:(offset+num_documents_dev), :]
    Y_dev = Y[offset:(offset+num_documents_dev), :]

    offset += num_documents_dev
    X_test = X[offset:(offset+num_documents_test), :]
    Y_test = Y[offset:(offset+num_documents_test), :]

else:
    filepath_train = sys.argv[5]
    X_train, Y_train, num_features = read_multilabel_dataset(filepath_train, sparse=sparse)
    num_labels = Y_train.shape[1]
    #num_features = X_train.shape[1]
    filepath_dev = sys.argv[6]
    X_dev, Y_dev, _ = read_multilabel_dataset(filepath_dev, \
                                              num_labels=num_labels, \
                                              num_features=num_features, \
                                              sparse=sparse)
    filepath_test = sys.argv[7]
    X_test, Y_test, _ = read_multilabel_dataset(filepath_test, \
                                                num_labels=num_labels, \
                                                num_features=num_features, \
                                                sparse=sparse)
    num_words = num_features
    num_classes = num_labels
    if sparse:
        num_documents_train = len(X_train)
        num_documents_dev = len(X_dev)
        num_documents_test = len(X_test)
    else:
        num_documents_train = X_train.shape[0]
        num_documents_dev = X_dev.shape[0]
        num_documents_test = X_test.shape[0]

#pdb.set_trace()

#scale_weights = 1.
weights = np.zeros((num_words, num_classes))
t = 0
for epoch in xrange(num_epochs):

    if loss_function == 'sparsemax':
        num_settings = len(sparsemax_scales)
    elif loss_function == 'softmax':
        num_settings = len(softmax_thresholds)

    matched_labels = np.zeros(num_settings)
    union_labels = np.zeros(num_settings)
    loss = 0.
    for i in xrange(num_documents_train):
        y = Y_train[i,:].copy() / sum(Y_train[i,:])
        eta = learning_rate / np.sqrt(float(t+1))
        if sparse:
            x = X_train[i]
            scores = np.zeros(num_classes)
            for fid, fval in x.iteritems():
                scores += fval * weights[fid, :]
        else:
            x = X_train[i,:]
            scores = x.dot(weights)

        gold_labels = compute_support(y)

        predicted_labels_eval = []
        if loss_function == 'sparsemax':
            probs, tau, _ =  project_onto_simplex(scores)
            predicted_labels = compute_support(probs)
            loss_t = \
                -scores.dot(y) + .5*(scores**2 - tau**2).dot(predicted_labels) + .5/sum(gold_labels)
            for sparsemax_scale in sparsemax_scales:
                scaled_probs, _, _ =  project_onto_simplex(sparsemax_scale * scores)
                predicted_labels_eval.append(compute_support(scaled_probs))

            #print loss_t, -scores.dot(y - .5*probs) + .5*tau + .5./sum(gold_labels)
            #predicted_labels = (probs > probability_threshold).astype(float)
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -y + probs
            if sparse:
                grad = {}
                for fid, fval in x.iteritems():
                    grad[fid] = fval * delta
            else:
                grad = x[:,None].dot(delta[None,:])
        elif loss_function == 'softmax':
            probs = np.exp(scores) / np.sum(np.exp(scores))

            for probability_threshold in softmax_thresholds:
                predicted_labels = (probs > probability_threshold).astype(float)
                predicted_labels_eval.append(predicted_labels)

            loss_t = -scores.dot(y) + np.log(np.sum(np.exp(scores)))
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -y + probs
            if sparse:
                grad = {}
                for fid, fval in x.iteritems():
                    grad[fid] = fval * delta
            else:
                grad = x[:,None].dot(delta[None,:])
        elif loss_function == 'logistic':
            probs = 1. / (1. + np.exp(-scores))
            predicted_labels = (probs > .5).astype(float)
            #pdb.set_trace()
            loss_t = \
                -np.log(probs).dot(gold_labels) - np.log(1. - probs).dot(1. - gold_labels)
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -gold_labels + probs
            if sparse:
                grad = {}
                for fid, fval in x.iteritems():
                    grad[fid] = fval * delta
            else:
                grad = x[:,None].dot(delta[None,:])

        #if epoch > 10 and sum(gold_labels) >= 2.:
        #    print y, probs, loss_t

        for k, predicted_labels in enumerate(predicted_labels_eval):
            matched_labels[k] += gold_labels.dot(predicted_labels)
            union_labels[k] += sum(compute_support(gold_labels + predicted_labels))

        assert eta * regularization_constant < 1. #, pdb.set_trace()
        weights *= (1. - eta*regularization_constant)
        if sparse:
            # TODO.
            for fid, fval in grad.iteritems():
                weights[fid] -= eta * fval
        else:
            weights -= eta * grad

        t += 1

        #print y, probs

    acc_train = np.zeros(num_settings)
    for k in xrange(len(acc_train)):
        acc_train[k] = matched_labels[k] / union_labels[k]

    matched_labels = np.zeros(num_settings)
    union_labels = np.zeros(num_settings)
    num_correct = np.zeros(num_settings)
    num_total = np.zeros(num_settings)
    num_predicted_labels = np.zeros(num_settings)
    num_gold_labels = np.zeros(num_settings)
    squared_loss_dev = 0.
    for i in xrange(num_documents_dev):
        y = Y_dev[i,:].copy() / sum(Y_dev[i,:])
        if sparse:
            x = X_dev[i]
            scores = np.zeros(num_classes)
            for fid, fval in x.iteritems():
                scores += fval * weights[fid, :]
        else:
            x = X_dev[i,:]
            scores = x.dot(weights)

        gold_labels = compute_support(y)
        predicted_labels_eval = []

        if loss_function == 'sparsemax':
            probs, _, _ =  project_onto_simplex(scores)
            for sparsemax_scale in sparsemax_scales:
                scaled_probs, _, _ =  project_onto_simplex(sparsemax_scale * scores)
                predicted_labels = compute_support(scaled_probs)
                predicted_labels_eval.append(predicted_labels)
        elif loss_function == 'softmax':
            probs = np.exp(scores) / np.sum(np.exp(scores))
            for probability_threshold in softmax_thresholds:
                predicted_labels = (probs > probability_threshold).astype(float)
                predicted_labels_eval.append(predicted_labels)
        elif loss_function == 'logistic':
            probs = 1. / (1. + np.exp(-scores))
            predicted_labels = (probs > .5).astype(float)

        squared_loss_dev += sum((probs - y)**2)
        for k in xrange(num_settings):
            predicted_labels = predicted_labels_eval[k]
            matched_labels[k] += gold_labels.dot(predicted_labels)
            union_labels[k] += sum(compute_support(gold_labels + predicted_labels))
            num_gold_labels[k] += sum(gold_labels)
            num_predicted_labels[k] += sum(predicted_labels)

            num_correct[k] += sum((gold_labels == predicted_labels).astype(float))
            num_total[k] += len(gold_labels)

    # HERE.
    loss /= num_documents_train
    loss += 0.5 * regularization_constant * np.linalg.norm(weights.flatten())**2
    squared_loss_dev /= (num_documents_dev*num_classes)
    print 'Epoch %d, loss: %f, squared_loss_dev: %f' % (epoch+1, loss, squared_loss_dev)

    for k in xrange(num_settings):
        acc_dev = matched_labels / union_labels
        hamming_dev = num_correct / num_total
        P_dev = matched_labels / num_predicted_labels
        R_dev = matched_labels / num_gold_labels
        F1_dev = 2*P_dev*R_dev / (P_dev + R_dev)

        print 'setting: %d, acc train: %f, acc_dev: %f, hamming_dev: %f, P_dev: %f, R_dev: %f, F1_dev: %f' % \
            (k, acc_train[k], acc_dev[k], hamming_dev[k], P_dev[k], R_dev[k], F1_dev[k])

    #pdb.set_trace()













