import numpy as np
import matplotlib.pyplot as plt
from multilabel_dataset import make_multilabel_classification
from multilabel_dataset_reader import read_multilabel_dataset
#from sklearn.datasets import make_multilabel_classification
import time
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


def classify_dataset(filepath, weights, classifier_type, \
                     hyperparameter_name, \
                     hyperparameter_values):
    matched_labels = np.zeros(num_settings)
    union_labels = np.zeros(num_settings)
    num_correct = np.zeros(num_settings)
    num_total = np.zeros(num_settings)
    num_predicted_labels = np.zeros(num_settings)
    num_gold_labels = np.zeros(num_settings)
    squared_loss_dev = 0.

    num_features = weights.shape[0]
    num_labels = weights.shape[1]

    num_documents = 0

    f = open(filepath)
    for line in f:
        line = line.rstrip('\n')
        fields = line.split()
        labels = [int(l) for l in fields[0].split(',')]
        features = {}
        for field in fields[1:]:
            name_value = field.split(':')
            assert len(name_value) == 2, pdb.set_trace()
            fid = int(name_value[0])
            fval = float(name_value[1])
            assert fid > 0, pdb.set_trace() # 0 is reserved for UNK.
            assert fid not in features, pdb.set_trace()
            if num_features >= 0 and fid >= num_features:
                fid = 0 # UNK.
            features[fid] = fval

        # Now classify this instance.
        x = features
        y = np.zeros(num_labels, dtype=float)
        for label in labels:
            y[label] = 1.
        y /= sum(y)

        scores = np.zeros(num_labels)
        for fid, fval in x.iteritems():
            scores += fval * weights[fid, :]

        gold_labels = compute_support(y)
        predicted_labels_eval = []

        if classifier_type == 'sparsemax':
            probs, _, _ =  project_onto_simplex(scores)
            for sparsemax_scale in hyperparameter_values:
                scaled_probs, _, _ =  project_onto_simplex(sparsemax_scale * scores)
                predicted_labels = compute_support(scaled_probs)
                predicted_labels_eval.append(predicted_labels)
        elif classifier_type == 'softmax':
            probs = np.exp(scores) / np.sum(np.exp(scores))
            for probability_threshold in hyperparameter_values:
                predicted_labels = (probs > probability_threshold).astype(float)
                predicted_labels_eval.append(predicted_labels)
        elif classifier_type == 'logistic':
            probs = 1. / (1. + np.exp(-scores))
            predicted_labels = (probs > .5).astype(float)
            predicted_labels_eval.append(predicted_labels)
        else:
            raise NotImplementedError

        squared_loss_dev += sum((probs - y)**2)
        for k in xrange(len(hyperparameter_values)):
            predicted_labels = predicted_labels_eval[k]
            matched_labels[k] += gold_labels.dot(predicted_labels)
            union_labels[k] += sum(compute_support(gold_labels + predicted_labels))
            num_gold_labels[k] += sum(gold_labels)
            num_predicted_labels[k] += sum(predicted_labels)

            num_correct[k] += sum((gold_labels == predicted_labels).astype(float))
            num_total[k] += len(gold_labels)

        num_documents += 1

    f.close()

    squared_loss_dev /= (num_documents*num_labels)
    print 'Number of documents in %s: %d, sq loss: %f' % \
        (filepath, num_documents, squared_loss_dev)

    for k in xrange(len(hyperparameter_values)):
        acc_dev = matched_labels / union_labels
        hamming_dev = num_correct / num_total
        P_dev = matched_labels / num_predicted_labels
        R_dev = matched_labels / num_gold_labels
        F1_dev = 2*P_dev*R_dev / (P_dev + R_dev)

        print '%s: %f, acc_dev: %f, hamming_dev: %f, P_dev: %f, R_dev: %f, F1_dev: %f' % \
            (hyperparameter_name, hyperparameter_values[k], \
             acc_dev[k], hamming_dev[k], P_dev[k], R_dev[k], F1_dev[k])


###########################

loss_function = sys.argv[1] #'softmax' #'logistic' # 'sparsemax'
num_epochs = int(sys.argv[2]) #20
learning_rate = float(sys.argv[3]) #0.001
regularization_constant = float(sys.argv[4])

sparsemax_scales = [1., 1.5,  2., 2.5, 3., 3.5, 4.]
softmax_thresholds = [.05, .06, .07, .08, .09, .1]

filepath_train = sys.argv[5]
X_train, Y_train, num_features = read_multilabel_dataset(filepath_train, \
                                                         sparse=True)
num_labels = Y_train.shape[1]
filepath_dev = sys.argv[6]
filepath_test = sys.argv[7]

num_words = num_features
num_classes = num_labels
num_documents_train = len(X_train)

weights = np.zeros((num_words, num_classes))
t = 0
for epoch in xrange(num_epochs):

    tic = time.time()
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
        x = X_train[i]
        scores = np.zeros(num_classes)
        for fid, fval in x.iteritems():
            scores += fval * weights[fid, :]

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
            grad = {}
            for fid, fval in x.iteritems():
                grad[fid] = fval * delta
        elif loss_function == 'softmax':
            probs = np.exp(scores) / np.sum(np.exp(scores))

            for probability_threshold in softmax_thresholds:
                predicted_labels = (probs > probability_threshold).astype(float)
                predicted_labels_eval.append(predicted_labels)

            loss_t = -scores.dot(y) + np.log(np.sum(np.exp(scores)))
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -y + probs
            grad = {}
            for fid, fval in x.iteritems():
                grad[fid] = fval * delta
        elif loss_function == 'logistic':
            probs = 1. / (1. + np.exp(-scores))
            predicted_labels = (probs > .5).astype(float)
            #pdb.set_trace()
            loss_t = \
                -np.log(probs).dot(gold_labels) - np.log(1. - probs).dot(1. - gold_labels)
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -gold_labels + probs
            grad = {}
            for fid, fval in x.iteritems():
                grad[fid] = fval * delta

        #if epoch > 10 and sum(gold_labels) >= 2.:
        #    print y, probs, loss_t

        for k, predicted_labels in enumerate(predicted_labels_eval):
            matched_labels[k] += gold_labels.dot(predicted_labels)
            union_labels[k] += sum(compute_support(gold_labels + predicted_labels))

        assert eta * regularization_constant < 1. #, pdb.set_trace()
        weights *= (1. - eta*regularization_constant)

        for fid, fval in grad.iteritems():
            weights[fid] -= eta * fval

        t += 1

        #print y, probs

    acc_train = np.zeros(num_settings)
    for k in xrange(len(acc_train)):
        acc_train[k] = matched_labels[k] / union_labels[k]

    loss /= num_documents_train
    loss += 0.5 * regularization_constant * np.linalg.norm(weights.flatten())**2

    elapsed_time = time.time() - tic

    print 'Epoch %d, loss: %f, time: %f' % (epoch+1, loss, elapsed_time)


    # Test the classifier on dev/test data.
    if loss_function == 'softmax':
        hyperparameter_name = 'softmax_thres'
        hyperparameter_values = softmax_thresholds
    else:
        hyperparameter_name = 'sparsemax_scale'
        hyperparameter_values = sparsemax_scales

    for k in xrange(len(hyperparameter_values)):
        print '%s: %f, acc train: %f' % \
            (hyperparameter_name, hyperparameter_values[k], acc_train[k])

    tic = time.time()
    classify_dataset(filepath_dev, weights, loss_function, \
                     hyperparameter_name, \
                     hyperparameter_values)
    elapsed_time = time.time() - tic
    print 'Time to test: %f' % elapsed_time


