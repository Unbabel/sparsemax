import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from multilabel_dataset_reader import read_multilabel_dataset
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

def evaluate_and_compute_gradient(weights_flattened, *args):
    X_train = args[0]
    Y_train = args[1]
    loss_function = args[2]
    regularization_constant = args[3]
    add_bias = args[4]

    num_documents_train = len(X_train)

    weights = weights_flattened.reshape(num_words, num_classes)
    gradient = np.zeros_like(weights)

    loss = 0.
    for i in xrange(num_documents_train):
        y = Y_train[i,:].copy() / sum(Y_train[i,:])
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
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -y + probs
            for fid, fval in x.iteritems():
                gradient[fid] += fval * delta
        elif loss_function == 'softmax':
            probs = np.exp(scores) / np.sum(np.exp(scores))
            loss_t = -scores.dot(y) + np.log(np.sum(np.exp(scores)))
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = -y + probs
            for fid, fval in x.iteritems():
                gradient[fid] += fval * delta
        elif loss_function == 'logistic':
            probs = 1. / (1. + np.exp(-scores))
            log_probs = -np.log(1. + np.exp(-scores))
            log_neg_probs = -np.log(1. + np.exp(scores))
            loss_t = \
                -log_probs.dot(gold_labels) - log_neg_probs.dot(1. - gold_labels)
            #loss_t = \
            #    -np.log(probs).dot(gold_labels) - np.log(1. - probs).dot(1. - gold_labels)
            loss += loss_t
            if not loss_t > -1e-6:
                print loss_t
            assert loss_t > -1e-6 #, pdb.set_trace()
            delta = -gold_labels + probs
            for fid, fval in x.iteritems():
                gradient[fid] += fval * delta

    gradient /= num_documents_train
    loss /= num_documents_train
    if add_bias:
        gradient[2:,:] += regularization_constant * weights[2:,:]
        reg = 0.5 * regularization_constant * np.linalg.norm(weights[2:,:].flatten())**2
    else:
        gradient += regularization_constant * weights
        reg = 0.5 * regularization_constant * np.linalg.norm(weights.flatten())**2

    objective_value = loss + reg

    print 'Loss: %f, Reg: %f, Loss+Reg: %f' % (loss, reg, objective_value)
    return objective_value, gradient.flatten()



def classify_dataset(filepath, weights, classifier_type, \
                     hyperparameter_name, \
                     hyperparameter_values, add_bias=False):
    num_settings = len(hyperparameter_values)

    rank_acc = np.zeros(num_settings)
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

    num_matched_by_label = np.zeros((num_settings, num_labels))
    num_predicted_by_label = np.zeros((num_settings, num_labels))
    num_gold_by_label = np.zeros((num_settings, num_labels))

    f = open(filepath)
    for line in f:
        line = line.rstrip('\n')
        fields = line.split()
        labels = [int(l) for l in fields[0].split(',')]
        features = {}
        if add_bias:
            features[1] = 1.
        for field in fields[1:]:
            name_value = field.split(':')
            assert len(name_value) == 2, pdb.set_trace()
            fid = int(name_value[0])
            fval = float(name_value[1])
            assert fid > 0, pdb.set_trace() # 0 is reserved for UNK.
            if add_bias:
                fid += 1 # 1 is reserved for bias feature.
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
            for probability_threshold in hyperparameter_values:
                predicted_labels = (probs > probability_threshold).astype(float)
                predicted_labels_eval.append(predicted_labels)
        else:
            raise NotImplementedError

        squared_loss_dev += sum((probs - y)**2)
        num_gold = sum(gold_labels)
        for k in xrange(len(hyperparameter_values)):
            predicted_labels = predicted_labels_eval[k]
            num_predicted = sum(predicted_labels)
            num_matched = gold_labels.dot(predicted_labels)
            num_union = sum(compute_support(gold_labels + predicted_labels))
            assert num_union == num_predicted + num_gold - num_matched

            for l in xrange(num_labels):
                if predicted_labels[l] == 1:
                    num_predicted_by_label[k, l] += 1.
                    if gold_labels[l] == 1:
                        num_matched_by_label[k, l] += 1.
                if gold_labels[l] == 1:
                    num_gold_by_label[k, l] += 1.

            rank_acc[k] += float(num_matched * (num_labels - num_union)) / float(num_gold * (num_labels - num_gold))
            matched_labels[k] += num_matched
            union_labels[k] += num_union
            num_gold_labels[k] += num_gold
            num_predicted_labels[k] += num_predicted

            num_correct[k] += sum((gold_labels == predicted_labels).astype(float))
            num_total[k] += len(gold_labels)

        num_documents += 1

    f.close()

    squared_loss_dev /= (num_documents*num_labels)
    print 'Number of documents in %s: %d, sq loss: %f' % \
        (filepath, num_documents, squared_loss_dev)

    acc_dev = matched_labels / union_labels
    hamming_dev = num_correct / num_total
    P_dev = matched_labels / num_predicted_labels
    R_dev = matched_labels / num_gold_labels
    F1_dev = 2*P_dev*R_dev / (P_dev + R_dev)

    Pl_dev = num_matched_by_label / num_predicted_by_label
    Rl_dev = num_matched_by_label / num_gold_by_label
    F1l_dev = 2*Pl_dev*Rl_dev / (Pl_dev + Rl_dev)

    Pl_dev = np.nan_to_num(Pl_dev) # Replace nans with zeros.
    Rl_dev = np.nan_to_num(Rl_dev) # Replace nans with zeros.
    F1l_dev = np.nan_to_num(F1l_dev) # Replace nans with zeros.

    rank_acc /= float(num_documents)

    print_all_labels = False

    for k in xrange(len(hyperparameter_values)):
        
        macro_P_dev = np.mean(Pl_dev[k, :])
        macro_R_dev = np.mean(Rl_dev[k, :])
        macro_F1_dev = 2*macro_P_dev*macro_R_dev / (macro_P_dev + macro_R_dev)
        #macro_F1_dev_wrong = np.mean(F1l_dev[k, :])
        print '%s: %f, acc_dev: %f, hamming_dev: %f, P_dev: %f, R_dev: %f, F1_dev: %f, macro_P_dev: %f, macro_R_dev: %f, macro_F1_dev: %f, rank_acc: %f' % \
            (hyperparameter_name, hyperparameter_values[k], \
             acc_dev[k], hamming_dev[k], P_dev[k], R_dev[k], F1_dev[k], macro_P_dev, macro_R_dev, macro_F1_dev, rank_acc[k])

        if print_all_labels:
            for l in xrange(num_labels): 
                print '  LABEL %d, %s: %f,  P_dev: %f, R_dev: %f, F1_dev: %f' % \
                    (l, hyperparameter_name, hyperparameter_values[k], \
                     Pl_dev[k, l], Rl_dev[k, l], F1l_dev[k, l])



###########################

loss_function = sys.argv[1] #'softmax' #'logistic' # 'sparsemax'
num_epochs = int(sys.argv[2]) #20
#learning_rate = float(sys.argv[3]) #0.001
regularization_constant = float(sys.argv[3])

sparsemax_scales = [1., 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.5, 3., 3.5, 4., 4.5, 5., 5.5, 6., 6.5, 7., 7.5, 8., 8.5, 9., 9.5, 10., 10.5, 11.0, 11.5, 12., 12.5, 13., 13.5, 14., 14.5, 15., 15.5, 16, 16.5, 17., 17.5, 18, 18.5, 19, 19.5, 20.]
softmax_thresholds = [.005, .006, .007, .008, .009, .01, .02, .03, .04, .05, .06, .07, .08, .09, .1, .15, .20, .25, .30, .35, .40, .45, .50]
logistic_thresholds = [.05, .06, .07, .08, .09, .1, .2, .3, .4, .5, .6, .7]

add_bias = True # False

filepath_train = sys.argv[4]
X_train, Y_train, num_features = read_multilabel_dataset(filepath_train, \
                                                         add_bias=add_bias, \
                                                         sparse=True)
num_labels = Y_train.shape[1]
filepath_dev = sys.argv[5]
filepath_test = sys.argv[6]

num_words = num_features
num_classes = num_labels
num_documents_train = len(X_train)

if loss_function == 'softmax':
    hyperparameter_name = 'softmax_thres'
    hyperparameter_values = softmax_thresholds
elif loss_function == 'sparsemax':
    hyperparameter_name = 'sparsemax_scale'
    hyperparameter_values = sparsemax_scales
elif loss_function == 'logistic':
    hyperparameter_name = 'logistic_thres'
    hyperparameter_values = logistic_thresholds
else:
    raise NotImplementedError

weights = np.zeros((num_words, num_classes))
weights_flattened = weights.flatten()

# Optimize with L-BFGS.
weights_flattened, value, d = \
    opt.fmin_l_bfgs_b(evaluate_and_compute_gradient,
                      x0=weights_flattened,
                      args=(X_train, Y_train, loss_function, regularization_constant, add_bias),
                      m=10,
                      factr=100,
                      pgtol=1e-08,
                      epsilon=1e-12,
                      approx_grad=False,
                      disp=True,
                      maxfun=1000,
                      maxiter=num_epochs) #1000)

weights = weights_flattened.reshape(num_words, num_classes)

if d['warnflag'] != 0:
    print 'Not converged', d

#pdb.set_trace()


#print 'Epoch %d, reg: %f, loss: %f, reg+loss: %f, time: %f' % (epoch+1, reg, loss, reg+loss, elapsed_time)

print 'Running on the dev set...'
tic = time.time()
classify_dataset(filepath_dev, weights, loss_function, \
                 hyperparameter_name, \
                 hyperparameter_values,
                 add_bias=add_bias)
elapsed_time = time.time() - tic
print 'Time to test: %f' % elapsed_time

#print 'Running on the test set...'
#tic = time.time()
#classify_dataset(filepath_test, weights, loss_function, \
#                 hyperparameter_name, \
#                 hyperparameter_values)
#elapsed_time = time.time() - tic
#print 'Time to test: %f' % elapsed_time

