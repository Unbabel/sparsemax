import numpy as np
import matplotlib.pyplot as plt
from multilabel_dataset import make_multilabel_classification
#from sklearn.datasets import make_multilabel_classification
import sys
import pdb

def write_dataset(filepath, X, Y, use_class_proportions):
    f = open(filepath, 'w')
    num_examples = Y.shape[0]
    num_labels = Y.shape[1]
    num_features = X.shape[1]
    for i in xrange(num_examples):
        labels = Y[i,:].nonzero()[0]
        fids = X[i,:].nonzero()[0]
        fvals = X[i,fids]
        if use_class_proportions:
            label_probabilities = Y[i, labels]
            f.write(','.join([str(label) + ':' + str(prob) for label, prob in zip(labels, label_probabilities)]))
        else:
            f.write(','.join([str(label) for label in labels]))
        f.write('\t')
        f.write(' '.join([str(fid+1) + ':' + str(fval) for fid, fval in zip(fids, fvals)]))
        f.write('\n')
    f.close()


if True:
    dataset_name = sys.argv[1]
    num_classes = int(sys.argv[2])
    num_documents_train = 1200
    num_documents_dev = 200
    num_documents_test = 1000
    num_words = num_classes
    num_average_labels = 1
    document_length = sys.argv[3] #100*num_classes # 5000 #1000 #500
    use_class_proportions = True # False

    num_documents_total = \
                          num_documents_train + num_documents_dev + num_documents_test
    X, Y = make_multilabel_classification(n_samples=num_documents_total,
                                          n_features=num_words,
                                          n_classes=num_classes,
                                          n_labels=num_average_labels,
                                          length=document_length,
                                          allow_unlabeled=False,
                                          use_class_proportions=use_class_proportions,
                                          random_state=1)

    #pdb.set_trace()

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

    write_dataset('%s-train.txt' % dataset_name, X_train, Y_train, use_class_proportions)
    write_dataset('%s-dev.txt' % dataset_name, X_dev, Y_dev, use_class_proportions)
    write_dataset('%s-test.txt' % dataset_name, X_test, Y_test, use_class_proportions)













