import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import pdb

def classify_dataset(X, y, weights):
    num_features = weights.shape[0]
    num_labels = weights.shape[1]
    num_examples = X.shape[0]
    num_correct = 0.
    for i in xrange(num_examples):
        x = X[i,:]
        scores = x.dot(weights)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        y_pred = np.argmax(probs)
        if y_pred == y[i]:
            num_correct += 1.

    acc = num_correct / num_examples

    print 'Dev acc: %f' % (acc)

def train(X, y, regularization_constant, num_epochs, learning_rate, X_dev, y_dev):
    num_labels = 1+max(y)
    num_features = X.shape[1]
    num_examples = X.shape[0]
    weights = np.zeros((num_features, num_labels))
    t = 0
    for epoch in xrange(num_epochs):
        tic = time.time()
        loss = 0.
        for i in xrange(num_examples):
            eta = learning_rate / np.sqrt(float(t+1))
            x = X[i,:]
            scores = x.dot(weights)
            probs = np.exp(scores) / np.sum(np.exp(scores))
            loss_t = -scores[y[i]] + np.log(np.sum(np.exp(scores)))
            loss += loss_t
            assert loss_t > -1e-9 #, pdb.set_trace()
            delta = probs
            delta[y[i]] -= 1.
            grad = x[:,None].dot(delta[None,:])
            #pdb.set_trace()
            assert eta * regularization_constant < 1. #, pdb.set_trace()
            weights *= (1. - eta*regularization_constant)
            weights -= eta * grad
            t += 1

        loss /= num_examples
        loss += 0.5 * regularization_constant * np.linalg.norm(weights.flatten())**2

        elapsed_time = time.time() - tic

        print 'Epoch %d, loss: %f, time: %f' % (epoch+1, loss, elapsed_time)

        # Test the classifier on dev/test data.
        tic = time.time()
        classify_dataset(X_dev, y_dev, weights)
        elapsed_time = time.time() - tic
        print 'Time to test: %f' % elapsed_time


num_epochs = int(sys.argv[1]) #20
learning_rate = float(sys.argv[2]) #0.001
regularization_constant = float(sys.argv[3])

input_size = 2
num_classes = 2
num_training_examples = 1000
num_dev_examples = 500
class_deviation = 0.5

np.random.seed(42)

num_examples = num_training_examples + num_dev_examples

# Generate class means.
class_means = np.zeros((num_classes, input_size))
class_deviations = class_deviation * np.ones(num_classes)
for k in xrange(num_classes):
    class_means[k, :] = np.random.randn(input_size)

X = np.zeros((num_examples, input_size))
y = np.zeros(num_examples, dtype=int)
for n in xrange(num_examples):
    label = int(np.floor(num_classes * np.random.rand()))
    x = class_means[label] + class_deviations[label] * np.random.randn(input_size)
    X[n, :] = x
    y[n] = label

X_train = X[:num_training_examples, :]
y_train = y[:num_training_examples]
X_dev = X[num_training_examples:(num_training_examples+num_dev_examples), :]
y_dev = y[num_training_examples:(num_training_examples+num_dev_examples)]

#assert num_classes == 2, pdb.set_trace()
assert input_size == 2, pdb.set_trace()

plot = True
if plot:
    ind_negative = np.nonzero(y_train == 0)[0]
    ind_positive = np.nonzero(y_train == 1)[0]

    plt.plot(X_train[ind_positive, 0], X_train[ind_positive, 1], 'bs')
    plt.plot(X_train[ind_negative, 0], X_train[ind_negative, 1], 'ro')


print 'Training...'

X_train_new = np.zeros_like(X_train)
width = 1.
for n in xrange(num_training_examples):
    x = X_train[n, :]
    X_others = X_train[range(n) + range(n+1, num_training_examples), :]
    all_distances = np.zeros(num_training_examples - 1)
    for m in xrange(X_others.shape[0]):
        d = x - X_others[m, :]
        distance = d.dot(d)
        all_distances[m] = distance

    weights = np.exp(-width*all_distances)
    weights /= sum(weights)
    average_rep = weights.dot(X_others)

    X_train_new[n, :] = average_rep

if plot:
    plt.plot(X_train_new[ind_positive, 0], X_train_new[ind_positive, 1], 'gs')
    plt.plot(X_train_new[ind_negative, 0], X_train_new[ind_negative, 1], 'co')

    plt.show()

X_dev_new = np.zeros_like(X_dev)
width = 1.
for n in xrange(num_dev_examples):
    x = X_dev[n, :]
    all_distances = np.zeros(num_training_examples)
    for m in xrange(X_train.shape[0]):
        d = x - X_train[m, :]
        distance = d.dot(d)
        all_distances[m] = distance

    weights = np.exp(-width*all_distances)
    weights /= sum(weights)
    average_rep = weights.dot(X_train)

    X_dev_new[n, :] = average_rep


train(X_train, y_train, regularization_constant, num_epochs, learning_rate, X_dev, y_dev)

pdb.set_trace()

train(X_train_new, y_train, regularization_constant, num_epochs, learning_rate, X_dev_new, y_dev)
