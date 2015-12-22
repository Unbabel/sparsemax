import theano
import numpy as np
import os
import time
import sys
import pdb

from theano import tensor as T
from collections import OrderedDict

sys.path.append('..')
import sparsemax_theano

class ffnn(object):
    def __init__(self, hidden_size, num_labels, num_features, embedding_size,
                 activation='logistic', attention=False):
        '''
        hidden_size :: dimension of the hidden layer
        num_labels :: number of labels
        num_features :: number of word embeddings in the vocabulary
        embedding_size :: dimension of the word embeddings
        context_size :: word window context size
        activation :: logistic or tanh
        '''
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_features = num_features
        self.embedding_size = embedding_size

        if activation == 'logistic':
            self.activation_function = T.nnet.sigmoid
        elif activation == 'tanh':
            self.activation_function = T.tanh
        else:
            raise NotImplementedError

        self.attention = attention

        self.create_parameters()
        self.initialize_parameters()

        # As many elements as words in the sentence.
        self.idxs = T.ivector()
        idxs = self.idxs

        emb = self.emb[:, idxs]
        x = T.mean(emb, axis=1)

        self.y = T.iscalar('y')  # label.
        y = self.y

        h = self.activation_function(T.dot(self.Wxh, x) + self.bh)

        #attention = True #False
        if self.attention:
            # l is a num_words-by-num_hidden matrix.
            l = self.activation_function(T.dot(self.Wel, emb).T + self.bl)
            #v = T.dot(l, T.dot(self.Whl, h))

            # m is a num_words-by-num_hidden matrix.
            m = self.activation_function(T.dot(self.Whm, h) + T.dot(self.Wlm, l.T).T + self.bm)
            v = T.dot(self.Wmp, m.T)

            #p = T.nnet.softmax(v)
            p = sparsemax_theano.sparsemax(v)
            xt = T.dot(emb, p.T)
            ht = self.activation_function(T.dot(self.Wxht, xt) + self.bht)
            #ht = self.activation_function(T.dot(self.Wxh, xt) + self.bh)
            s = T.nnet.softmax(T.dot(self.Why, ht.T).T + self.by)
        else:
            s = T.nnet.softmax(T.dot(self.Why, h.T).T + self.by)

        p_y_given_x_sentence = s[0] # check.
        self.y_pred = T.argmax(p_y_given_x_sentence)
        y_pred = self.y_pred

        self.num_mistakes = 1 - T.eq(y, y_pred)

        # cost and gradients and learning rate
        self.lr = T.scalar('lr')
        lr = self.lr
        self.sentence_nll = -T.log(p_y_given_x_sentence[y])
        #pdb.set_trace()
        params_to_update = self.params[1:]
        sentence_gradients = T.grad(self.sentence_nll, params_to_update)
        sentence_gradient_emb = T.grad(self.sentence_nll, emb)
        sentence_update_emb = [(self.emb, T.inc_subtensor(emb, -lr*sentence_gradient_emb))]
        self.sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(params_to_update, sentence_gradients))
        self.sentence_updates.update(sentence_update_emb)

        self.classify = theano.function(inputs=[idxs], outputs=[y_pred, p_y_given_x_sentence])


    def define_train(self, X, y, positions):
        i = T.lscalar()
        givens = {self.idxs: X[positions[i,0]:positions[i,1]],
                  self.y  : y[i]}

        self.train = theano.function(inputs=[i, self.lr],
                                     outputs=[self.sentence_nll, self.num_mistakes],
                                     updates=self.sentence_updates,
                                     givens=givens)

    def create_parameters(self):
        self.emb = theano.shared(np.zeros((self.embedding_size, self.num_features)).
                                 astype(theano.config.floatX))
        self.Wxh  = theano.shared(np.zeros((self.hidden_size,
                                            self.embedding_size)).
                                  astype(theano.config.floatX))
        self.Why   = theano.shared(np.zeros((self.num_labels, self.hidden_size)).
                                   astype(theano.config.floatX))
        self.bh  = theano.shared(np.zeros(self.hidden_size,
                                          dtype=theano.config.floatX))
        self.by   = theano.shared(np.zeros(self.num_labels,
                                           dtype=theano.config.floatX))

        if self.attention:
            self.Whm  = theano.shared(np.zeros((self.hidden_size,
                                                self.hidden_size)).
                                      astype(theano.config.floatX))
            self.Wlm  = theano.shared(np.zeros((self.hidden_size,
                                                self.hidden_size)).
                                      astype(theano.config.floatX))
            self.bm  = theano.shared(np.zeros(self.hidden_size,
                                              dtype=theano.config.floatX))
            self.Wmp  = theano.shared(np.zeros((self.hidden_size,
                                                self.hidden_size)).
                                      astype(theano.config.floatX))
            #self.Whl  = theano.shared(np.zeros((self.hidden_size,
            #                                    self.hidden_size)).
            #                          astype(theano.config.floatX))
            self.Wel  = theano.shared(np.zeros((self.hidden_size,
                                                self.embedding_size)).
                                      astype(theano.config.floatX))
            self.bl  = theano.shared(np.zeros(self.hidden_size,
                                              dtype=theano.config.floatX))
            self.Wxht  = theano.shared(np.zeros((self.hidden_size,
                                                 self.embedding_size)).
                                       astype(theano.config.floatX))
            self.bht  = theano.shared(np.zeros(self.hidden_size,
                                               dtype=theano.config.floatX))
        # bundle
        self.params = [ self.emb, self.Wxh, self.Why, self.bh, self.by ]
        self.names = ['embeddings', 'Wxh', 'Why', 'bh', 'by']

        if self.attention:
            self.params += [ self.Whm, self.Wlm, self.bm, self.Wmp, self.Wel, self.bl, self.Wxht, self.bht ]
            self.names += ['Whm', 'Wlm', 'bm', 'Wmp', 'Wel', 'bl', 'Wxht', 'bht']

    def initialize_parameters(self):
        for param in self.params:
            shape = param.get_value().shape
            if len(shape) == 1:
                n = shape[0]
                param.set_value(np.zeros(n, dtype=theano.config.floatX))
            elif len(shape) == 2:
                n_out, n_in = shape
                if self.activation_function == T.nnet.sigmoid:
                    coeff = 4.0
                else:
                    coeff = 1.0
                param.set_value(coeff*np.random.uniform(
                            low=-np.sqrt(6. / (n_in + n_out)),
                            high=np.sqrt(6. / (n_in + n_out)),
                            size=(n_out, n_in)).astype(theano.config.floatX))

        self.save('model')

    def save(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        for param, name in zip(self.params, self.names):
            #np.save(os.path.join(folder, name + '.npy'), param.get_value())
            print 'Saving %s...' % name
            f = open(os.path.join(folder, name + '.txt'), 'w')
            W = param.get_value()
            if len(W.shape) == 2:
                num_rows, num_cols = W.shape
                for i in xrange(num_rows):
                    f.write(' '.join([str(W[i,j]) for j in xrange(num_cols)]) + '\n')
            else:
                num_elems = W.shape[0]
                f.write(' '.join([str(W[i]) for i in xrange(num_elems)]) + '\n')
            f.close()

def read_dataset(dataset_file, input_alphabet={}, output_alphabet={}):
    if len(output_alphabet) == 0:
        add_features = True
    else:
        add_features = False
    f = open(dataset_file)
    X = []
    y = []
    for line in f:
        line = line.rstrip(' \n')
        fields = line.split(' ')
        label = fields[-1]
        if label not in output_alphabet:
            if add_features:
                lid = len(output_alphabet)
                output_alphabet[label] = lid
            else:
                assert False, pdb.set_trace()
        else:
            lid = output_alphabet[label]
        x = {}
        #pdb.set_trace()
        for field in fields[:-1]:
            key_value = field.split(':')
            assert len(key_value) == 2, pdb.set_trace()
            key = key_value[0]
            value = float(key_value[1])
            if '_' in key:
                continue # Ignore bigrams.
            if key not in input_alphabet:
                if not add_features:
                    continue
                fid = len(input_alphabet)
                input_alphabet[key] = fid
            else:
                fid = input_alphabet[key]
            assert fid not in x, pdb.set_trace()
            x[fid] = 1.0 #value
        X.append(x)
        y.append(lid)
    f.close()
    return X, y, input_alphabet, output_alphabet


def convert_data_to_shared_arrays(input_sequences, output_labels):
    num_sentences = len(output_labels)
    num_words = sum([len(sequence) for sequence in input_sequences])
    X = np.zeros(num_words, dtype='int32')
    y = np.zeros(num_sentences, dtype='int32')
    start_end_positions = np.zeros((num_sentences, 2))
    offset = 0
    for j in xrange(len(input_sequences)):
        input_sequence = input_sequences[j]
        start_position = offset
        end_position = offset + len(input_sequence)
        offset = end_position

        start_end_positions[j, :] = [start_position, end_position]
        X[start_position:end_position] = \
            np.asarray(input_sequence.keys()).astype('int32')
        y[j] = output_labels[j]

    th_X  = theano.shared(X.astype('int32'), borrow=True)
    th_y  = theano.shared(y.astype('int32'), borrow=True)
    th_start_end_positions = theano.shared(start_end_positions.astype('int32'),
                                           borrow=True)
    return th_X, th_y, th_start_end_positions



if __name__ == '__main__':
    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    test_file = sys.argv[3]
    num_hidden_units = int(sys.argv[4])
    num_epochs = int(sys.argv[5])
    learning_rate = float(sys.argv[6])

    embedding_dimension = 64

    # Load the training dataset.
    #input_sequences, output_sequences, input_alphabet, output_alphabet = \
    #    read_dataset(train_file, cutoff=1)

    input_sequences, output_labels, input_alphabet, output_alphabet = read_dataset(train_file)

    #pdb.set_trace()
    num_labels = len(set(output_labels))
    num_features = 1+max([max(seq.keys()) for seq in input_sequences])
    print 'Number of labels: ', num_labels
    print 'Number of features: ', num_features

    #print 'Number of labels: ', len(output_alphabet)
    #print 'Number of features: ', len(input_alphabet)

    # Load the validation dataset.
    #input_sequences_dev, output_sequences, _, _ = \
    #    read_dataset(dev_file, input_alphabet, output_alphabet, \
    #                 locked_alphabets=True)

    input_sequences_dev, output_labels_dev, _, _ = read_dataset(dev_file, input_alphabet, output_alphabet)

    # Load the test dataset.
    #input_sequences_test, output_sequences_test, _, _ = \
    #    read_dataset(test_file, input_alphabet, output_alphabet, \
    #                 locked_alphabets=True)

    input_sequences_test, output_labels_test, _, _ = read_dataset(test_file, input_alphabet, output_alphabet)

    #pdb.set_trace()

    print 'Converting data to shared arrays...'
    X, y, positions = \
        convert_data_to_shared_arrays(input_sequences, output_labels)
    X_dev, y_dev, positions_dev = \
        convert_data_to_shared_arrays(input_sequences_dev,
                                      output_labels_dev)
    X_test, y_test, positions_test = \
        convert_data_to_shared_arrays(input_sequences_test,
                                      output_labels_test)

    #pdb.set_trace()
    #print 'Number of labels: ', len(set(y.get_value()))
    #print 'Number of features: ', 1+max(X.get_value())

    # instanciate the model
    np.random.seed(1234)
    #ffnn = ffnn(num_hidden_units,
    #            len(output_alphabet),
    #            len(input_alphabet),
    #            embedding_dimension,
    #            activation='logistic')
    ffnn = ffnn(num_hidden_units,
                len(set(y.get_value())),
                1+max(X.get_value()),
                embedding_dimension,
                activation='tanh', #'logistic',
                attention=True)

    print 'Defining train...'
    ffnn.define_train(X, y, positions)

    print 'Starting training...'
    for epoch in xrange(num_epochs):
        tic = time.time()
        total_loss = 0.
        total_num_mistakes = 0
        for j in xrange(len(positions.get_value())):
            loss, num_mistakes = ffnn.train(j, learning_rate)
            total_loss += loss
            total_num_mistakes += num_mistakes
        num_sentences = len(y.get_value())
        acc_train = 1. - float(total_num_mistakes) / num_sentences

        ffnn.save('model.%d' % (epoch+1))

        # evaluation // back into the real world : idx -> words
        accuracies = []
        for inputs, outputs, pos in zip([X_dev, X_test], [y_dev, y_test], [positions_dev, positions_test]):
            acc = 0.
            num_sentences = 0
            for j in xrange(len(pos.get_value())):
                #input_sequence = inputs[j]
                #output_sequence_gold = outputs[j]
                #context_sequence = contextwin(input_sequence, window_size)
                start = pos.get_value()[j,0]
                end = pos.get_value()[j,1]
                xx = inputs.get_value()[start:end]
                yy = outputs.get_value()[j]
                output_label_pred, p_y_given_x_sentence = ffnn.classify(xx)

                acc += float(np.sum(yy == output_label_pred))
                num_sentences += 1
            acc /= num_sentences
            accuracies.append(acc)
        acc_dev = accuracies[0]
        acc_test = accuracies[1]


        print 'Iter: %d, Obj: %f, Acc train: %f, Acc dev: %f, Acc test: %f, Time: %f' % \
            (epoch+1, total_loss, acc_train, acc_dev, acc_test, time.time() - tic)
