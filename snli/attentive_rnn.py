import theano
import numpy as np
import os
import time
import sys
import pdb

from theano import tensor as T
from collections import OrderedDict


class rnn(object):
    def __init__(self, hidden_size, num_labels, num_features, embedding_size,
                 activation='logistic'):
        '''
        hidden_size :: dimension of the hidden layer
        num_labels :: number of labels
        num_features :: number of word embeddings in the vocabulary
        embedding_size :: dimension of the word embeddings
        activation :: logistic or tanh
        '''
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.num_features = num_features
        self.embedding_size = embedding_size
        self.bidirectional = True

        if activation == 'logistic':
            self.activation_function = T.nnet.sigmoid
        elif activation == 'tanh':
            self.activation_function = T.tanh
        else:
            raise NotImplementedError

        self.create_parameters()
        self.initialize_parameters()

        # As many elements as words in the sentence.
        self.idxs = T.ivector()
        idxs = self.idxs

        emb = self.emb[:, idxs]
        x = emb.T

        self.y = T.iscalar('y')  # label.
        y = self.y

        #[h, s], _ = theano.scan(fn=self.recurrence_old,
        #                        sequences=x,
        #                        outputs_info=[self.h0, None],
        #                        n_steps=x.shape[0])

        h, _ = theano.scan(fn=self.recurrence,
                           sequences=x,
                           outputs_info=self.h0,
                           n_steps=x.shape[0])

        if self.bidirectional:
            l, _ = theano.scan(fn=self.recurrence_right_to_left,
                               sequences=x[::-1, :],
                               outputs_info=self.l0,
                               n_steps=x.shape[0])
            l = l[::-1, :]
            #s = T.nnet.softmax(T.dot(self.Why, h.T).T +
            #                   T.dot(self.Wly, l.T).T + self.by)
            s = T.nnet.softmax(T.dot(self.Why, h[-1, :]) +
                               T.dot(self.Wly, l[0, :]) + self.by)
        else:
            #s = T.nnet.softmax(T.dot(self.Why, h.T).T + self.by)
            s = T.nnet.softmax(T.dot(self.Why, h[-1, :]) + self.by)

        p_y_given_x_sentence = s[0] # check.
        self.y_pred = T.argmax(p_y_given_x_sentence)
        y_pred = self.y_pred

        self.num_mistakes = 1 - T.eq(y, y_pred)

        # cost and gradients and learning rate
        self.lr = T.scalar('lr')
        lr = self.lr
        #self.sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
        #                       [T.arange(x.shape[0]), y])
        self.sentence_nll = -T.log(p_y_given_x_sentence[y])

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
        self.Whh  = theano.shared(np.zeros((self.hidden_size, self.hidden_size)).
                                  astype(theano.config.floatX))
        self.Why   = theano.shared(np.zeros((self.num_labels, self.hidden_size)).
                                   astype(theano.config.floatX))
        self.bh  = theano.shared(np.zeros(self.hidden_size,
                                          dtype=theano.config.floatX))
        self.by   = theano.shared(np.zeros(self.num_labels,
                                           dtype=theano.config.floatX))
        self.h0  = theano.shared(np.zeros(self.hidden_size,
                                          dtype=theano.config.floatX))

        # bundle
        self.params = [ self.emb, self.Wxh, self.Whh, self.Why, self.bh, self.by, self.h0 ]
        self.names = ['embeddings', 'Wxh', 'Whh', 'Why', 'bh', 'by', 'h0']

        if self.bidirectional:
            self.Wxl  = theano.shared(np.zeros((self.hidden_size,
                                                self.embedding_size)).
                                      astype(theano.config.floatX))
            self.Wll  = theano.shared(np.zeros((self.hidden_size, self.hidden_size)).
                                      astype(theano.config.floatX))
            self.Wly   = theano.shared(np.zeros((self.num_labels, self.hidden_size)).
                                       astype(theano.config.floatX))
            self.bl  = theano.shared(np.zeros(self.hidden_size,
                                              dtype=theano.config.floatX))
            self.l0  = theano.shared(np.zeros(self.hidden_size,
                                              dtype=theano.config.floatX))

            self.params += [ self.Wxl, self.Wll, self.Wly, self.bl, self.l0 ]
            self.names += [ 'Wxl', 'Wll', 'Wly', 'bl', 'l0']

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

    def recurrence(self, x_t, h_tm1):
        h_t = self.activation_function(T.dot(self.Wxh, x_t) + T.dot(self.Whh, h_tm1) + self.bh)
        return h_t

    def recurrence_right_to_left(self, x_t, l_tp1):
        l_t = self.activation_function(T.dot(self.Wxl, x_t) + T.dot(self.Wll, l_tp1) + self.bl)
        return l_t

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


class rnn_gru(rnn):
    def __init__(self, hidden_size, num_labels, num_features, embedding_size,
                 activation='logistic'):
        '''
        hidden_size :: dimension of the hidden layer
        num_labels :: number of labels
        num_features :: number of word embeddings in the vocabulary
        embedding_size :: dimension of the word embeddings
        context_size :: word window context size
        activation :: logistic or tanh
        '''
        rnn.__init__(self, hidden_size, num_labels, num_features,
                     embedding_size, activation=activation)

    def create_parameters(self):
        rnn.create_parameters(self)
        # Create specific GRU parameters.
        self.Wxz  = theano.shared(np.zeros((self.hidden_size,
                                            self.embedding_size)).
                                  astype(theano.config.floatX))
        self.Whz  = theano.shared(np.zeros((self.hidden_size, self.hidden_size)).
                                  astype(theano.config.floatX))
        self.Wxr  = theano.shared(np.zeros((self.hidden_size,
                                            self.embedding_size)).
                                  astype(theano.config.floatX))
        self.Whr  = theano.shared(np.zeros((self.hidden_size, self.hidden_size)).
                                  astype(theano.config.floatX))
        self.bz  = theano.shared(np.zeros(self.hidden_size,
                                          dtype=theano.config.floatX))
        self.br  = theano.shared(np.zeros(self.hidden_size,
                                          dtype=theano.config.floatX))

        self.params += [ self.Wxz, self.Whz, self.Wxr, self.Whr, self.bz, self.br  ]
        self.names += ['Wxz', 'Whz', 'Wxr', 'Whr', 'bz', 'br']

        if self.bidirectional:
            self.Wxz_r  = theano.shared(np.zeros((self.hidden_size,
                                                self.embedding_size)).
                                      astype(theano.config.floatX))
            self.Wlz  = theano.shared(np.zeros((self.hidden_size, self.hidden_size)).
                                      astype(theano.config.floatX))
            self.Wxr_r  = theano.shared(np.zeros((self.hidden_size,
                                                self.embedding_size)).
                                      astype(theano.config.floatX))
            self.Wlr  = theano.shared(np.zeros((self.hidden_size, self.hidden_size)).
                                      astype(theano.config.floatX))
            self.bz_r  = theano.shared(np.zeros(self.hidden_size,
                                              dtype=theano.config.floatX))
            self.br_r  = theano.shared(np.zeros(self.hidden_size,
                                              dtype=theano.config.floatX))

            self.params += [ self.Wxz_r, self.Wlz, self.Wxr_r, self.Wlr, self.bz_r, self.br_r ]
            self.names += [ 'Wxz_r', 'Wlz', 'Wxr_r', 'Wlr', 'bz_r', 'br_r']


    def recurrence(self, x_t, h_tm1):
        z_t = T.nnet.sigmoid(T.dot(self.Wxz, x_t) + T.dot(self.Whz, h_tm1) + self.bz)
        r_t = T.nnet.sigmoid(T.dot(self.Wxr, x_t) + T.dot(self.Whr, h_tm1) + self.br)
        hu_t = self.activation_function(T.dot(self.Wxh, x_t) + T.dot(self.Whh, r_t * h_tm1) + self.bh)
        h_t = (1-z_t) * h_tm1 + z_t * hu_t
        return h_t

    def recurrence_right_to_left(self, x_t, l_tp1):
        z_t = T.nnet.sigmoid(T.dot(self.Wxz_r, x_t) + T.dot(self.Wlz, l_tp1) + self.bz_r)
        r_t = T.nnet.sigmoid(T.dot(self.Wxr_r, x_t) + T.dot(self.Wlr, l_tp1) + self.br_r)
        lu_t = self.activation_function(T.dot(self.Wxl, x_t) + T.dot(self.Wll, r_t * l_tp1) + self.bl)
        l_t = (1-z_t) * l_tp1 + z_t * lu_t
        return l_t


def read_dataset(dataset_file, input_alphabet={'__START__': 0, '__UNK__': 1}, \
                 output_alphabet={}, \
                 locked_alphabets=False, cutoff=1):
    input_sequences = []
    output_labels = []
    word_freq = [0 for word in input_alphabet]
    words = []
    f = open(dataset_file)
    for line in f:
        line = line.rstrip('\n')        
        fields = line.split('\t')
        assert len(fields) == 3, pdb.set_trace()
        label = fields[0]
        premise = fields[1]
        hypothesis = fields[2]
        premise_words = premise.split(' ')
        hypothesis_words = hypothesis.split(' ')
        sentence = premise_words + ['__START__'] + hypothesis_words
        w = []
        for word in sentence:
            if word in input_alphabet:
                wid = input_alphabet[word]
                word_freq[wid] += 1
            elif not locked_alphabets:
                wid = len(input_alphabet)
                input_alphabet[word] = wid
                word_freq.append(1)
            else:
                wid = 1 # Unknown symbol.
            w.append(wid)
        if label in output_alphabet:
            tid = output_alphabet[label]
        else:
            tid = len(output_alphabet)
            output_alphabet[label] = tid
        t = tid
        input_sequences.append(np.array(w))
        output_labels.append(t)
    f.close()

    if not locked_alphabets:
        print 'Number of words before cutoff: %d' % len(input_alphabet)
        words = ['' for wid in xrange(len(word_freq))]
        for word, wid in input_alphabet.iteritems():
            words[wid] = word
        new_input_alphabet = {}
        for wid in xrange(len(word_freq)):
            if wid == 0 or wid == 1 or word_freq[wid] > cutoff:
                new_wid = len(new_input_alphabet)
                new_input_alphabet[words[wid]] = new_wid
        input_alphabet = new_input_alphabet
        for input_sequence in input_sequences:
            for t in xrange(len(input_sequence)):
                word = words[input_sequence[t]]
                if word in new_input_alphabet:
                    input_sequence[t] = new_input_alphabet[word]
                else:
                    input_sequence[t] = 1 # Unknown symbol.
        print 'Number of words after cutoff: %d' % len(new_input_alphabet)

    return input_sequences, output_labels, input_alphabet, output_alphabet 


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
            np.asarray(input_sequence).astype('int32')
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
    window_size = 3 #1

    # Load the training dataset.
    input_sequences, output_sequences, input_alphabet, output_alphabet = \
        read_dataset(train_file, cutoff=1)

    print 'Number of labels: ', len(output_alphabet)
    print 'Number of features: ', len(input_alphabet)

    # Load the validation dataset.
    input_sequences_dev, output_sequences_dev, _, _ = \
        read_dataset(dev_file, input_alphabet, output_alphabet, \
                     locked_alphabets=True)

    # Load the test dataset.
    input_sequences_test, output_sequences_test, _, _ = \
        read_dataset(test_file, input_alphabet, output_alphabet, \
                     locked_alphabets=True)

    print 'Converting data to shared arrays...'
    X, y, positions = \
        convert_data_to_shared_arrays(input_sequences, output_sequences)
    X_dev, y_dev, positions_dev = \
        convert_data_to_shared_arrays(input_sequences_dev,
                                      output_sequences_dev)
    X_test, y_test, positions_test = \
        convert_data_to_shared_arrays(input_sequences_test,
                                      output_sequences_test)

    #pdb.set_trace()

    # instanciate the model
    np.random.seed(1234)
    #random.seed(1234)
    #rnn = rnn_gru(num_hidden_units,
    rnn = rnn_gru(num_hidden_units,
              len(output_alphabet),
              len(input_alphabet),
              embedding_dimension,
              activation='tanh')


    print 'Defining train...'
    rnn.define_train(X, y, positions)

    print 'Starting training...'
    for epoch in xrange(num_epochs):
        tic = time.time()
        total_loss = 0.
        total_num_mistakes = 0
        for j in xrange(len(positions.get_value())):
            loss, num_mistakes = rnn.train(j, learning_rate)
            total_loss += loss
            total_num_mistakes += num_mistakes
        num_sentences = len(y.get_value())
        acc_train = 1. - float(total_num_mistakes) / num_sentences

        rnn.save('model.%d' % (epoch+1))

        print 'Testing...'

        # evaluation // back into the real world : idx -> words
        accuracies = []
        for inputs, outputs, pos in zip([X_dev, X_test], [y_dev, y_test], [positions_dev, positions_test]):
            acc = 0.
            num_sentences = 0
            for j in xrange(len(pos.get_value())):
                start = pos.get_value()[j,0]
                end = pos.get_value()[j,1]
                xx = inputs.get_value()[start:end]
                yy = outputs.get_value()[j]
                output_label_pred, p_y_given_x_sentence = rnn.classify(xx)

                acc += float(np.sum(yy == output_label_pred))
                num_sentences += 1
            acc /= num_sentences
            accuracies.append(acc)
        acc_dev = accuracies[0]
        acc_test = accuracies[1]


        print 'Iter: %d, Obj: %f, Acc train: %f, Acc dev: %f, Acc test: %f, Time: %f' % \
            (epoch+1, total_loss, acc_train, acc_dev, acc_test, time.time() - tic)
