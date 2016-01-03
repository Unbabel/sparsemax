import array
import numpy as np
import scipy.sparse as sp
#from sklearn.utils import check_random_state
#from sklearn.preprocessing import MultiLabelBinarizer
import pdb

def make_multilabel_classification(n_samples=100, n_features=20, n_classes=5,
                                   n_labels=2, length=50, allow_unlabeled=True,
                                   sparse=False, return_indicator='dense',
                                   return_distributions=False,
                                   use_class_proportions=False,
                                   random_state=None):
    """Generate a random multilabel classification problem.

    For each sample, the generative process is:
        - pick the number of labels: n ~ Poisson(n_labels)
        - n times, choose a class c: c ~ Multinomial(theta)
        - pick the document length: k ~ Poisson(length)
        - k times, choose a word: w ~ Multinomial(theta_c)

    In the above process, rejection sampling is used to make sure that
    n is never zero or more than `n_classes`, and that the document length
    is never zero. Likewise, we reject classes which have already been chosen.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, optional (default=100)
        The number of samples.

    n_features : int, optional (default=20)
        The total number of features.

    n_classes : int, optional (default=5)
        The number of classes of the classification problem.

    n_labels : int, optional (default=2)
        The average number of labels per instance. More precisely, the number
        of labels per sample is drawn from a Poisson distribution with
        ``n_labels`` as its expected value, but samples are bounded (using
        rejection sampling) by ``n_classes``, and must be nonzero if
        ``allow_unlabeled`` is False.

    length : int, optional (default=50)
        The sum of the features (number of words if documents) is drawn from
        a Poisson distribution with this expected value.

    allow_unlabeled : bool, optional (default=True)
        If ``True``, some instances might not belong to any class.

    sparse : bool, optional (default=False)
        If ``True``, return a sparse feature matrix

    return_indicator : 'dense' (default) | 'sparse' | False
        If ``dense`` return ``Y`` in the dense binary indicator format. If
        ``'sparse'`` return ``Y`` in the sparse binary indicator format.
        ``False`` returns a list of lists of labels.

    return_distributions : bool, optional (default=False)
        If ``True``, return the prior class probability and conditional
        probabilities of features given classes, from which the data was
        drawn.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Returns
    -------
    X : array of shape [n_samples, n_features]
        The generated samples.

    Y : array or sparse CSR matrix of shape [n_samples, n_classes]
        The label sets.

    p_c : array, shape [n_classes]
        The probability of each class being drawn. Only returned if
        ``return_distributions=True``.

    p_w_c : array, shape [n_features, n_classes]
        The probability of each feature being drawn given each class.
        Only returned if ``return_distributions=True``.

    """
    #generator = check_random_state(random_state)
    generator = np.random
    p_c = generator.rand(n_classes)
    #p_c = np.random.rand(n_classes)
    p_c /= p_c.sum()
    cumulative_p_c = np.cumsum(p_c)
    p_w_c = generator.rand(n_features, n_classes)
    #p_w_c = np.random.rand(n_features, n_classes)
    p_w_c /= np.sum(p_w_c, axis=0)

    def sample_example():
        _, n_classes = p_w_c.shape

        # pick a nonzero number of labels per document by rejection sampling
        y_size = n_classes + 1
        while (not allow_unlabeled and y_size == 0) or y_size > n_classes:
            y_size = generator.poisson(n_labels)

        # pick n classes
        y = set()
        while len(y) != y_size:
            # pick a class with probability P(c)
            c = np.searchsorted(cumulative_p_c,
                                generator.rand(y_size - len(y)))
            y.update(c)
        y = list(y)

        # pick a non-zero document length by rejection sampling
        n_words = 0
        while n_words == 0:
            n_words = generator.poisson(length)

        # generate a document of length n_words
        if len(y) == 0:
            # if sample does not belong to any class, generate noise word
            words = generator.randint(n_features, size=n_words)
            return words, y

        # sample words with replacement from selected classes
        if use_class_proportions:
            class_proportions = generator.rand(len(y))
            class_proportions /= sum(class_proportions)
            cumulative_p_w_sample = p_w_c.take(y, axis=1).dot(class_proportions).cumsum()
            words = np.searchsorted(cumulative_p_w_sample, generator.rand(n_words))
            return words, y, class_proportions
        else:
            cumulative_p_w_sample = p_w_c.take(y, axis=1).sum(axis=1).cumsum()
            cumulative_p_w_sample /= cumulative_p_w_sample[-1]
            words = np.searchsorted(cumulative_p_w_sample, generator.rand(n_words))
            return words, y

    X_indices = array.array('i')
    X_indptr = array.array('i', [0])
    Y = []
    all_class_proportions = []
    for i in range(n_samples):
        if use_class_proportions:
            words, y, class_proportions = sample_example()
            X_indices.extend(words)
            X_indptr.append(len(X_indices))
            Y.append(y)
            all_class_proportions.append(class_proportions)
        else:
            words, y = sample_example()
            X_indices.extend(words)
            X_indptr.append(len(X_indices))
            Y.append(y)
    X_data = np.ones(len(X_indices), dtype=np.float64)
    X = sp.csr_matrix((X_data, X_indices, X_indptr),
                      shape=(n_samples, n_features))
    X.sum_duplicates()
    if not sparse:
        X = X.toarray()

    if use_class_proportions:
        Y_dense = np.zeros((X.shape[0], n_classes), dtype=float)
    else:
        Y_dense = np.zeros((X.shape[0], n_classes), dtype=int)
    for i, y in enumerate(Y):
        if use_class_proportions:
            Y_dense[i, y] = all_class_proportions[i]
        else:
            Y_dense[i, y] = 1
    Y = Y_dense

    # return_indicator can be True due to backward compatibility
    #if return_indicator in (True, 'sparse', 'dense'):
    #    lb = MultiLabelBinarizer(sparse_output=(return_indicator == 'sparse'))
    #    Y = lb.fit([range(n_classes)]).transform(Y)
    #elif return_indicator is not False:
    #    raise ValueError("return_indicator must be either 'sparse', 'dense' "
    #                     'or False.')
    if return_distributions:
        return X, Y, p_c, p_w_c
    return X, Y
