
"""
cs534 Implementation Assignment 2
"""

import numpy as np


class NaiveBayes(object):

    def __init__(self, train_set, train_labels, dev_set, dev_labels):
        """
        Constructor
        """

        # instance variables
        self.train_set = np.array(train_set)
        self.train_labels = np.array(train_labels)
        self.dev_set = np.array(dev_set)
        self.dev_labels = np.array(dev_labels)
        self.V = None
        self.V_card = None
        self.features = {}
        self.classes = []
        self.Nk = {}
        self.priors = {}
        self.likelihoods = {}

        # generate vocubulary on instaintiation
        print 'Generating vocabulary...'
        self._get_vocabulary()


    def train_classifier(self, model='binomial', alpha=1, beta=2):
        """
        Training for Bernoulli or Multinomial

        use alpha and beta to configure Laplace smoothing. This will come in
        handy for the Priors and Overfitting part in the assignment
        """

        print 'Retreiving classes...'
        self._get_classes()
        print 'Generating dense features...'
        self._get_features(model=model)
        print 'Estimating priors...'
        self._compute_priors()
        print 'Estimating likelihoods...'
        self._compute_likelihoods(alpha, beta, model=model)

    def test_classifier(self, model='binomial'):
        """
        Predict the class for each document
        """

        pred_labels = []

        print 'Classifying dev set...'
        for i in range(len(self.dev_set)):
            if model == 'binomial':
                doc = self._get_binomial_feature(self.dev_set[i])
            if model == 'multinomial':
                doc = self._get_multinomial_feature(self.dev_set[i])

            pred_labels.append(self._predict(doc, model=model))

        self.pred_labels = pred_labels

    def get_confusion_matrix(self):
        """
        Print confusion matrix to console
        """

        cm = np.zeros((len(self.classes), len(self.classes)))
        _class = self.classes

        for i in range(len(self.pred_labels)):
            if self.pred_labels[i] == self.dev_labels[i]:
                if self.pred_labels[i] == _class[0]:
                    cm[0,0] += 1
                if self.pred_labels[i] == _class[1]:
                    cm[1,1] += 1
            else:
                if self.pred_labels[i] == _class[0]:
                    cm[0,1] += 1
                if self.pred_labels[i] == _class[1]:
                    cm[1,0] += 1

        cm = (cm/float(len(self.pred_labels)))*100

        print "{:20s} {:20s} {:20s}\n{:20s} {:2.2f} {:2.2f}\n{:20s} {:2.2f} {:2.2f}".format(
                '', 'True ' + _class[0], 'True '+ _class[1], 'Pred ' + _class[0],
                cm[0,0], cm[0,1], 'Pred ' + _class[1], cm[1,0], cm[1,1])

    def _get_vocabulary(self):
        """
        Constructs a unique set of words found in all training documents
        """

        V_tmp = [ item for sub in self.train_set for item in sub ]
        V_tmp = np.array(V_tmp)
        self.V = np.unique(V_tmp)
        self.V_card = len(self.V)


    def _get_classes(self):
        """
        Get the class labels and number of documents in each class
        """

        self.classes, Nk = np.unique(self.train_labels, return_counts=True)

        # set classes as feature, priors and likelihood and total words keys
        for i in range(len(self.classes)):
            self.Nk[self.classes[i]] = Nk[i]
            self.features[self.classes[i]] = []
            self.priors[self.classes[i]] = 0
            self.likelihoods[self.classes[i]] = 0

    def _get_features(self, model='binomial'):
        """
        Convert the documents to value features

        If model is binomial, the features will be binarized.
        If model is multinomial, the feature are represented as frequencies
        """

        for i in range(len(self.train_set)):
            if model == 'binomial':
                f = self._get_binomial_feature(self.train_set[i])
            if model == 'multinomial':
                f = self._get_multinomial_feature(self.train_set[i])
            self.features[self.train_labels[i]].append(f)

    def _get_binomial_feature(self, doc):
        """
        Helper function for `_get_features'
        """

        feature = np.zeros(self.V_card)
        idx_set = [ np.where(self.V == i) for i in doc ]
        for i in idx_set:
            feature[i] = 1

        return feature

    def _get_multinomial_feature(self, doc):
        """
        Helper function for `_get_features'
        """

        # make doc a numpy array for np.count_nonzero method
        doc = np.array(doc)

        feature = np.zeros(self.V_card)
        idx_set = [ np.where(self.V == i)[0] for i in doc ]
        for i in idx_set:
            if i:
                feature[i] = np.count_nonzero(doc == self.V[i])

        return feature

    def _compute_priors(self):
        """
        Maximum likelihood estimation
        """

        N = float(sum(self.Nk.values()))
        for _class in self.classes:
            self.priors[_class] = self.Nk[_class]/N

    def _compute_likelihoods(self, alpha, beta, model='binomial'):
        """
        Estimate word likelihoods
        """

        for _class in self.classes:
            n = np.sum(self.features[_class], axis=0)
            Tk = np.sum(n)
            if model == 'binomial':
                self.likelihoods[_class] = (n+alpha)/float(self.Nk[_class]+beta)
            if model == 'multinomial':
                self.likelihoods[_class] = (n+alpha)/float(Tk+beta)

    def _bernoulli(self, doc, _class):
        """
        Bernoulli model
        """

        log_sum = 0
        for i in range(len(doc)):
            log_sum += (doc[i]*np.log(self.likelihoods[_class][i]))+((1-doc[i])*np.log(1-self.likelihoods[_class][i]))

        return log_sum

    def _multinomial(self, doc, _class):
        """
        Multinomial model
        """

        log_sum = 0
        for i in range(len(doc)):
            log_sum += doc[i]*np.log(self.likelihoods[_class][i])

        return log_sum


    def _predict(self, doc, model='binomial'):
        """
        Classify a document
        """

        max_score = -np.inf
        pred_class = None

        for _class in self.classes:

            # log prior
            log_sum = np.log(self.priors[_class])

            if model == 'binomial':
                log_sum += self._bernoulli(doc, _class)
            if model == 'multinomial':
                log_sum += self._multinomial(doc, _class)

            if log_sum > max_score:
                max_score = log_sum
                pred_class = _class

        return pred_class


if __name__ == '__main__':

    # read documents and labels
    with open('clintontrump-data/clintontrump.tweets.train', 'r') as f:
        tweets_train = f.readlines()

    tweets_train = [ w.split() for w in tweets_train ]

    with open('clintontrump-data/clintontrump.tweets.dev', 'r') as f:
        tweets_dev = f.readlines()

    tweets_dev = [ w.split() for w in tweets_dev ]

    with open('clintontrump-data/clintontrump.labels.train', 'r') as f:
        labels_train = f.read().split()

    with open('clintontrump-data/clintontrump.labels.dev', 'r') as f:
        labels_dev = f.read().split()


    # ===============
    # Bernoulli Model
    # ===============
    bernoulli = NaiveBayes(tweets_train, labels_train, tweets_dev, labels_dev)
    bernoulli.train_classifier(model='binomial', alpha=1, beta=2)

    bernoulli.test_classifier()

    # performance
    bernoulli.get_confusion_matrix()

    # =================
    # Multinomial Model
    # =================

    multi = NaiveBayes(tweets_train, labels_train, tweets_dev, labels_dev)
    multi.train_classifier(model='multinomial', alpha=1, beta=multi.V_card)

    multi.test_classifier(model='multinomial')

    # performance
    multi.get_confusion_matrix()


