
"""
idtree.py
cs534: Implementation assignment 3
"""

import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
import random

class TrainTree(object):

    def __init__(self, data_csv, k):
        """ Constructor """

        data = np.genfromtxt(data_csv, delimiter=';')
        self.x = data[:,0:4].T
        self.y = data[:,4]

        # stopping criteria
        self.k = k

    def _entropy(self, y):
        """
        Compute entropy from target values: H(y) = -sum(pi*log2(pi))
        """

        # number of values in y to compute prior
        N = float(len(y))

        # sum pi*log2(pi) for each class k
        H = 0.0
        for k in set(y):
            pi = np.count_nonzero(y==k)/N
            H -= pi*np.log2(pi)

        return H

    def _partition(self, x, y, feature_bagging = [0,1,2,3]):
        """
        Enumerate all 'meaningful' paritions
        """
        cnt = 0 # counter for partition dictionary
        P = {}

        # for each feature i, sort data based on i and find theata where
        # class label changes then split and add both partitions to P
        for i in feature_bagging:

            # get the sorting indicies for feature x[i]
            s = x[i].argsort()
            # sort the data according to s
            sx = x[:,s]
            sy = y[s]

            # set the current label to sy[0] then move right
            cur_lab = sy[0]
            for j in range(1, len(sy)):
                if sy[j] != cur_lab:
                    # update the current label and iterate counter
                    cur_lab = np.copy(sy[j])
                    cnt += 1
                    # compute theta (half the dist in x[i] where lab changes)
                    theta = (sx[i,j-1] + sx[i,j])/2.0
                    if theta != sx[i,j-1]:
                        # add parition to P
                        P[cnt] = {'feature': i,
                                'theta'  : theta,
                                'T'      : {'x': sx[:,0:j], 'y': sy[0:j]},
                                'F'      : {'x': sx[:,j:], 'y': sy[j:]}}

        return P

    def _information_gain(self, H, P):
        """
        Return the partition that results in the maximum information gain:
        I(Y,X) = H(y) - sum(P(X=x)H(Y|X=x)
        """
        I = 0.0
        # dictionary of all I for each feature and theta for report
        Itf = {}
        # best partition base on I(Y,X)
        Pb = None

        # for each parition p compute information gain
        for p in P.keys():
            Ht = self._entropy(P[p]['T']['y'])
            Hf = self._entropy(P[p]['F']['y'])
            # P(X=x)
            N = float(len(P[p]['T']['y']) + len(P[p]['F']['y']))
            pt = len(P[p]['T']['y'])/N
            pf = len(P[p]['F']['y'])/N
            # compute information gain for partition
            Itmp = H - (pt*Ht + pf*Hf)
            if P[p]['feature'] not in Itf.keys():
                Itf[P[p]['feature']] = {}
            Itf[P[p]['feature']][P[p]['theta']] = Itmp
            if Itmp > I:
                I = Itmp
                Pb = p

        return Pb, I, Itf

    def train(self, x, y, feature_bagging=False):
        """
        Recursive method for top down induction
        """

        # dictionary to store tree
        tree = {}

        # if y only has 1 label then return it
        if len(set(y)) == 1:
            rv = np.zeros(len(np.unique(self.y)))
            i = int(y[0])
            rv[i] = len(y)
            return rv

        # if number of instances in y is less than k then return y
        if len(y) < self.k:
            rv = np.zeros(len(np.unique(self.y)))
            u,c = np.unique(y, return_counts=True)
            for i in range(len(c)):
                rv[int(u[i])] = c[i]
            return rv

        # compute entropy
        H = self._entropy(y)

        # enumerate partitions
        # random numbers for feature bagging
        if feature_bagging:
            r1 = random.randint(0,3)
            r2 = random.randint(0,3)
            while r1 == r2:
                r2 = random.randint(0,3)
            P = self._partition(x,y, feature_bagging=[r1,r2])
        else:
            P = self._partition(x,y)

        # get best partition based on I(Y,X)
        Pb,I,junk = self._information_gain(H, P)

        # if there is no gain then return y
        if I <= 0:
            rv = np.zeros(len(np.unique(self.y)))
            i = int(y[0])
            rv[i] = len(y)
            return rv

        # for both evaluations (T and F) recursively run train()
        for i in range(2):
            tree['x[{0}]<={1}'.format(P[Pb]['feature'], P[Pb]['theta'])] = self.train(P[Pb]['T']['x'], P[Pb]['T']['y'])
            tree['x[{0}]>{1}'.format(P[Pb]['feature'], P[Pb]['theta'])] = self.train(P[Pb]['F']['x'], P[Pb]['F']['y'])

        return tree

    def get_accuracy(self, tree):

        self.missed = 0.0
        self._get_missed(tree)
        accuracy = (1 - (self.missed/len(self.x[0])))*100
        return accuracy


    def _get_missed(self, tree):
        """
        Sum the total number of missclassifications
        """

        for v in tree.values():
            if isinstance(v, dict):
                self._get_missed(v)
            else:
                if np.count_nonzero(v) > 1:
                    self.missed += v[np.where(v != 0)].min()


class RandomForest(TrainTree):

    def __init__(self, train_csv, k, L):
        """
        Subclass of TrainTree. Builds L number of trees
        """
        # call super constructor
        super(RandomForest, self).__init__(train_csv, k)

        self.forest = []
        self.accuracy = []

        for i in range(0, L):
            x,y = self.randomize_wr(self.x, self.y, len(self.x[0]))
            tree = self.train(x, y, feature_bagging=True)
            self.forest.append(tree)
        for tree in self.forest:
            self.accuracy.append(self.get_accuracy(tree))
        self.accuracy = np.mean(self.accuracy)

    def randomize_wr(self, x, y, n):
        """
        Generates a random dataset of size n drawing with replacement from
        original dataset
        """

        xr = np.zeros((len(x), n))
        yr = np.zeros(n)
        for s in range(0, n):
            r = random.randint(0, len(x[0])-1)
            xr[:,s] = np.copy(x[:,r])
            yr[s] = np.copy(y[r])

        return xr, yr


class TestTree(object):

    def __init__(self, test_csv, tree):
        """ Constructor """

        data = np.genfromtxt(test_csv, delimiter=';')
        self.x = data[:,0:4].T
        self.y = data[:,4].astype(np.int)

        missed = 0.0
        if isinstance(tree, dict):
            self.yhat = self.test(tree)

            for i in range(len(self.y)):
                if self.y[i] != self.yhat[i]:
                    missed += 1
            self.accuracy = (1 - (missed/len(self.y)))*100
        else:
            yhat = np.zeros((len(tree), len(self.y)))

            for i in range(len(tree)):
                yhat[i] = self.test(tree[i])
            yhat_mv = self.majority_vote(yhat)
            for i in range(len(self.y)):
                if self.y[i] != yhat_mv[i]:
                    missed += 1
            self.accuracy = (1 - (missed/len(self.y)))*100

    def test(self, tree):
        """
        Classify each feature in the test set base on trained tree
        """

        x = self.x

        yhat = []
        for i in range(len(x[0])):
            yhat.append(self.classify_sample(x[:,i], tree))

        return np.array(yhat)

    def majority_vote(self, yhat):
        """
        Majority voting for all trees
        """

        yhat_mv = np.zeros(len(yhat[0]))
        for i in range(len(yhat[0])):
            u,c = np.unique(yhat[:,i], return_counts=True)
            yhat_mv[i] = u[int(np.argmax(c))]

        return yhat_mv



    def classify_sample(self, x, tree):
        """
        Classify specified sample
        """

        Q = []
        Q.append(tree)
        while Q:
            sub_tree = Q.pop(0)
            for k in sub_tree.keys():
                if eval(k):
                    if isinstance(sub_tree[k], np.ndarray):
                        return np.argmax(sub_tree[k])
                    else:
                        Q.append(sub_tree[k])


if __name__ == '__main__':

    # PART 1
    # I() for each feature and thresholda at root
    train = TrainTree('data/iris_train.csv', 0)
    P = train._partition(train.x, train.y)
    H = train._entropy(train.y)
    Pb, I, Itf = train._information_gain(H, P)
    pprint(Itf)

    # iterate k
    train_acc = []
    test_acc = []
    for i in range(0, 100):
        train = TrainTree('data/iris_train.csv', i)
        tree = train.train(train.x, train.y)
        test_accuracy = train.get_accuracy(tree)
        test = TestTree('data/iris_test.csv', tree)
        train_acc.append(test_accuracy)
        test_acc.append(test.accuracy)

    # plot training and testing error over k
    fig = plt.figure()
    plt.plot(train_acc, '-k', label='train')
    plt.plot(test_acc, '--k', label='test')
    plt.legend(loc='lower right')
    plt.ylim((50,105))
    plt.xlabel("$k$")
    plt.ylabel("Accuracy (%)")
    fig.savefig('iter_k.png')

    # PART 2
    L = [5,10,15,20,25,30]
    K = range(0,100)
    N = 10
    for l in L:
        train_acc_k = []
        test_acc_k = []
        for k in K:
            train_acc = []
            test_acc = []
            for n in range(N):
                rf = RandomForest('data/iris_train.csv', k, l)
                test = TestTree('data/iris_train.csv', rf.forest)
                train_acc.append(rf.accuracy)
                test_acc.append(test.accuracy)
            print np.mean(train_acc), np.mean(test_acc)
            train_acc_k.append(np.mean(train_acc))
            test_acc_k.append(np.mean(test_acc))

        # for each L, plot testing and training error over k
        fig = plt.figure()
        plt.plot(train_acc_k, '-k', label='train')
        plt.plot(test_acc_k, '--k', label='test')
        plt.legend(loc='lower right')
        plt.title("$L$={0}".format(l))
        plt.ylim((50,105))
        plt.xlabel("$k$")
        plt.ylabel("Accuracy (%)")
        fig.savefig('L_{0}.png'.format(l))

