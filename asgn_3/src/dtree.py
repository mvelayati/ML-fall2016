
"""
idtree.py
cs534: Implementation assignment 3
"""

import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt

class TrainTree(object):

    def __init__(self, data_csv, k):
        """ Constructor """

        data = np.genfromtxt(data_csv, delimiter=';')
        self.x = data[:,0:4].T
        self.y = data[:,4]

        # stopping criteria
        self.k = k

        # train tree
        self.tree = self.train(self.x, self.y)

        self.missed = 0.0
        self.get_missed(self.tree)
        self.accuracy = (1 - (self.missed/len(self.x[0])))*100

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

    def _partition(self, x, y):
        """
        Enumerate all 'meaningful' paritions
        """
        cnt = 0 # counter for partition dictionary
        P = {}

        # for each feature i, sort data based on i and find theata where
        # class label changes then split and add both partitions to P
        for i in range(len(x)):

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
            if Itmp > I:
                I = Itmp
                Pb = p

        return Pb, I

    def train(self, x, y):
        """
        Recursive method for top down induction
        """

        # dictionary to store tree
        tree = {}

        # if y only has 1 label then return it
        if len(set(y)) == 1:
            rv = [0,0,0]
            i = int(y[0])
            rv[i] = len(y)
            return rv

        # if number of instances in y is less than k then return y
        if len(y) <= self.k:
            rv = [0,0,0]
            u,c = np.unique(y, return_counts=True)
            for i in range(len(c)):
                rv[int(u[i])] = c[i]
            return rv

        # compute entropy
        H = self._entropy(y)

        # enumerate partitions
        P = self._partition(x,y)

        # get best partition based on I(Y,X)
        Pb,I = self._information_gain(H, P)

        # if there is no gain then return y
        if I <= 0:
            rv = [0,0,0]
            i = int(y[0])
            rv[i] = len(y)
            return rv

        # for both evaluations (T and F) recursively run train()
        for i in range(2):
            tree['x{0}<{1}'.format(P[Pb]['feature'], P[Pb]['theta'])] = self.train(P[Pb]['T']['x'], P[Pb]['T']['y'])
            tree['x{0}>{1}'.format(P[Pb]['feature'], P[Pb]['theta'])] = self.train(P[Pb]['F']['x'], P[Pb]['F']['y'])

        return tree

    def get_missed(self, tree):
        """
        Sum the total number of missclassifications
        """

        for v in tree.values():
            if isinstance(v, dict):
                self.get_missed(v)
            else:
                if np.count_nonzero(v) > 1:
                    self.missed += v[v!=0].min()

class TestTree(object):

    def __init__(self, train_csv, trained_tree):
        """ Constructor """

        data = np.genfromtxt(train_csv, delimiter=';')
        self.x = data[:,0:4].T
        self.y = data[:,4]

        self.missed = 0.0

        self.test(trained_tree)
        self.accuracy = (1 - (self.missed/len(self.y)))*100

    def _split(self, x, y, test, classify=True):
        """
        Splits data based on feature and threshold. If the trained
        tree is on a leaf node, calculate missclassifications
        """

        x = self.x
        y = self.y

        # parse the test string
        row = int(test[1])
        condition = test[2]
        theta = float(test[3:])

        # sort and split
        s = x[row].argsort()
        xs = x[:,s]
        ys = y[s]
        for i in xs[row]:
            if xs[row][i] >= theta:
                break
        xpL = xs[:,0:i]
        ypL = ys[0:i]
        xpG = xs[:,i:]
        ypG = ys[i:]

        # messy but it works
        if classify:
            if condition == '<':
                if ypL.any():
                    u,c = np.unique(ypL, return_counts=True)
                    self.missed += np.sum(c[c != np.max(c)])
            else:
                if ypG.any():
                    u,c = np.unique(ypG, return_counts=True)
                    self.missed += np.sum(c[c != np.max(c)])
        else:
            if condition == '<':
                return xpL, ypL
            else:
                return xpG, ypG


    def test(self, trained_tree):
        """
        BFS graph traversal of trained tree to classify test data
        """
        x = self.x
        y = self.y
        Q = [] # Stack
        Q.append(trained_tree)
        while Q:
            cur_dict = Q.pop(0)
            for k in cur_dict.keys():
                if isinstance(cur_dict[k], list):
                    self._split(x,y,k,classify=True)
                else:
                    x,y = self._split(x,y,k,classify=False)
                    Q.append(cur_dict[k])


if __name__ == '__main__':

    # PART 1

    # iterate k
    train_acc = []
    test_acc = []
    for i in range(0, 100):
        train = TrainTree('data/iris_train.csv', i)
        test = TestTree('data/iris_test.csv', train.tree)
        train_acc.append(train.accuracy)
        test_acc.append(test.accuracy)

    fig = plt.figure()
    plt.plot(train_acc, '-k', label='train')
    plt.plot(test_acc, '--k', label='test')
    plt.legend(loc='lower right')
    plt.ylim((0,110))
    plt.xlabel("$k$")
    plt.ylabel("Accuracy (%)")
    fig.savefig('iter_k.png')


