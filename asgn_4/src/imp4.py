"""
imp4.py

cs534 implementation assignment 4
"""

import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import stats


class Kmeans(object):

    def __init__(self, x, y, k, n):

        # instance variables
        self.x = x
        self.y = y
        self.k = k
        self.m = np.zeros(self.k)
        self.S = {}

        # =========
        # algorithm
        # =========
        # run n iterations and choose the best WSSE
        best_sse = np.inf
        best_xk = None
        for i in range(n):

            # random initial solution
            self.initialize()

            # mp is the previous set of means
            mp = np.zeros(self.k)

            # run until convergence (until mp == m)
            while np.array_equal(mp, self.m) == False:
                mp = np.copy(self.m)
                xk = self.assignment()
                rnt = self.update()
                # if no points are assigned to class k
                if rnt == 1:
                    print "Bad initialization: reinitializing..."
                    self.initialize()
            if self.f0() < best_sse:
                best_sse = self.f0()
                best_xk = xk

        # purity
        self.accuracy = self.purity(best_xk)
        print "\tBest WSSE: ", best_sse
        print "\tAccuracy: ", self.accuracy*100

    def _distance(self, x, y):
        """
        distance between vectors
        """
        return np.linalg.norm(x-y)**2

    def initialize(self):
        """
        initial random guess
        """
        self.m = np.zeros((k, len(self.x[0])))
        rand_idx = []
        for i in range(self.k):
            r = random.randint(0, len(self.x)-1)
            while r in rand_idx:
                r = random.randint(0, len(self.x)-1)
            self.m[i] = self.x[r]

        return 0

    def assignment(self):
        """
        reassign vectors
        """
        # vector xk is aligned with x for class labels
        xk = np.zeros(len(self.x))

        for i in range(self.k):
            self.S[i] = []
        for p in range(len(self.x)):
            min_dist = np.inf
            asgn = 0
            for i in range(len(self.m)):
                if np.all(np.isnan(self.m[i])):
                    return 1
                dist = self._distance(self.x[p], self.m[i])
                if dist < min_dist:
                    min_dist = dist
                    asgn = i
            xk[p] = asgn
            self.S[asgn].append(self.x[p])

        return xk

    def update(self):
        """
        recompute the centroids
        """
        for i in range(self.k):
            summer = np.zeros(len(self.x[0]))
            if len(self.S[i]) == 0:
                return 1
            for j in self.S[i]:
                summer += j
            self.m[i] = summer/len(self.S[i])

        return 0

    def f0(self):
        """
        k-means objective function
        """
        k_sum = 0
        for i in range(self.k):
            S_sum = 0
            mu = np.mean(self.S[i])
            for x in self.S[i]:
                S_sum += np.linalg.norm(x - mu)**2
            k_sum += S_sum

        return k_sum

    def purity(self, xk):
        """
        compute purity measure
        """
        classify = np.zeros(self.k)
        for i in range(self.k):
            classify[i] = stats.mode(self.y[xk==i])[0]

        p = np.zeros(len(self.x))
        for i in range(self.k):
            p[xk==i] = classify[i]

        cnt = 0
        for i in range(len(self.x)):
            if self.y[i] != p[i]:
                cnt += 1

        return 1 - (cnt/float(len(p)))

class PCA(object):

    def __init__(self, x):

        self.x = x

        # 1: compute mean vector
        mu = np.mean(x, axis=0)
        # 2: compute covariance matrix
        cov_mat = (x - mu).T.dot(x - mu)/(len(x)-1)
        # 3: Eigen decomposition
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)
        self.eig_vals = eig_vals
        self.eig_vecs = eig_vecs
        # 4: Sort eigen pairs
        self.eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
        self.eig_pairs.sort(key=lambda x: x[0], reverse=True)

    def reproject(self, d):

        project = np.hstack((self.eig_pairs[i][1].reshape(len(self.x[0]), 1) for i in range(d)))
        return np.real(self.x.dot(project))

    def var_explained(self, var):

        sum_eig = np.sum(self.eig_vals)
        var_exp = []
        cnt = 0
        for i in sorted(self.eig_vals, reverse=True):
            var_exp.append((i / sum_eig)*100)
            cnt += 1
            if np.sum(var_exp) >= var:
                return cnt

class LDA(object):

    def __init__(self, x, y):

        self.x = x
        self.y = y

        self.compute_means()


    def compute_means(self):

        labs = np.unique(self.y)
        mu = np.zeros((len(labs), len(self.x[0])))
        for i in range(len(labs)):
            mu[i] = np.mean(self.x[y==labs[i]], axis=0)

        self.m = mu

    def project(self):

        labs = np.unique(self.y)
        s1 = (self.x[y==labs[0]] - self.m[0]).T.dot(self.x[y==labs[0]] - self.m[0])
        s2 = (self.x[y==labs[1]] - self.m[1]).T.dot(self.x[y==labs[1]] - self.m[1])
        S = s1 + s2
        w = np.linalg.inv(S).dot(self.m[0] - self.m[1])
        return self.x.dot(w).reshape(len(self.x), 1)



if __name__ == '__main__':

    x = np.genfromtxt('data/walking.train.data', delimiter=' ')
    y = np.genfromtxt('data/walking.train.labels')

    lda = LDA(x, y)
    xd = lda.project()

    pca = PCA(x)
    print "\nComponents for 80%: ", pca.var_explained(80)
    print "Components for 90%: ", pca.var_explained(90)
    xp = pca.reproject(1)

    k = 2
    print "\nk-means raw"
    km = Kmeans(x,y,k,10)
    print "k-means pca"
    km = Kmeans(xp,y,k,10)
    print "k-means LDA"
    km = Kmeans(xd,y,k,10)

