

import numpy as np
from pprint import pprint

# data
x = np.array([[1,3,2,4,5,7],[2,4,6,5,7,9],[7,5,6,8,9,10],[12,14,13,15,18,17]])

y = np.array([1,1,2,2,3,3])

# import iris data
data = np.genfromtxt("data/iris_train.csv", delimiter=';')
x = data[:,0:3].T
print x
y = data[:,4]

def entropy(y):
    N = float(len(y))
    H = 0.0
    for i in set(y):
        pi = np.count_nonzero(y==i)/N
        # will need to check for zeros before taking logarithm
        H -= pi*np.log2(pi)

    return H

H = entropy(y)

def partition(x,y):
    cnt = 0
    P = {}
    for i in range(len(x)):
        s = x[i].argsort()
        sx = x[:,s]
        sy = y[s]
        cur_lab = sy[0]
        for j in range(1, len(sy)):
            if sy[j] != cur_lab:
                cnt += 1
                cur_lab = np.copy(sy[j])
                theta = (sx[i,j-1] + sx[i,j])/2.0
                P[cnt] = {'feature': i,
                          'theata': theta,
                          'T': {'x': sx[:,0:j], 'y': sy[0:j]},
                          'F': {'x': sx[:,j:], 'y': sy[j:]}}

    return P

P = partition(x,y)


def info_gain(H, P):
    feature, theata = 0,0
    I = 0.0
    Pb = 0
    for p in P.keys():
        Ht = entropy(P[p]['T']['y'])
        Hf = entropy(P[p]['F']['y'])
        N = float(len(P[p]['T']['y']) + len(P[p]['F']['y']))
        p1 = len(P[p]['T']['y'])/N
        p2 = len(P[p]['F']['y'])/N
        Itmp = H - (p1*Ht + p2*Hf)
        if Itmp > I:
            I = Itmp
            Pb = p
    return Pb, I


print info_gain(H, P)

def ID3(x, y):
    if len(set(y)) == 1:
        return y
    H = entropy(y)
    P = partition(x,y)
    Pb,I = info_gain(H, P)
    if I <= 0:
        return y
    tree = {}
    for i in range(2):
        tree['{0} < {1}'.format(P[Pb]['feature'], P[Pb]['theata'])] = ID3(P[Pb]['T']['x'], P[Pb]['T']['y'])
        tree['{0} > {1}'.format(P[Pb]['feature'], P[Pb]['theata'])] = ID3(P[Pb]['F']['x'], P[Pb]['F']['y'])
    return tree

pprint(ID3(x,y))

