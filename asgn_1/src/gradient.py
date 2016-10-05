
# module imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# load training data
train = np.genfromtxt('train.csv', delimiter=',')
# and the test data
test = np.genfromtxt('test.csv', delimiter=',')

# extract (x,y)
x = train[:,:-1]
y = train[:,-1]

xp = test[:,:-1]
yp = test[:,-1]

# normalize the features and set unit variance
x = (x-np.mean(x))/np.std(x)
xp = (xp-np.mean(xp))/np.std(xp)

# set bias term to 1
x[:,0] = 1
xp[:,0] = 1

# initialize parameters with zero vector of size m
w = np.zeros(len(x[0]))

def gradient_descent(x, y, w, p, lr, e):

    cost = []
    m = y.size
    gradient = np.inf

    cnt = 0

    # repeat until convergence
    while (np.linalg.norm(gradient) > e):

        # hypothesis
        h = x.dot(w)

        # error
        error = h - y

        # compute gradient and regularization
        gradient = (x.T.dot(error)/m) + (2*p*w)

        # update parameters
        w = w - lr * gradient

        # store objective
        if cnt % 100 == 0:
            cost.append(np.sum(error**2)/(2*m))
            #print cost[-1]

        cnt += 1

    return w, cost

#=======================================
# Part 2: Explore differnt lambda values
#=======================================

# hyper-parameters
p = 0.85
lr = 0.001
e = 0.001

# test effect of lambda on traning and test SSE
sp = np.arange(0.1,10,0.1)
sp_loss = []
test_loss = []
for i in sp:
    w = np.zeros(len(x[0]))
    w, cost = gradient_descent(x,y,w,i,lr,e)
    sp_loss.append(cost[-1])
    h = xp.dot(w)
    error = h-yp
    test_loss.append(np.sum(error**2)/(2*yp.size))

    print i

print sp_loss
print test_loss

# plot lambda test
mpl.rc('text', usetex = True)
fig = plt.figure()
plt.plot(sp, sp_loss, '-k', label='training set')
plt.plot(sp, test_loss, '--k', label='test set')
plt.xscale('log')
plt.xlabel(r'$\rm{log}_{10}(\lambda$)')
plt.ylabel(r'$\rm{J}(w)$')
plt.legend(loc='upper right')
fig.savefig("test_lambda.png")

