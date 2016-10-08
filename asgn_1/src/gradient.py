# module imports
import random
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

# load training data
train = np.genfromtxt('train.csv', delimiter=',')
# and the test data
test = np.genfromtxt('test.csv', delimiter=',')

# extract (x,y)
x = train[:, :-1]
y = train[:, -1]

xp = test[:, :-1]
yp = test[:, -1]

# normalize the features and set unit variance
x = (x - np.mean(x)) / np.std(x)
xp = (xp - np.mean(xp)) / np.std(xp)

# set bias term to 1
x[:, 0] = 1
xp[:, 0] = 1


# ==========================
# Gradient Descent function
# ==========================

def gradient_descent(x, y, w, p, l_rate, e):
    cost = []
    reg_term = []
    m = y.size
    norm_gradient = []
    norm_gradient_dummy = []
    gradient = np.inf
    norm_gradient.append(np.linalg.norm(gradient))

    cnt = 0

    # repeat until convergence
    while (np.linalg.norm(gradient) > e):

        # hypothesis
        h = x.dot(w)

        # error
        error = h - y

        # compute gradient and regularization
        dummy = x.T.dot(error)
        gradient = (dummy) + (2 * p * w)
        norm_gradient.append(np.linalg.norm(gradient))
        norm_gradient_dummy.append(np.linalg.norm(dummy))

        # update parameters
        # w = np.subtract(w, l_rate * gradient)
        w = w - l_rate * gradient

        # store objective
        if cnt % 100 == 0:
            # cost.append(np.sum(error**2)/(2*m))
            reg_term.append(np.linalg.norm(w))
            cost.append(np.sum(error ** 2) / 2)
            print("{} , {} , {} , {}".format(cost[-1], norm_gradient[-1], l_rate, p))

        cnt += 1

    return w, cost


# =====================
# 10-fold sets creator
# =====================

def create_tenfold_sets(x):
    dataset_indices = range(100)
    ten_fold_indices = []
    while len(dataset_indices) > 0:
        sample = random.sample(dataset_indices, 10)
        ten_fold_indices.append(sample)
        dataset_indices = list(set(dataset_indices).difference(sample))

    ten_fold_datasets = []
    for i in range(len(ten_fold_indices)):
        ten_fold_data = []
        for index in ten_fold_indices[i]:
            ten_fold_row = x[index, :]
            ten_fold_data.append(ten_fold_row)

        ten_fold_datasets.append(ten_fold_data)

    return ten_fold_datasets

# =========================================
# Part 1: Explore different learning rates
# =========================================

def part1():

    # hyper-parameters
    p = 0.85
    # lr = 0.001
    e = 0.001
    sp = .1

    lr = np.arange(0.0006, 0.00063, 0.00001)
    sp_loss = []
    test_loss = []
    norm_grad = []

    for l_rate in lr:
        w = np.zeros(len(x[0]))
        w, cost = gradient_descent(x, y, w, sp, l_rate, e)
        norm_grad.append(np.linalg.norm(w))
        sp_loss.append(cost[-1])
        h = xp.dot(w)
        error = h - yp
        test_loss.append((np.sum(error ** 2) / 2))

    # plot learning rate test
    # mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(5, 4))
    plt.plot(lr, sp_loss, '-k')
    plt.plot(lr, test_loss, '--k')
    # plt.xscale('log')
    # plt.xlabel(r'$\rm{log}_{10}(\learning rate$)', fontsize=10)
    # plt.ylabel(r'$\rm{J}(w)$', fontsize=10)
    # plt.legend(loc='upper right', fontsize=10)
    # fig.savefig("test_learning_rate.png")
    plt.show()


# =======================================
# Part 2: Explore different lambda values
# =======================================

def part2():

    # hyper-parameters
    p = 0.85
    lr = 0.0006
    e = 0.001

    # test effect of lambda on traning and test SSE
    sp = np.arange(0.01, 10, 0.01)
    sp = np.arange(0.1,10,0.1)
    sp_loss = []
    test_loss = []
    norm_grad = []
    for i in sp:
        w = np.zeros(len(x[0]))
        w, cost = gradient_descent(x, y, w, i, lr, e)
        norm_grad.append(np.linalg.norm(w))
        sp_loss.append(cost[-1])
        h = xp.dot(w)
        error = h - yp
        test_loss.append((np.sum(error ** 2) / 2))

        # print i

    # print sp_loss
    # print test_loss

    # plot lambda test
    # mpl.rc('text', usetex = True)
    fig = plt.figure(figsize=(5, 4))
    mpl.rc('text', usetex = True)
    mpl.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}", r"\renewcommand{\vec}[1]{\mathbf{#1}}"]
    fig = plt.figure(figsize=(5,4))
    plt.plot(sp, sp_loss, '-k', label=r'$\rm{train}$')
    plt.plot(sp, test_loss, '--k', label=r'$\rm{test}$')
    plt.xscale('log')
    plt.xlabel(r'$\rm{log}_{10}(\lambda_i$)', fontsize=10)
    plt.ylabel(r'$\rm{J}(\vec{w}^*)$', fontsize=10)
    plt.legend(loc='upper right', fontsize=10)
    fig.savefig("test_lambda.png", dpi=200)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(sp, norm_grad, '-k')
    plt.xscale('log')
    plt.xlabel(r'$\rm{log}_{10}(\lambda_i)$', fontsize=10)
    plt.ylabel(r'$||\vec{w}^*||$', fontsize=10)
    fig.savefig('norm_lambda.png', dpi=200)


# =================================
# Part 3: 10-fold cross-validation
# =================================

def part3():

    # hyper-parameters
    p = 0.85
    lr = 0.0006
    e = 0.001

    # test effect of lambda on traning and test SSE
    sp = np.arange(0.01, 0.05, 0.01)
    sp_loss = []
    sse_lambda_array = []
    norm_grad = []


    def ten_fold_cross_validation(dataset_x, dataset_y):
        w = np.zeros(len(x[0]))
        dataset = np.c_[dataset_x, dataset_y]
        dataset_temp = np.zeros((10, len(dataset[0])))
        for lambda_cof in sp:
            test_loss = 0
            ten_fold_dataset = create_tenfold_sets(dataset)
            ten_fold_indices = range(len(ten_fold_dataset))
            for ten_fold_index in ten_fold_indices:
                for index in [x for x in ten_fold_indices if x != ten_fold_index]:
                    dummy = np.vstack((dataset_temp, np.array(ten_fold_dataset[index])))
                    dataset_temp = dummy
                dataset_temp = np.split(dataset_temp, 10)[2]
                w, cost = gradient_descent(dataset_temp[:, :-1], dataset_temp[:, -1], w, lambda_cof, lr, e)
                h = np.array(ten_fold_dataset[ten_fold_index])[:,:-1].dot(w)
                error = h - np.array(ten_fold_dataset[ten_fold_index])[:, -1]
                test_loss += (np.sum(error ** 2) / 2)
                dataset_temp = np.zeros((10, len(dataset[0])))
            sse_lambda_array.append(test_loss)
        return sse_lambda_array

    sse_lambda_array = ten_fold_cross_validation(x, y)

    fig = plt.figure(figsize=(5, 4))
    plt.plot(sp, sse_lambda_array, '-k')
    plt.xscale('log')
    #plt.xlabel(r'$\rm{log}_{10}(\lambda)$', fontsize=10)
    #plt.ylabel(r'$||\nabla \rm{J}(w)||$', fontsize=10)
    fig.savefig('3.png', dpi=200)


def main():
    #part1()
    #part2()
    part3()

main()