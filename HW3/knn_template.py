# Starting code for UVA CS 4501 Machine Learning- KNN

__author__ = '**'
import numpy as np
np.random.seed(37)
# for plot
import matplotlib.pyplot as plt
#more imports
import math
import random
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
## the only purpose of the above import is in case that you want to compare your knn with sklearn knn

# Load file into np arrays
# x is the features
# y is the labels
def read_file(file):
    data = np.loadtxt(file, skiprows=1)
    np.random.shuffle(data)
    x = data[:, :-1]
    y = data[:, -1].astype(int)
    return x, y

# 2. Generate the i-th fold of k fold validation
# Input:
# x is an np array for training data
# y is an np array for labels
# i is an int indicating current fold
# nfolds is the total number of cross validation folds
# Note: k = number of bins, iteratively pick one bin as your test set
def fold(x, y, i, nfolds):
    # your code
    split_x = np.array(np.split(x, nfolds))
    split_y = np.array(np.split(y, nfolds))
    x_test = split_x[i]
    y_test = split_y[i]
    x_train = np.concatenate(np.delete(np.copy(split_x), i, 0))
    y_train = np.concatenate(np.delete(np.copy(split_y), i, 0))
    return x_train, y_train, x_test, y_test


# 3. Classify each testing points based on the training points
# Input
# x_train: a numpy array of training data 
# x_test: a numpy array
# k: the number of neighbors to take into account when predicting the label
# Output
# y_predict: a numpy array 
def classify(x_train, y_train, x_test, k):
    # your code
    # Euclidean distance as the measurement of distance in KNN
    y_predict = np.zeros(x_test.shape[0])
    delta_sq = pow(1, 2)
    for h in range(x_test.shape[0]):
        distances = np.zeros(x_train.shape[0])
        for i in range(x_train.shape[0]):
            #print("x_train["+str(i)+"] - x_test["+str(i)+"]="+str(x_train[i] - x_test[i]))
            sum = np.sum((x_train[i] - x_test[h]) * (x_train[i] - x_test[h]))
            #print("np.sum(x_train["+str(i)+"] - x_test["+str(i)+"])="+str(sum))
            dist = pow(sum, 2)
            distances[i] = math.sqrt(dist)
            # print(distances[i])
            # get indices of k smallest distances
            # print("distances.argsort(): "+str(distances.argsort()))
        min_dist_indices = distances.argsort()[:k]
        # get predominant classification among k nearest neighbors
        # break ties with 1+(1/2*k)
        classes = np.zeros(len(min_dist_indices))
        m=0
        for index in min_dist_indices:
            # classes[np.where(min_dist_indices == index)] = y_train[index]
            #print("distances["+str(index)+"]: "+str(distances[index]))
            classes[m] = y_train[index]
            m += 1
        # print(classes)
        mode = stats.mode(classes)
        y_predict[h] = mode.mode[0]
    return y_predict

# 4. Calculate accuracy by comparing with true labels
# Input
# y_predict is a numpy array of 1s and 0s for the class prediction
# y is a numpy array of 1s and 0s for the true class label
def calc_accuracy(y_predict, y):
    # your code
    acc = 0
    for i in range(len(y_predict)):
        acc += 1 - abs(y[i] - y_predict[i])
    # print("acc: "+str(acc))
    return acc / float(len(y_predict))

# 5. Draw the bar plot of k vs. accuracy
# klist: a list of values of ks
# accuracy_list: a list of accuracies
def barplot(klist, accuracy_list):
    # your code
    # use matplot lib to generate bar plot with K on x axis and cross validation accuracy on y-axis
    plt.bar(klist, accuracy_list, align='center', alpha=0.5)
    plt.xlabel('K values')
    plt.ylabel('cross validation accuracy')
    plt.title('Cross validation accuracy by K')
    plt.show()
    return

# 1. Find the best K
def findBestK(x, y, klist, nfolds):
    kbest = 0
    best_acc = 0
    accuracy_list = []
    for k in klist:
        # your code here
        # to get nfolds cross validation accuracy for k neighbors
        # implement fold(x, y, i, nfolds),classify(x_train, y_train, x_test, k) and calc_accuracy(y_predict, y)
        accuracy = 0
        for i in range(nfolds):
            x_train, y_train, x_test, y_test = fold(x, y, i, nfolds)
            y_predict = classify(x_train, y_train, x_test, k)
            accuracy += calc_accuracy(y_predict, y_test)  # CROSS VALIDATION accuracy for k neighbors
        accuracy = accuracy / float(nfolds)
        if accuracy > best_acc:
            kbest = k
            best_acc = accuracy
        accuracy_list.append(accuracy)
        print(k, accuracy)
        # test against sklearn
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(x_train, y_train)
        acc_check = neigh.score(x_test, y_test)
        print(k, acc_check)
    # plot cross validation error for each k : implement function barplot(klist, accuracy_list)
    barplot(klist, accuracy_list)
    return kbest


if __name__ == "__main__":
    filename = "Movie_Review_Data.txt"
    # read data
    x, y = read_file(filename)
    x_ver = np.zeros((2, len(y)))
    y_ver = np.zeros(len(y))
    nfolds = 4
    klist = [3, 5, 7, 9, 11, 13]
    # Implementation covers two tasks, both part of findBestK function
    # Task 1 : implement kNN classifier for a given x,y,k 
    # Task 2 : implement 4 fold cross validation to select best k from klist
     
    bestk = findBestK(x, y, klist, nfolds)
    # report best k, and accuracy, discuss why some k work better than others
