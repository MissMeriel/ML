#!/usr/bin/python

import sys, csv, random, math
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import statistics
#Your code here

def loadData(fileDj):
    data = []
    f = open(fileDj, 'r')
    dataframe = csv.reader(f, delimiter=' ')
    labels, images = [], []
    data = []
    for line in dataframe:
        sample = [float(line[i]) for i in range(len(line)-1)]
        labels.append(sample)
        images.append(int(line[-1]))
        data.append([float(line[i]) for i in range(len(line))])
    #data = np.array(data).astype('float32')
    return data

## K-means functions 
### initialize centroids randomly
def getInitialCentroids(X, k):
    initialCentroids = {}
    #Your code here
    initialCentroids = []
    ### Initialize a random point for each cluster
    bounds = []
    for j in range(len(X[0]) - 1):
        minimum = min([r[j] for r in X])
        maximum = max([r[j] for r in X])
        bounds.append([minimum, maximum])
    for i in range(k):
        random_point = []
        for j in range(len(X[0])-1):
            random_point.append(random.uniform(bounds[j][0], bounds[j][1]))
        initialCentroids.append(random_point)
    return initialCentroids

### distance metric is ______
def getDistance(pt1,pt2):
    dist = 0
    #Your code here
    dist = math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)
    return dist

### Decide class memberships of all samples in X
### by assigning them to the nearest cluster centroids
def allocatePoints(X,clusters):
    #Your code here
    allocation = []
    # for each sample
    for i in range(len(X)):
        # get distance from each centroid
        cluster_dists = []
        for c in range(len(clusters)):
            distance = getDistance(X[i], clusters[c])
            cluster_dists.append(distance)
        a = cluster_dists.index(min(cluster_dists))
        allocation.append(a)
    return allocation

### Re-estimate the k clusters, assuming memberships are correct.
### If no samples changed cluster membership, you're done.
def updateCentroids(clusters, X, allocation):
    #Your code here
    for c in range(len(clusters)):
        new_centroid = [0, 0]
        for i in range(len(X)):
            if(allocation[i] == c):
                new_centroid[0] += (X[i][0] / float(allocation.count(c)))
                new_centroid[1] += (X[i][1] / float(allocation.count(c)))
        #clusters[c] = [new_centroid[i] / float(len(X)) for i in len(new_centroid)]
        clusters[c] = new_centroid
    return clusters

def visualizeClusters(X, clusters):
    #Your code here
    allocation = allocatePoints(X, clusters)
    groups = [[] for i in range(len(clusters))]
    for i in range(len(X)):
        a = allocation[i]
        groups[a].append(X[i])
    for c in range(len(clusters)):
        group = groups[c]
        plt.scatter([group[i][0] for i in range(len(group))], [group[i][1] for i in range(len(group))])
        #plt.scatter()
    plt.show()
    return

def kmeans(X, k, maxIter=1000):
    clusters = getInitialCentroids(X, k)
    allocation = allocatePoints(X, clusters)
    clusters = updateCentroids(clusters, X, allocation)
    #Your code here
    clusters_old = None
    i = 0
    while(clusters != clusters_old and i < maxIter):
        clusters_old = clusters.copy()
        allocation = allocatePoints(X, clusters)
        clusters = updateCentroids(clusters, X, allocation)
        #print("clusters_old", clusters_old)
        #print("clusters", clusters)
        i += 1
    #print("iterations to find optimality:", i)
    #print("optimal clusters:", clusters)
    return clusters

### Find lowest optimal k
def kneeFinding(X, kList):
    threshold = 0.03
    avg = 0.0
    #Your code here
    i = 0
    k_opt = kList[i]
    ks = []
    for k in kList:
        clusters = kmeans(X, k)
        p = purity(X, clusters)
        avg_new = statistics.mean(p)
        #print("k", k, "avg", avg_new)
        ks.append(avg_new)
        if(abs(avg - avg_new) < threshold ):
            k_opt = kList[i-1]
            print("optimal k:", k_opt)
            return k_opt
        avg = avg_new
        i += 1
    plt.plot(ks)
    plt.show()
    return k_opt

### Find accuracy per cluster
def purity(X, clusters, labels=None):
    purities = [0] * len(clusters)
    #Your code here
    if(labels == None):
        labels = allocatePoints(X, clusters)
    for c in range(len(clusters)):
        counter = []
        for i in range(len(X)):
            a = labels[i]
            a_true = X[i][-1]
            # align cluster count with X cluster name
            if (a == c):
                counter.append(a_true)
        cntr = Counter(counter)
        try:
            purities[c] = max(cntr.values()) / sum(cntr.values())
        except:
            purities[c] = 0
    return purities



## GMM functions 

#calculate the initial covariance matrix
#covType: diag, full
def getInitialsGMM(X, k, covType):
    if covType == 'full':
        dataArray = np.transpose(np.array([pt[0:-1] for pt in X]))
        covMat = np.cov(dataArray)
    else:
        covMatList = []
        for i in range(len(X[0])-1):
            data = [pt[i] for pt in X]
            cov = np.asscalar(np.cov(data))
            covMatList.append(cov)
        covMat = np.diag(covMatList)
    #Your code here
    initialClusters = getInitialCentroids(X, k)
    return initialClusters, covMat

def calc_denominator(x, point_index, clusters, cluster_index, k, EMatrix, covMatList):
    denominator = 0
    for j in range(k):
        c = clusters[j]
        p = x[0:len(x)-1]
        try:
            np.transpose(p - c)
            n = np.matmul(np.transpose(p - c), np.linalg.inv(covMatList[j]))
        except ValueError as e:
            print(e)
            print("x", x)
            print("p", p)
            print("c", c)
            print("covMatList[", j, "]", covMatList[j])
            exit()
        n = np.matmul(n, (p - c))
        d = (math.sqrt(2*math.pi*np.linalg.det(covMatList[j])))
        denominator += ((np.exp(-0.5 * n)/d) * EMatrix[point_index][j])
    return denominator

def calcLogLikelihood(point, point_index, clusters, cluster_index, k, EMatrix, covMatList):
    loglikelihood = 0
    #Your code here
    c = clusters[cluster_index]
    denominator = calc_denominator(point, point_index, clusters, cluster_index, k, EMatrix, covMatList)
    p = point[0:len(point) - 1]
    s1 = np.matmul(np.transpose(p - c), np.linalg.inv(covMatList[cluster_index]))
    s2 = np.matmul(s1, (p - c))
    numerator = np.exp((-0.5 * s2) )#/ (math.sqrt(math.pow(2 * math.pi, len(point)-1) * np.linalg.det(covMatList[cluster_index]))))
    numerator = numerator * EMatrix[point_index][cluster_index]
    loglikelihood = numerator / denominator
    return loglikelihood

# return softmax of list
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return list(e_x / e_x.sum())

#E-step (expectation)
# See hw6 sheet
# clusters contains k centroids
def updateEStep(X, clusters, k, EMatrix, covMatList):
    EMatrix_new = []
    #Your code here
    X = np.array(X)
    i = 0
    for point in X:
        c_probs = []
        j = 0
        for c in clusters:
            # get probability this point belongs to this cluster
            ll = calcLogLikelihood(point, i, clusters, j, k, EMatrix, covMatList)
            c_probs.append(ll)
            j += 1
        i += 1
        c_probs = softmax(c_probs)
        EMatrix_new.append(c_probs)
    return EMatrix_new


#M-step (maximization)
#19c slide 46
def updateMStep(X, clusters, EMatrix, covMatList):
    #Your code here
    # for each cluster
    for j in range(len(clusters)):
        temp = [EMatrix[i][j] for i in range(len(X))]
        temp2 = [0] * (len(X[0])-1)
        for i in range(len(X)):
            p = []
            for f in range(len(X[i])):
                p.append(X[i][f] * EMatrix[i][j])
            temp2 = [temp2[m] + p[m] for m in range(len(X[0])-1)]
        clusters[j] = [temp2[i] / sum(temp) for i in range(len(temp2))]
    return clusters


def gmmCluster(X, k, covType, maxIter=1000):
    #initial clusters
    clustersGMM, covMat = getInitialsGMM(X, k, covType)
    labels = []
    #Your code here
    i = 0
    covMatList = [covMat.copy()] * k
    # EMatrix with uniform probabilities
    EMatrix = [[1.0/k] * k] * len(X)
    EMatrix_old = None
    EMatrix_init = EMatrix.copy()
    #while(EMatrix != EMatrix_old and i < maxIter):
    while (i < maxIter):
        EMatrix_old = EMatrix.copy()
        #print(EMatrix_old)
        EMatrix = updateEStep(X, clustersGMM, k, EMatrix, covMatList)
        #print(EMatrix)
        clusters = updateMStep(X, clustersGMM, EMatrix, covMatList)
        i += 1
    #print("i", i)
    #print("EMatrix_init==EMatrix:", EMatrix_init==EMatrix)
    labels = []
    for i in range(len(X)):
        max_exp_cluster = EMatrix[i].index(max(EMatrix[i]))
        labels.append(max_exp_cluster)
    visualizeClustersGMM(X, labels, clustersGMM, covType)
    return labels, clustersGMM


def purityGMM(X, clusters, labels):
    purities = []
    #Your code here
    purities = purity(X, clusters, labels)
    return purities

def visualizeClustersGMM(X, labels, clusters, covType):
    #Your code here
    groups = [[] for i in range(len(clusters))]
    for i in range(len(X)):
        a = labels[i]
        groups[a].append(X[i])
    for c in range(len(clusters)):
        group = groups[c]
        plt.scatter([group[i][0] for i in range(len(group))], [group[i][1] for i in range(len(group))])
    plt.show()
    return

def main():
    #######dataset path
    datadir = sys.argv[1]
    pathDataset1 = datadir+'/humanData.txt'
    pathDataset2 = datadir+'/audioData.txt'
    dataset1 = loadData(pathDataset1)
    dataset2 = loadData(pathDataset2)
    #Q2,Q3
    # https://qiyanjun.github.io/2019f-UVA-CS6316-MachineLearning//Lectures/L19b-clustering2-kMeans.pdf
    # slide 12
    #lecture 19 slide 50
    clusters = kmeans(dataset1, 2, maxIter=1000)

    visualizeClusters(dataset1, clusters)

    #Q4
    kneeFinding(dataset1, range(1,7))

    #Q5
    p = purity(dataset1, clusters)
    print("kmeans purity for dataset1:", p)


    #Q7
    labels11, clustersGMM11 = gmmCluster(dataset1, 2, 'diag')
    labels12, clustersGMM12 = gmmCluster(dataset1, 2, 'full')

    #Q8
    labels21, clustersGMM21 = gmmCluster(dataset2, 2, 'diag')
    labels22, clustersGMM22 = gmmCluster(dataset2, 2, 'full')

    #Q9
    purities11 = purityGMM(dataset1, clustersGMM11, labels11)
    print("purities for dataset 1, diag:", purities11)
    purities12 = purityGMM(dataset1, clustersGMM12, labels12)
    print("purities for dataset 1, full:", purities11)
    purities21 = purityGMM(dataset2, clustersGMM21, labels21)
    print("purities for dataset 2, diag:", purities11)
    purities22 = purityGMM(dataset2, clustersGMM22, labels22)
    print("purities for dataset 2, full:", purities11)

if __name__ == "__main__":
    main()