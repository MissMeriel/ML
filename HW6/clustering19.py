#!/usr/bin/python

import sys, csv, random, math
import numpy as np
import matplotlib.pyplot as plt
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
    #print("initialCentroids", initialCentroids)
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
            #print("allocatePoints i", i)
            cluster_dists.append(distance)
        #print("cluster_dists", cluster_dists)
        #print("max(cluster_dists)", max(cluster_dists))
        a = cluster_dists.index(min(cluster_dists))
        #print("a", a)
        allocation.append(a)
    print("instances of 0 in allocation", allocation.count(0))
    print("instances of 1 in allocation", allocation.count(1))
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
    print("initial centroids", clusters)
    allocation = allocatePoints(X, clusters)
    clusters = updateCentroids(clusters, X, allocation)
    print("updated centroids", clusters)
    #Your code here
    clusters_old = None
    i = 0
    while(clusters != clusters_old and i < maxIter):
        clusters_old = clusters.copy()
        allocation = allocatePoints(X, clusters)
        clusters = updateCentroids(clusters, X, allocation)
        print("clusters_old", clusters_old)
        print("clusters", clusters)
        i += 1
    print("iterations to find optimality:", i)
    print("optimal clusters:", clusters)
    return clusters

### Find lowest optimal k
def kneeFinding(X, kList):
    #Your code here
    #for k in kList:
    k = 0
    # while distance between clusters is reasonable
    #while :
    #    clusters = kmeans(X, kList[k])
    #    k += 1
    return

### Find accuracy per cluster
def purity(X, clusters):
    purities = [0] * len(clusters)
    #Your code here
    allocation = allocatePoints(X, clusters)
    for c in range(len(clusters)):
        for i in range(len(X)):
            a = allocation[i]
            a_true = X[i][-1]
            # align cluster count with X cluster name
            if(a == a_true and a == c):
                purities[c] += 1
        pop = allocation.count(c)
        purities[c] = purities[c] / float(pop)
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
    return initialClusters


def calcLogLikelihood(X, clusters, k):
    loglikelihood = 0
    #Your code here
    return loglikelihood

#E-step (expectation)
def updateEStep(X, clusters, k):
    EMatrix = []
    #Your code here
    return EMatrix

#M-step (maximization)
#19c slide 46
def updateMStep(X, clusters, EMatrix):
    #Your code here

    return clusters

def visualizeClustersGMM(X, labels, clusters, covType):
    #Your code here
    return

def gmmCluster(X, k, covType, maxIter=1000):
    #initial clusters
    clustersGMM = getInitialsGMM(X,k,covType)
    labels = []
    #Your code here

    visualizeClustersGMM(X, labels, clustersGMM, covType)
    return labels, clustersGMM

def purityGMM(X, clusters, labels):
    purities = []
    #Your code here
    return purities


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
    exit()

    #Q4
    kneeFinding(dataset1,range(1,7))

    #Q5
    purity(dataset1, clusters)


    #Q7
    labels11,clustersGMM11 = gmmCluster(dataset1, 2, 'diag')
    labels12,clustersGMM12 = gmmCluster(dataset1, 2, 'full')

    #Q8
    labels21,clustersGMM21 = gmmCluster(dataset2, 2, 'diag')
    labels22,clustersGMM22 = gmmCluster(dataset2, 2, 'full')

    #Q9
    purities11 = purityGMM(dataset1, clustersGMM11, labels11)
    purities12 = purityGMM(dataset1, clustersGMM12, labels12)
    purities21 = purityGMM(dataset2, clustersGMM21, labels21)
    purities22 = purityGMM(dataset2, clustersGMM22, labels22)

if __name__ == "__main__":
    main()