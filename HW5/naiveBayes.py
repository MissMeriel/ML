#!/usr/bin/python

import sys
import os
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from os import listdir
from os.path import isfile, join
import re, math

###############################################################################

stop_words = [
    "a", "about", "above", "across", "after", "afterwards",
    "again", "all", "almost", "alone", "along", "already", "also",
    "although", "always", "am", "among", "amongst", "amoungst", "amount", "an",
    "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere",
    "are", "as", "at", "be", "became", "because", "become","becomes", "becoming",
    "been", "before", "behind", "being", "beside", "besides", "between", "beyond",
    "both", "but", "by","can", "cannot", "cant", "could", "couldnt", "de", "describe",
    "do", "done", "each", "eg", "either", "else", "enough", "etc", "even", "ever",
    "every", "everyone", "everything", "everywhere", "except", "few", "find","for",
    "found", "four", "from", "further", "get", "give", "go", "had", "has", "hasnt",
    "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon",
    "hers", "herself", "him", "himself", "his", "how", "however", "i", "ie", "if", "in",
    "indeed", "is", "it", "its", "itself", "keep", "least", "less", "ltd", "made", "many",
    "may", "me", "meanwhile", "might", "mine", "more", "moreover", "most", "mostly", "much",
    "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next","no",
    "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often",
    "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours",
    "ourselves", "out", "over", "own", "part","perhaps", "please", "put", "rather", "re",
    "same", "see", "seem", "seemed", "seeming", "seems", "she", "should","since", "sincere","so",
    "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still",
    "such", "take","than", "that", "the", "their", "them", "themselves", "then", "thence",
    "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they",
    "this", "those", "though", "through", "throughout", "thru", "thus", "to", "together",
    "too", "toward", "towards", "under", "until", "up", "upon", "us", "very", "was", "we",
    "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter",
    "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
    "who", "whoever", "whom", "whose", "why", "will", "with",
    "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "o", "s"
]
vocabulary = []

def loadData(path):
    # get dictionary
    dictionary = "dictionary.txt"
    vocabulary = get_vocabulary(dictionary)
    #print(vocabulary)
    # get training datasets
    negtrainpath = path + "training_set/"+"neg/"
    negtrainfiles = [f for f in listdir(negtrainpath) if isfile(join(negtrainpath, f))]
    postrainpath = path + "training_set/" + "pos/"
    postrainfiles = [f for f in listdir(postrainpath) if isfile(join(postrainpath, f))]
    # get testing datasets
    negtestpath = path + "test_set/" + "neg/"
    negtestfiles = [f for f in listdir(negtestpath) if isfile(join(negtestpath, f))]
    postestpath = path + "test_set/" + "pos/"
    postestfiles = [f for f in listdir(postestpath) if isfile(join(postestpath, f))]
    #represent each document in BOW format
    xtrain = np.zeros((len(negtrainfiles)+len(postrainfiles), len(vocabulary)))
    ytrain = np.zeros(len(negtrainfiles)+len(postrainfiles))
    i = 0
    # process training files
    for fileDj in negtrainfiles:
        #print(negtrainpath + fileDj)
        BOWDj = transfer(negtrainpath + fileDj, vocabulary)
        #print(BOWDj)
        xtrain[i,:] = BOWDj
        ytrain[i] = 0  # zero denotes negative review, 1 denotes positive
        i += 1
    for fileDj in postrainfiles:
        BOWDj = transfer(postrainpath + fileDj, vocabulary)
        xtrain[i,:] = BOWDj
        ytrain[i] = 1  # 1 denotes positive
        i += 1
    # process test files
    xtest = np.zeros((len(negtestfiles) + len(postestfiles), len(vocabulary)))
    ytest = np.zeros(len(negtestfiles) + len(postestfiles))
    i = 0
    for fileDj in negtestfiles:
        BOWDj = transfer(negtestpath + fileDj, vocabulary)
        xtest[i, :] = BOWDj
        ytest[i] = 0  # zero denotes negative review
        i += 1
    for fileDj in postestfiles:
        BOWDj = transfer(postestpath + fileDj, vocabulary)
        xtest[i, :] = BOWDj
        ytest[i] = 1  # 1 denotes positive
        i += 1
    return xtrain, xtest, ytrain, ytest

def get_vocabulary(dictionary):
    global vocabulary
    with open(dictionary, 'r') as f:
        line = f.readline().strip()
        while line:
            vocabulary.append(line)
            line = f.readline().strip()
    return vocabulary

def transfer(fileDj, vocabulary):
    BOWDj = np.zeros(len(vocabulary))
    with open(fileDj, 'r') as f:
        line = f.readline().replace('.,`!?', " ")
        line  = re.sub('\W+',' ', line)
        line = preprocess1(line)
        line = line.strip().split()
        while line:
            #print(line)
            for word in line:
                if word in vocabulary:
                    i = vocabulary.index(word)
                    BOWDj[i] = BOWDj[i] + 1
                elif word not in stop_words:
                    BOWDj[100] = BOWDj[100] + 1
            line = f.readline()
            line = re.sub('\W+',' ', line)
            line = line.strip().split()
    return BOWDj

def preprocess1(line):
    line = line.replace("loving", "love")
    line = line.replace("loves", "love")
    line = line.replace("loved", "love")
    line = line.replace("wasted", "waste")
    line = line.replace("wasting", "waste")
    line = line.replace("wastes", "waste")
    line = line.replace("richer", "rich")
    line = line.replace("richest", "rich")
    line = line.replace("powerfully", "powerful")
    line = line.replace("previously", "previous")
    line = line.replace("similarly", "similar")
    line = line.replace("personally", "personal")
    line = line.replace("earlier", "early")
    line = line.replace("earliest", "early")
    line = line.replace("worst", "worse")
    line = line.replace("darker", "dark")
    line = line.replace("darkest", "dark")
    line = line.replace("easier", "easy")
    line = line.replace("easiest", "easy")
    line = line.replace("ease", "easy")
    line = line.replace("wonderfully", "wonderful")
    line = line.replace("wilder", "wild")
    line = line.replace("wildest", "wild")
    line = line.replace("poorer", "poor")
    line = line.replace("poorest", "poor")
    #line = line.replace("", "")
    #line = line.replace("", "")
    return line


def preprocess2(line):
    return line


# See lecture 17c slide 54-56
# thetaPos == probability of a word appearing in a positive review
def naiveBayesMulFeature_train(xtrain, ytrain):
    # get class percentages
    unique, counts = np.unique(ytrain, return_counts=True)
    negratio = counts[0] / float(len(ytrain))
    posratio = counts[1] / float(len(ytrain))
    # find avg probability of a word per class using laplace smoothing fxn
    alpha = .001
    neg_word_count = np.zeros(len(xtrain[0]))
    pos_word_count = np.zeros(len(xtrain[0]))
    for i in range(len(ytrain)):
        if ytrain[i] == 0:
            neg_word_count[:] = neg_word_count[:] + xtrain[i]
        else:
            pos_word_count[:] = pos_word_count[:] + xtrain[i]
    neg_word_prob = np.zeros(len(xtrain[0]))
    pos_word_prob = np.zeros(len(xtrain[0]))
    for i in range(len(xtrain[0])):
        neg_word_prob[i] = (neg_word_count[i] + alpha) / (sum(neg_word_count) + len(vocabulary) + 1)
        pos_word_prob[i] = (pos_word_count[i] + alpha) / (sum(pos_word_count) + len(vocabulary) + 1)
    # get argmax of probabilities for collection of words in diff classes
    thetaNeg = np.zeros(len(xtrain[0]))
    for i in range(len(xtrain[0])):
        #hmap = math.log(neg_word_prob[i]/float(sum(neg_word_count))) + math.log(negratio)
        hmap = math.log(neg_word_prob[i]) + math.log(negratio)
        #print("probability of {} in neg review: {}".format(vocabulary[i], hmap))
        thetaNeg[i] = hmap
    thetaPos = np.zeros(len(xtrain[0]))
    for i in range(len(xtrain[0])):
        #hmap = math.log(pos_word_prob[i]/sum(pos_word_count)) + math.log(posratio)
        hmap = math.log(pos_word_prob[i]) + math.log(posratio)
        #print("probability of {} in pos review: {}".format(vocabulary[i], hmap))
        thetaPos[i] = hmap
    return thetaPos, thetaNeg


def naiveBayesMulFeature_test(xtest, ytest, thetaPos, thetaNeg):
    yPredict = []
    Accuracy = 0
    for i in range(len(ytest)):
        y_hat_pos = sum(xtest[i] * thetaPos)
        y_hat_neg = sum(xtest[i] * thetaNeg)
        classification = 0
        if(y_hat_pos > y_hat_neg):
            yPredict.append(1)
            classification = 1
        else:
            yPredict.append(0)
        if(classification == ytest[i]):
            Accuracy += 1
    Accuracy = Accuracy / float(len(ytest))
    return yPredict, Accuracy


def naiveBayesMulFeature_sk_MNBC(xtrain, ytrain, xtest, ytest):
    clf = MultinomialNB(alpha=5, fit_prior=True, class_prior=None)
    clf.fit(xtrain, ytrain)
    Accuracy = clf.score(xtest, ytest)
    return Accuracy



def L(p_hat, x, n):
    p = math.pow(p_hat, x) * math.pow(1 - p_hat, n - x)
    return p

def naiveBayesBernFeature_train2(xtrain, ytrain):
    thetaPosTrue = []
    thetaNegTrue = []
    # train
    # get class percentages
    unique, counts = np.unique(ytrain, return_counts=True)
    negratio = counts[0] / float(len(ytrain))
    posratio = counts[1] / float(len(ytrain))
    # find avg probability of a word per class using laplace smoothing fxn
    neg_word_count = np.zeros(len(xtrain[0]))
    pos_word_count = np.zeros(len(xtrain[0]))
    # fraction of documents in which words appear
    for i in range(len(ytrain)):
        if ytrain[i] == 0:
            neg_word_count[:] = neg_word_count[:] + xtrain[i]
        else:
            pos_word_count[:] = pos_word_count[:] + xtrain[i]
    # turn counts into bernoulli "successes"
    #for i in range(len(neg_word_count)):
    #    if(neg_word_count[i] > 0):
    #        neg_word_count[i] = 1
    # smoothing
    neg_word_prob = np.zeros(len(xtrain[0]))
    pos_word_prob = np.zeros(len(xtrain[0]))
    alpha = .001
    # get likelihood for data given parameter
    for i in range(len(xtrain[0])):
        p_hat = neg_word_count[i] / sum(neg_word_count)
        x = neg_word_count[i]
        n = 1400
        #neg_word_prob[i] = L(p_hat, x, n) # neg_word_count[i] + alpha) / (sum(neg_word_count) + len(vocabulary) + 1)
        p_hat = pos_word_count[i] / sum(pos_word_count)
        x = pos_word_count[i]
        #pos_word_prob[i] = L(p_hat, x, n) # (pos_word_count[i] + alpha) / (sum(pos_word_count) + len(vocabulary) + 1)
    # get argmax of probabilities for collection of words in diff classes
    thetaNeg = np.zeros(len(xtrain[0]))
    for i in range(len(xtrain[0])):
        p_hat = neg_word_count[i] / sum(neg_word_count)
        x = neg_word_count[i]
        n = 1400
        hmap = 0 # math.log(neg_word_prob[i]) + math.log(negratio)
        try:
            hmap = -x * math.log(p_hat)-(n-x)*math.log(1-p_hat) # -math.log(neg_word_prob[i]**neg_word_count[i] * (1-neg_word_prob[i])**(1400-neg_word_count[i]))
        except:
            hmap = 0
        print("probability of {} in neg review: {} (log={})".format(vocabulary[i], neg_word_count[i]/sum(neg_word_count), hmap))
        thetaNeg[i] = hmap
    thetaPos = np.zeros(len(xtrain[0]))
    for i in range(len(xtrain[0])):
        p_hat = pos_word_count[i] / sum(pos_word_count)
        x = pos_word_count[i]
        hmap = 0 # math.log(pos_word_prob[i]**pos_word_count[i]) + math.log(posratio)
        try:
            hmap = -x * math.log(p_hat)-(n-x)*math.log(1-p_hat) # -math.log(pos_word_prob[i]**pos_word_count[i] * (1-pos_word_prob[i])**(1400-pos_word_count[i]))
        except:
            hmap = 0
        print("probability of {} in pos review: {} (log={})".format(vocabulary[i], pos_word_count[i]/sum(pos_word_count), hmap))
        thetaPos[i] = hmap
    thetaPosTrue = thetaPos
    thetaNegTrue = thetaNeg
    return thetaPosTrue, thetaNegTrue


def naiveBayesBernFeature_train(xtrain, ytrain):
    prior_class = 0.5
    # find avg probability of a word per class using laplace smoothing fxn
    neg_word_count = np.zeros(len(xtrain[0]))
    pos_word_count = np.zeros(len(xtrain[0]))
    # fraction of documents in which words appear
    for i in range(len(ytrain)):
        if ytrain[i] == 0:
            neg_word_count[:] = neg_word_count[:] + xtrain[i]
        else:
            pos_word_count[:] = pos_word_count[:] + xtrain[i]
    # get likelihood (with add-one smoothing)
    neg_word_prob = np.zeros(len(xtrain[0]))
    pos_word_prob = np.zeros(len(xtrain[0]))
    for i in range(len(xtrain[0])):
        neg_word_prob[i] = (neg_word_count[i]+1) / (sum(neg_word_count)+2)
        pos_word_prob[i] = (pos_word_count[i]+1) / (sum(pos_word_count)+2)
    # apply bernoulli
    score = math.log(prior_class)
    #for i in range(len(xtrain[0])):
    thetaPosTrue = pos_word_prob
    thetaNegTrue = neg_word_prob
    return thetaPosTrue, thetaNegTrue

def naiveBayesBernFeature_test(xtest, ytest, thetaPosTrue, thetaNegTrue):
    yPredict = []
    Accuracy = 0
    for i in range(len(ytest)):
        pos_score = 0
        neg_score = 0
        for j in range(len(xtest[0])):
            if(xtest[i][j] > 0):
                pos_score += math.log(thetaPosTrue[j])
                neg_score += math.log(thetaNegTrue[j])
            else:
                pos_score += math.log(1 - thetaPosTrue[j])
                neg_score += math.log(1 - thetaNegTrue[j])
        classification = 0
        if(pos_score > neg_score):
            yPredict.append(1)
            classification = 1
        else:
            yPredict.append(0)
        if(classification == ytest[i]):
            Accuracy += 1
    Accuracy = Accuracy / float(len(ytest))
    return yPredict, Accuracy


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python naiveBayes.py dataSetPath")
        sys.exit()

    print("--------------------")
    textDataSetsDirectoryFullPath = sys.argv[1]
    


    xtrain, xtest, ytrain, ytest = loadData(textDataSetsDirectoryFullPath)


    thetaPos, thetaNeg = naiveBayesMulFeature_train(xtrain, ytrain)
    print("thetaPos =", thetaPos)
    print("thetaNeg =", thetaNeg)
    print("--------------------")

    yPredict, Accuracy = naiveBayesMulFeature_test(xtest, ytest, thetaPos, thetaNeg)
    print("MNBC classification accuracy =", Accuracy)

    Accuracy_sk = naiveBayesMulFeature_sk_MNBC(xtrain, ytrain, xtest, ytest)
    print("Sklearn MultinomialNB accuracy =", Accuracy_sk)


    thetaPosTrue, thetaNegTrue = naiveBayesBernFeature_train(xtrain, ytrain)
    print("thetaPosTrue =", thetaPosTrue)
    print("thetaNegTrue =", thetaNegTrue)
    print("--------------------")

    yPredict, Accuracy = naiveBayesBernFeature_test(xtest, ytest, thetaPosTrue, thetaNegTrue)
    print("BNBC classification accuracy =", Accuracy)
    print("--------------------")
