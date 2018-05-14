import numpy as np
import matplotlib.pyplot as plt
from math import log

def dealMatrix(matrix): #转置
    matrix = matrix.transpose()
    return matrix

def dealEntropy(dataSet, num): #计算熵
    n = len(dataSet)
    count = {}
    H = 0
    for data in dataSet:
        currentLabel = data[num]
        if currentLabel not in count.keys():  #若字典中不存在该类别标签，即创建  
            count[currentLabel] = 0  
        count[currentLabel] += 1    #递增类别标签的值  
    for key in count:
        px = float(count[key]) / float(n)  #计算某个标签的概率  
        H -= px * log(px, 2)  #计算信息熵  
    return H

def splitDataSet(dataSet, axis, value): #分割
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
     
    return retDataSet

def chooseBestFeatureToSplit(dataSet):
    numberFeatures = len(dataSet[0])-1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0;
    bestFeature = -1;
    for i in range(numberFeatures):
        featList = [example[i] for example in dataSet]
        uniqueVals = set(featList)
        newEntropy =0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        if(infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature










f = open('iris.txt','r')
first_ele = True
for data in f.readlines():
    data = data.strip(',"Iris-virginica"\n')
    data = data.strip(',"Iris-setosa"\n')
    data = data.strip(',"Iris-versicolor"\n')
    nums = data.split(",")
    if first_ele:
        nums = [float(x) for x in nums]
        matrix = np.array(nums)
        first_ele = False
    else:
        nums = [float(x) for x in nums]
        matrix = np.c_[matrix, nums]
D = dealMatrix(matrix)
print(dealEntropy(D, 0))