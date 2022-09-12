import random
import math
"""
This Program utilizes K-Means to cluster the Iris Dataset with "k" number(s) of Centroids.

Training
[[0.29, 0.58, 0.17, 0.04, 'setosa'], [0.37, 0.56, 0.18, 0.03, 'setosa'], [0.38, 0.55, 0.14, 0.01, 'setosa'], [0.38, 0.56, 0.17, 0.03, 'setosa'], [0.38, 0.62, 0.18, 0.04, 'setosa'], [0.38, 0.62, 0.18, 0.01, 'setosa'], [0.38, 0.63, 0.18, 0.03, 'setosa'], [0.38, 0.64, 0.21, 0.03, 'setosa'], [0.4, 0.59, 0.19, 0.03, 'setosa'], [0.4, 0.63, 0.19, 0.01, 'setosa'], [0.4, 0.63, 0.19, 0.03, 'setosa'], [0.41, 0.56, 0.17, 0.03, 'setosa'], [0.41, 0.59, 0.18, 0.03, 'setosa'], [0.41, 0.6, 0.17, 0.03, 'setosa'], [0.41, 0.64, 0.15, 0.03, 'setosa'], [0.42, 0.64, 0.18, 0.03, 'setosa'], [0.42, 0.65, 0.22, 0.06, 'setosa'], [0.44, 0.59, 0.18, 0.04, 'setosa'], [0.44, 0.62, 0.24, 0.03, 'setosa'], [0.44, 0.62, 0.21, 0.03, 'setosa'], [0.44, 0.64, 0.21, 0.05, 'setosa'], [0.44, 0.64, 0.19, 0.03, 'setosa'], [0.44, 0.65, 0.19, 0.03, 'setosa'], [0.44, 0.67, 0.18, 0.03, 'setosa'], [0.44, 0.69, 0.19, 0.05, 'setosa'], [0.44, 0.69, 0.22, 0.03, 'setosa'], [0.45, 0.64, 0.21, 0.08, 'setosa'], [0.45, 0.64, 0.17, 0.04, 'setosa'], [0.45, 0.65, 0.18, 0.03, 'setosa'], [0.45, 0.65, 0.18, 0.04, 'setosa'], [0.45, 0.71, 0.17, 0.03, 'setosa'], [0.46, 0.59, 0.13, 0.03, 'setosa'], [0.46, 0.63, 0.18, 0.01, 'setosa'], [0.46, 0.64, 0.18, 0.03, 'setosa'], [0.47, 0.65, 0.19, 0.05, 'setosa'], [0.47, 0.68, 0.19, 0.03, 'setosa'], [0.47, 0.69, 0.19, 0.03, 'setosa'], [0.49, 0.73, 0.22, 0.04, 'setosa'], [0.5, 0.69, 0.22, 0.05, 'setosa'], [0.5, 0.69, 0.17, 0.05, 'setosa'], [0.51, 0.74, 0.15, 0.03, 'setosa'], [0.53, 0.67, 0.19, 0.01, 'setosa'], [0.54, 0.71, 0.18, 0.03, 'setosa'], [0.56, 0.73, 0.19, 0.05, 'setosa']]
[[0.26, 0.64, 0.45, 0.13, 'versicolor'], [0.28, 0.77, 0.51, 0.13, 'versicolor'], [0.29, 0.64, 0.42, 0.13, 'versicolor'], [0.29, 0.71, 0.51, 0.17, 'versicolor'], [0.29, 0.81, 0.56, 0.17, 'versicolor'], [0.31, 0.71, 0.47, 0.13, 'versicolor'], [0.32, 0.65, 0.38, 0.14, 'versicolor'], [0.32, 0.71, 0.51, 0.17, 'versicolor'], [0.32, 0.72, 0.5, 0.14, 'versicolor'], [0.32, 0.81, 0.63, 0.19, 'versicolor'], [0.33, 0.71, 0.56, 0.15, 'versicolor'], [0.33, 0.73, 0.45, 0.13, 'versicolor'], [0.33, 0.74, 0.51, 0.15, 'versicolor'], [0.35, 0.67, 0.5, 0.18, 'versicolor'], [0.35, 0.74, 0.53, 0.13, 'versicolor'], [0.35, 0.74, 0.5, 0.15, 'versicolor'], [0.35, 0.77, 0.65, 0.21, 'versicolor'], [0.36, 0.73, 0.53, 0.17, 'versicolor'], [0.36, 0.78, 0.51, 0.17, 'versicolor'], [0.36, 0.78, 0.6, 0.15, 'versicolor'], [0.36, 0.83, 0.59, 0.19, 'versicolor'], [0.36, 0.87, 0.62, 0.18, 'versicolor'], [0.37, 0.72, 0.46, 0.17, 'versicolor'], [0.37, 0.73, 0.54, 0.17, 'versicolor'], [0.37, 0.77, 0.58, 0.19, 'versicolor'], [0.37, 0.78, 0.6, 0.18, 'versicolor'], [0.37, 0.79, 0.55, 0.17, 'versicolor'], [0.37, 0.82, 0.55, 0.17, 'versicolor'], [0.37, 0.85, 0.59, 0.17, 'versicolor'], [0.38, 0.72, 0.53, 0.17, 'versicolor'], [0.38, 0.76, 0.54, 0.19, 'versicolor'], [0.38, 0.78, 0.59, 0.18, 'versicolor'], [0.38, 0.85, 0.56, 0.18, 'versicolor'], [0.4, 0.86, 0.6, 0.19, 'versicolor'], [0.4, 0.86, 0.56, 0.18, 'versicolor'], [0.41, 0.76, 0.62, 0.23, 'versicolor'], [0.41, 0.82, 0.58, 0.19, 'versicolor'], [0.41, 0.9, 0.6, 0.18, 'versicolor'], [0.42, 0.81, 0.6, 0.21, 'versicolor'], [0.28, 0.77, 0.64, 0.19, 'virginica'], [0.32, 0.63, 0.58, 0.22, 'virginica'], [0.35, 0.74, 0.65, 0.24, 'virginica'], [0.35, 0.74, 0.65, 0.24, 'virginica'], [0.35, 0.81, 0.63, 0.23, 'virginica'], [0.36, 0.79, 0.62, 0.23, 'virginica'], [0.36, 0.81, 0.65, 0.19, 'virginica'], [0.38, 0.76, 0.65, 0.23, 'virginica'], [0.38, 0.77, 0.62, 0.23, 'virginica']]
[[0.38, 0.86, 0.64, 0.22, 'versicolor'], [0.32, 0.86, 0.74, 0.23, 'virginica'], [0.33, 0.78, 0.72, 0.18, 'virginica'], [0.35, 0.82, 0.68, 0.24, 'virginica'], [0.36, 0.74, 0.65, 0.31, 'virginica'], [0.36, 0.82, 0.72, 0.27, 'virginica'], [0.36, 0.82, 0.72, 0.28, 'virginica'], [0.36, 0.99, 0.86, 0.26, 'virginica'], [0.37, 0.81, 0.72, 0.23, 'virginica'], [0.38, 0.83, 0.74, 0.28, 'virginica'], [0.38, 0.83, 0.67, 0.26, 'virginica'], [0.38, 0.86, 0.67, 0.29, 'virginica'], [0.38, 0.87, 0.71, 0.27, 'virginica'], [0.38, 0.91, 0.76, 0.27, 'virginica'], [0.38, 0.92, 0.74, 0.21, 'virginica'], [0.4, 0.82, 0.71, 0.23, 'virginica'], [0.4, 0.88, 0.69, 0.27, 'virginica'], [0.4, 0.88, 0.65, 0.29, 'virginica'], [0.41, 0.82, 0.68, 0.29, 'virginica'], [0.41, 0.83, 0.65, 0.26, 'virginica'], [0.41, 0.87, 0.76, 0.29, 'virginica'], [0.41, 0.88, 0.73, 0.29, 'virginica'], [0.42, 0.81, 0.77, 0.32, 'virginica'], [0.42, 0.86, 0.73, 0.32, 'virginica'], [0.44, 0.79, 0.69, 0.29, 'virginica'], [0.44, 0.81, 0.72, 0.31, 'virginica'], [0.46, 0.92, 0.78, 0.32, 'virginica'], [0.49, 1.01, 0.82, 0.26, 'virginica']]

Total items in cluster A: 44
Total items in cluster B: 48
Total items in cluster C: 28
Number of Centroids(k): 3
WCSSE: 0.9696391062856806
[[0.4375, 0.6415909090909092, 0.18522727272727274, 0.03363636363636366], [0.3516666666666666, 0.76375, 0.5589583333333332, 0.17875000000000005], [0.3921428571428572, 0.8535714285714285, 0.7185714285714286, 0.2692857142857143]]

TEST
[[0.41, 0.64, 0.21, 0.03, 'setosa'], [0.43, 0.63, 0.21, 0.03, 'setosa'], [0.47, 0.69, 0.2, 0.03, 'setosa'], [0.51, 0.68, 0.2, 0.04, 'setosa'], [0.51, 0.68, 0.21, 0.03, 'setosa'], [0.51, 0.68, 0.25, 0.05, 'setosa']]
[[0.29, 0.83, 0.6, 0.2, 'versicolor'], [0.32, 0.65, 0.44, 0.13, 'versicolor'], [0.32, 0.73, 0.51, 0.15, 'versicolor'], [0.36, 0.75, 0.56, 0.17, 'versicolor'], [0.37, 0.76, 0.6, 0.17, 'versicolor'], [0.4, 0.72, 0.6, 0.2, 'versicolor'], [0.4, 0.75, 0.6, 0.2, 'versicolor'], [0.4, 0.76, 0.56, 0.16, 'versicolor'], [0.45, 0.8, 0.6, 0.21, 'versicolor']]
[[0.41, 0.92, 0.65, 0.2, 'versicolor'], [0.33, 0.76, 0.67, 0.27, 'virginica'], [0.33, 0.84, 0.67, 0.25, 'virginica'], [0.35, 1.03, 0.92, 0.31, 'virginica'], [0.37, 0.75, 0.65, 0.27, 'virginica'], [0.37, 0.99, 0.81, 0.25, 'virginica'], [0.39, 0.97, 0.84, 0.24, 'virginica'], [0.4, 0.81, 0.65, 0.24, 'virginica'], [0.4, 0.87, 0.73, 0.24, 'virginica'], [0.4, 1.01, 0.88, 0.28, 'virginica'], [0.4, 1.03, 0.81, 0.31, 'virginica'], [0.41, 0.89, 0.75, 0.32, 'virginica'], [0.43, 0.96, 0.8, 0.24, 'virginica'], [0.44, 0.89, 0.76, 0.28, 'virginica'], [0.51, 1.03, 0.89, 0.29, 'virginica']]

Total items in cluster A: 6
Total items in cluster B: 9
Total items in cluster C: 15
Number of Centroids(k): 3
WCSSE: 0.49790938736419166
[[0.47333333333333333, 0.6666666666666666, 0.21333333333333335, 0.035], [0.36777777777777776, 0.7499999999999999, 0.5633333333333334, 0.17666666666666664], [0.396, 0.9166666666666667, 0.7653333333333333, 0.26600000000000007]]

"""
def getCSVInfo():
    f = open("HW4_test.csv", "r")
    input = f.read()
    allInfo = input.split("\n")
    #print(allInfo)
    lystSplitOnComma = []
    for i in allInfo:
        splitLyst = i.split(",")
        lystSplitOnComma.append(splitLyst)
    #print(lystSplitOnComma)
    f.close()
    return lystSplitOnComma

def convertCSVtoUsable(csvStringLyst):
    csvFullLyst = []
    for i in range(1,len(csvStringLyst)-1):
        attA = float(csvStringLyst[i][0])
        attB = float(csvStringLyst[i][1])
        attC = float(csvStringLyst[i][2])
        attD = float(csvStringLyst[i][3])
        attE = csvStringLyst[i][4].replace("'", "")
        csvFullLyst.append([attA, attB, attC, attD, attE])
    csvFullLyst = normalize(csvFullLyst)  
    return csvFullLyst

def normalize(csvFloatLyst):
    #try for later: try to normalize as sections instead of the whole
    min = 100
    max = 0
    #print(csvFloatLyst)
    for i in range(0,len(csvFloatLyst)):
        #print(csvFloatLyst[i])
        for j in range(0,len(csvFloatLyst[i])-1):
            if csvFloatLyst[i][j] < min:
                min = csvFloatLyst[i][j]
            if csvFloatLyst[i][j] > max:
                max = csvFloatLyst[i][j]
    datarange = max - min
    for i in range(0,len(csvFloatLyst)):
        for j in range(0,len(csvFloatLyst[i])-1):
            csvFloatLyst[i][j] = round(csvFloatLyst[i][j] / datarange, 2)
    return csvFloatLyst
    
def generateCentroids(k):
    centroidLyst = []
    for i in range(0,k):
        temp = []
        for j in range(0,4):
            temp.append(round(random.random(), 2))
        centroidLyst.append(temp)
    return centroidLyst

def calcEuclidDist(point0, point1, point2, point3, centroid0, centroid1, centroid2, centroid3):
     #print(point0, point1, centroid0, centroid1)
     return math.sqrt(math.pow(point0 - centroid0, 2) + math.pow(point1 - centroid1, 2) + math.pow(point2 - centroid2, 2)+ math.pow(point3 - centroid3, 2))


def getEuclDist(centroidList, allPoints):
    distList = []
    for point in range(len(allPoints)):
        temp = []
        for centroid in range(len(centroidList)):
            #print(allPoints[point][0], allPoints[point][1], allPoints[point][2], allPoints[point][3], centroidList[centroid][0], centroidList[centroid][1], centroidList[centroid][2], centroidList[centroid][3])
            ed = calcEuclidDist(allPoints[point][0], allPoints[point][1], allPoints[point][2], allPoints[point][3], centroidList[centroid][0], centroidList[centroid][1], centroidList[centroid][2], centroidList[centroid][3])
            temp.append(ed)
        distList.append(temp)
    return distList


def calcMinDist(euclDistList):
    positionAndMinList = []
    for i in range(len(euclDistList)):
        minimum = min(euclDistList[i])
        minPosition = euclDistList[i].index(minimum)
        positionAndMinList.append([minPosition,  minimum])
    return positionAndMinList

def zeroSumHeuristic():
    temp = []
    for j in range(0,4):
        temp.append(round(random.random(), 2))
    return temp

def calcNewCentroids(positionAndMinList, allPoints, centroidList):
    sum0 = []
    sum1 = []
    sum2 = []
    sum3 = []
    newCentroids = []
    for i in range(len(centroidList)):
        for j in range(len(allPoints)):
            if i == positionAndMinList[j][0]:
                sum0.append(allPoints[j][0])
                sum1.append(allPoints[j][1])
                sum2.append(allPoints[j][2])
                sum3.append(allPoints[j][3])
        #print("Sums: ", sum(sum0),sum(sum1),sum(sum2),sum(sum3))
        if sum(sum0) == 0 and sum(sum1)==0 and sum(sum2)==0 and sum(sum3) == 0:
            newCentroids.append(zeroSumHeuristic())
        else:
            newCentroids.append([sum(sum0)/len(sum0), sum(sum1)/len(sum1), sum(sum2)/len(sum2), sum(sum3)/len(sum3)])
        sum0 = []
        sum1 = []
        sum2 = []
        sum3 = []
    return newCentroids

def calcWCSSE(euclDistList):
    #print(euclDistList)
    WCSSEList = []
    for i in range(len(euclDistList)):
        minimum = min(euclDistList[i]) ** 2
        WCSSEList.append(minimum)
    return sum(WCSSEList)

def getClusterInfo(centroidList, allPoints, positionAndMinList):
    # print("Centroids: ", centroidList)
    # print("allPoints: ", allPoints)
    # print("centroidDistance: ", positionAndMinList)
    clusterA = []
    clusterB = []
    clusterC = []
    for i in range(0,len(positionAndMinList)):
        minDistPosition = positionAndMinList[i][0]
        if minDistPosition == 0:
            clusterA.append(allPoints[i])
        elif minDistPosition == 1:
            clusterB.append(allPoints[i])
        elif minDistPosition == 2:
            clusterC.append(allPoints[i])
    #print(clusterA) # Uncomment to see items in each cluster
    #print(clusterB) # Uncomment to see items in each cluster
    #print(clusterC) # Uncomment to see items in each cluster
    return clusterA, clusterB, clusterC

def clusterNumbers(clusterA, clusterB, clusterC):
    print("Total items in cluster A: " + str(len(clusterA)))
    print("Total items in cluster B: " + str(len(clusterB)))
    print("Total items in cluster C: " + str(len(clusterC)))

def kmeans(allPoints):
    k = 3
    #centroidList = generateCentroids(k)
    centroidList = [[0.4375, 0.6415909090909092, 0.18522727272727274, 0.03363636363636366], [0.3516666666666666, 0.76375, 0.5589583333333332, 0.17875000000000005], [0.3921428571428572, 0.8535714285714285, 0.7185714285714286, 0.2692857142857143]]
    #for i in range(0,10):
    euclDistList = getEuclDist(centroidList, allPoints)
    positionAndMinList = calcMinDist(euclDistList)
    newCentroids = calcNewCentroids(positionAndMinList, allPoints, centroidList)
    centroidList = newCentroids
    clusterA, clusterB, clusterC = getClusterInfo(centroidList, allPoints, positionAndMinList)
    clusterNumbers(clusterA, clusterB, clusterC)
    print("Number of Centroids(k): " + str(k))
    print("WCSSE: " + str(calcWCSSE(euclDistList)))
    #print(newCentroids)


def main():
    csvList = getCSVInfo()
    allPoints = convertCSVtoUsable(csvList)
    # for x in allPoints:
    #     print(x)
    kmeans(allPoints)

main()
