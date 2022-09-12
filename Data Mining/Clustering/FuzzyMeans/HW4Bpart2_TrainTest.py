import random
import math
"""

This program utilizes Fuzzy Means to obtain a WCSSE via training and testing over the Iris Data Set.

Train (obtaining the "optimal" centroids using a portion of the Iris Data Set)
Centroids: [[0.3874429390553983, 0.842403383162783, 0.6972979360673723, 0.2581072022156241], [0.3489514108637187, 0.7540753185519056, 0.5435846869049809, 0.17076410511976794], [0.4360567950922947, 0.6411515556228333, 0.18721025988388268, 0.0343628448156183]]
WCSSE: 0.9658661351854331

Test (Testing the "optimal" centroids on the second half of the Iris Data Set)
Centroids: [[0.3962379420676815, 0.9044777937620602, 0.7494223927017092, 0.26101344648536223], [0.3676602398351401, 0.7620935457927819, 0.5803111955338698, 0.1814851394677523], [0.4693542274587017, 0.6672545779457457, 0.21769702320921167, 0.036612050362626575]]
WCSSE: 0.562226072821451
"""
def getCSVInfo():
    f = open("HW4_train.csv", "r")
    input = f.read()
    allInfo = input.split("\n")
    del allInfo[0]
    #print(allInfo)
    lystSplitOnComma = []
    for i in allInfo:
        splitLyst = i.split(",")
        lystSplitOnComma.append(splitLyst)
    #print(lystSplitOnComma)
    f.close()
    return lystSplitOnComma

def convertToFloat(csvStringLyst):
    csvFloatLyst = []
    for i in range(1,len(csvStringLyst)-1):
        #print(csvStringLyst[i])
        temp = []
        for j in range(0,len(csvStringLyst[i])-1):
            temp.append(float(csvStringLyst[i][j]))
        csvFloatLyst.append(temp)
    csvFloatLyst = normalize(csvFloatLyst)
    return csvFloatLyst

def normalize(csvFloatLyst):
    #try for later: try to normalize as sections instead of the whole
    min = 100
    max = 0
    for i in range(0,len(csvFloatLyst)):
        for j in range(0,len(csvFloatLyst[i])):
            if csvFloatLyst[i][j] < min:
                min = csvFloatLyst[i][j]
            if csvFloatLyst[i][j] > max:
                max = csvFloatLyst[i][j]
    datarange = max - min
    for i in range(0,len(csvFloatLyst)):
        for j in range(0,len(csvFloatLyst[i])):
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

def calcEuclDist(point0, point1, point2, point3 ,centroid0, centroid1, centroid2, centroid3):
    #print(math.sqrt(math.pow(point0 - centroid0, 2) + math.pow(point1 - centroid1, 2) + math.pow(point2 - centroid2, 2) + math.pow(point3 - centroid3, 2)))
    return math.sqrt(math.pow(point0 - centroid0, 2) + math.pow(point1 - centroid1, 2) + math.pow(point2 - centroid2, 2) + math.pow(point3 - centroid3, 2))

def getEuclidDistList(centroidLyst, pointsLyst):
    #returns euclid distance euclidDistLyst[0] has dist of all centroids associated with one point
    euclidDistLyst = []
    for point in pointsLyst:
        temp = []
        for centroid in centroidLyst:
            temp.append(calcEuclDist(point[0], point[1], point[2], point[3], centroid[0], centroid[1], centroid[2], centroid[3]))
        euclidDistLyst.append(temp)
    return euclidDistLyst

def calcProbSquared(euclidDistLyst):
    #ADD MORE PROBS FOR REAL
    # for x in euclidDistLyst:
    #     print(x)
    probSquaredLyst = []
    OFFSET = 0.0001
    for euclideanDistances in euclidDistLyst:
        prob1 = ( (( 1 / (euclideanDistances[0] + OFFSET)) ** 2 ) / ( ( 1 / (euclideanDistances[0] + OFFSET) ** 2 ) + (( 1 / (euclideanDistances[1] + OFFSET)) ** 2 ) + (( 1 / (euclideanDistances[2] + OFFSET)) ** 2 ) ) ) ** 2
        prob2 = ( (( 1 / (euclideanDistances[1] + OFFSET)) ** 2 ) / ( ( 1 / (euclideanDistances[0] + OFFSET) ** 2 ) + (( 1 / (euclideanDistances[1] + OFFSET)) ** 2 ) + (( 1 / (euclideanDistances[2] + OFFSET)) ** 2 ) ) ) ** 2
        prob3 = ( (( 1 / (euclideanDistances[2] + OFFSET)) ** 2 ) / ( ( 1 / (euclideanDistances[0] + OFFSET) ** 2 ) + (( 1 / (euclideanDistances[1] + OFFSET)) ** 2 ) + (( 1 / (euclideanDistances[2] + OFFSET)) ** 2 ) ) ) ** 2
        probSquaredLyst.append([prob1,prob2, prob3])
    return probSquaredLyst

def getPointTimesSquaredList(pointsLyst, probSquaredLyst):
    newCentroidVaribleList = []
    for i in range(0,len(pointsLyst)):
        squared0point0 = probSquaredLyst[i][0] * pointsLyst[i][0]
        squared0point1 = probSquaredLyst[i][0] * pointsLyst[i][1]
        squared0point2 = probSquaredLyst[i][0] * pointsLyst[i][2]
        squared0point3 = probSquaredLyst[i][0] * pointsLyst[i][3]
        #print(squared0point0, squared0point1, squared0point2, squared0point3)
        squared1point0 = probSquaredLyst[i][1] * pointsLyst[i][0]
        squared1point1 = probSquaredLyst[i][1] * pointsLyst[i][1]
        squared1point2 = probSquaredLyst[i][1] * pointsLyst[i][2]
        squared1point3 = probSquaredLyst[i][1] * pointsLyst[i][3]
        #print(squared1point0, squared1point1, squared1point2, squared1point3)
        squared2point0 = probSquaredLyst[i][2] * pointsLyst[i][0]
        squared2point1 = probSquaredLyst[i][2] * pointsLyst[i][1]
        squared2point2 = probSquaredLyst[i][2] * pointsLyst[i][2]
        squared2point3 = probSquaredLyst[i][2] * pointsLyst[i][3]
        #print(squared2point0, squared2point1, squared2point2, squared2point3)
        newCentroidVaribleList.append([squared0point0, squared0point1, squared0point2, squared0point3, squared1point0, squared1point1, squared1point2, squared1point3, squared2point0, squared2point1, squared2point2, squared2point3])
    return newCentroidVaribleList

def calcSumForProbSquaredLyst(probSquaredLyst):
    sumProbSquaredList = []
    sumProbSquared0 = 0
    sumProbSquared1 = 0
    sumProbSquared2 = 0
    for i in range(0,len(probSquaredLyst)):
        sumProbSquared0 += probSquaredLyst[i][0]
        sumProbSquared1 += probSquaredLyst[i][1]
        sumProbSquared2 += probSquaredLyst[i][2]
    sumProbSquaredList.append(sumProbSquared0)
    sumProbSquaredList.append(sumProbSquared1)
    sumProbSquaredList.append(sumProbSquared2)
    #print(sumProbSquaredList)
    return sumProbSquaredList

def calcSumForPointTimesSquareList(pointTimesSquaredList):
    sumPointTimesSquare = []
    sum0 = 0
    sum1 = 0
    sum2 = 0
    sum3 = 0
    sum4 = 0
    sum5 = 0
    sum6 = 0
    sum7 = 0
    sum8 = 0
    sum9 = 0
    sum10 = 0
    sum11 = 0
    for i in range(0,len(pointTimesSquaredList)):
        sum0 += pointTimesSquaredList[i][0]
        sum1 += pointTimesSquaredList[i][1]
        sum2 += pointTimesSquaredList[i][2]
        sum3 += pointTimesSquaredList[i][3]
        sum4 += pointTimesSquaredList[i][4]
        sum5 += pointTimesSquaredList[i][5]
        sum6 += pointTimesSquaredList[i][6]
        sum7 += pointTimesSquaredList[i][7]
        sum8 += pointTimesSquaredList[i][8]
        sum9 += pointTimesSquaredList[i][9]
        sum10 += pointTimesSquaredList[i][10]
        sum11 += pointTimesSquaredList[i][11]
    sumPointTimesSquare.append(sum0)
    sumPointTimesSquare.append(sum1)
    sumPointTimesSquare.append(sum2)
    sumPointTimesSquare.append(sum3)
    sumPointTimesSquare.append(sum4)
    sumPointTimesSquare.append(sum5)
    sumPointTimesSquare.append(sum6)
    sumPointTimesSquare.append(sum7)
    sumPointTimesSquare.append(sum8)
    sumPointTimesSquare.append(sum9)
    sumPointTimesSquare.append(sum10)
    sumPointTimesSquare.append(sum11)
    return sumPointTimesSquare

def calcNewCentroids(pointTimesSquaredList, probSquaredLyst):
    newCentroidLyst = []
    sumProbSquaredList = calcSumForProbSquaredLyst(probSquaredLyst)
    sumPointTimesSquare = calcSumForPointTimesSquareList(pointTimesSquaredList)

    centroid0spot0 = sumPointTimesSquare[0]/ sumProbSquaredList[0]
    centroid0spot1 = sumPointTimesSquare[1]/ sumProbSquaredList[0]
    centroid0spot2 = sumPointTimesSquare[2]/ sumProbSquaredList[0]
    centroid0spot3 = sumPointTimesSquare[3]/ sumProbSquaredList[0]
    newCentroidLyst.append([centroid0spot0, centroid0spot1, centroid0spot2, centroid0spot3])
    centroid1spot0 = sumPointTimesSquare[4]/ sumProbSquaredList[1]
    centroid1spot1 = sumPointTimesSquare[5]/ sumProbSquaredList[1]
    centroid1spot2 = sumPointTimesSquare[6]/ sumProbSquaredList[1]
    centroid1spot3 = sumPointTimesSquare[7]/ sumProbSquaredList[1]
    newCentroidLyst.append([centroid1spot0, centroid1spot1, centroid1spot2, centroid1spot3])
    centroid2spot0 = sumPointTimesSquare[8]/ sumProbSquaredList[2]
    centroid2spot1 = sumPointTimesSquare[9]/ sumProbSquaredList[2]
    centroid2spot2 = sumPointTimesSquare[10]/ sumProbSquaredList[2]
    centroid2spot3 = sumPointTimesSquare[11]/ sumProbSquaredList[2]
    newCentroidLyst.append([centroid2spot0, centroid2spot1, centroid2spot2, centroid2spot3])
    return newCentroidLyst

def startCalcNewCentroids(pointsLyst, probSquaredLyst):
    pointTimesSquaredList = getPointTimesSquaredList(pointsLyst, probSquaredLyst)
    newCentroidLyst = calcNewCentroids(pointTimesSquaredList, probSquaredLyst)
    return newCentroidLyst

def calcWCSSE(euclidDistLyst):
    WCSSEsum = []
    for i in range(0,len(euclidDistLyst)):
        WCSSEsum.append(min(euclidDistLyst[i]) ** 2)
    print("WCSSE: " + str(sum(WCSSEsum)))

def findPointAssociatedWithCentroid(euclidDistLyst):
    #for graph specifically
    positionList = []
    for dist in euclidDistLyst:
        minimum = min(dist)
        CentroidAssociated = dist.index(minimum)
        positionList.append(CentroidAssociated)
    return positionList

def getClusterInfo(euclidDistLyst, csvStringLyst):
    clusterA = []
    clusterB = []
    clusterC = []
    for i in range(len(euclidDistLyst)):
        minimum = min(euclidDistLyst[i])
        minPos = euclidDistLyst[i].index(minimum)
        if minPos == 0:
            clusterA.append(csvStringLyst[i])
        elif minPos == 1:
            clusterB.append(csvStringLyst[i])
        elif minPos == 2:
            clusterC.append(csvStringLyst[i])
    print("Virginica",len(clusterA))
    for a in clusterA:
        print(a)
    print("Versicolor", len(clusterB))
    for b in clusterB:
        print(b)
    print("Setosa", len(clusterC))
    for c in clusterC:
        print(c)

def kmeans(k,pointsLyst, csvStringLyst):
    centroidLyst = generateCentroids(k)
    centroidLyst = [[0.3874429390553983, 0.842403383162783, 0.6972979360673723, 0.2581072022156241], 
                    [0.3489514108637187, 0.7540753185519056, 0.5435846869049809, 0.17076410511976794], 
                    [0.4360567950922947, 0.6411515556228333, 0.18721025988388268, 0.0343628448156183]]
    #for i in range(0,50):
    euclidDistLyst = getEuclidDistList(centroidLyst, pointsLyst)
    probSquaredLyst = calcProbSquared(euclidDistLyst)
    newCentroidLyst = startCalcNewCentroids(pointsLyst, probSquaredLyst)
    centroidLyst = newCentroidLyst
    #positionList = findPointAssociatedWithCentroid(euclidDistLyst)
    #print(centroidLyst)  #UNCOMMENT TO PRINT CENTROIDS
    calcWCSSE(euclidDistLyst)
    getClusterInfo(euclidDistLyst, csvStringLyst)

def main():
    csvStringLyst = getCSVInfo()
    pointsLyst = convertToFloat(csvStringLyst)

    kmeans(3,pointsLyst, csvStringLyst)
main()