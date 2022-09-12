import math
import random
from inputLayer import InputLayer
from hiddenLayer import HiddenLayer
from outputLayer import OutputLayer
from normalize import Normalize
from storeBestData import StoreBestData

"""

This program trains an artificial neural network to double the inputed number.

Training
RMSE: 0.012543168259994216
[-1.664010968271021, 0.16230076177834776, -1.7137104379088788, -1.0622743038970073]
[-1.4554067556612462, 0.0715807499212076, -2.522465683023197, -1.5030041426908416]
LR = .001

Testing
RMSE: 0.015697552166878105
[-1.6640113785195707, 0.16230078078598814, -1.7137111500388171, -1.0622747194074116]
[-1.4554053072708872, 0.07158269386509208, -2.5224642495829768, -1.503002517439754]
LR = .001
"""

def main():
    def RMSE(forRMSE):
        n = []
        for i in forRMSE:
            n.append(i**2)
        n = math.sqrt(sum(n)/len(n))
        return n

    learningRate = .001
    nInp = Normalize()
    nTrg = Normalize()
    s = StoreBestData()

    inputsList = []
    for i in range(500):
        inputsList.append(random.uniform(-5,5))
    allInputs = nInp.normalize(inputsList)
    

    actualTargetList = []
    for a in inputsList:
        actualTargetList.append(a * 2)
    allTarget = nTrg.normalize(actualTargetList)
    

    #nT = Normalize()
    #nActualTargetLyst = nT.normalize(actualTargetList)
################################3
    testNorm = Normalize()
    bestRMSE = 100
    for i in range(10):
        print("PASS", i)
        innerNodes = []
        outerNodes = []
        RMSEat100 = []
        for x in range(10):
            innerNodes.append(random.uniform(-1,1))
            outerNodes.append(random.uniform(-1,1))
        for epoch in range(10000):
            forRMSE = []
            observedTargets = []
            #random.shuffle(allInputs)
            for inp in allInputs:
                inputLayer = InputLayer(inp)
                target = inp*2
                outputLayer = OutputLayer(target)
                hiddenLayer = HiddenLayer(learningRate, inputLayer.getInputNodes(),innerNodes, outerNodes)
                obsResult = hiddenLayer.forwardProp()
                delta = outputLayer.runOutputLayer(obsResult)
                innerNodes, outerNodes = hiddenLayer.backpropagation(delta)
                observedTargets.append(outputLayer.obsResult)
                forRMSE.append(outputLayer.target - outputLayer.obsResult)
            x = RMSE(forRMSE)
            if epoch % 100 == 0:
                RMSEat100.append(x)
                #print(epoch,x)
            if x < bestRMSE:
                #observedTargets = nTrg.denormalizeLyst(observedTargets)
                
                t = testNorm.normalize(observedTargets)
                nnTargets = nTrg.denormalizeLyst(t)
                s.storeData(bestRMSE, inputsList, actualTargetList,nnTargets, innerNodes, outerNodes, RMSEat100)
            bestRMSE = x
        print(i, bestRMSE)
    s.convertToCSV()
###################TEST############################3
    # testInpNorm = Normalize()
    # testActualInpNorm = Normalize()
    # testObservedNorm = Normalize()
    # inputsList = []

    # actualTargetList = []
    # for i in range(500):
    #     inputsList.append(random.uniform(-5,5))
    # for j in inputsList:
    #     actualTargetList.append(j*2)

    # testAllInputs = testInpNorm.normalize(inputsList)
    # testActualInpNorm.normalize(actualTargetList)
    # observedTargets = []
    # forRMSE = []
    # num = 0
    # for test in testAllInputs:
    #     inputLayer = InputLayer(test)
    #     target = test*2
    #     outputLayer = OutputLayer(target)
    #     hiddenLayer = HiddenLayer(learningRate, inputLayer.getInputNodes(),[-1.664010968271021, 0.16230076177834776, -1.7137104379088788, -1.0622743038970073], [-1.4554067556612462, 0.0715807499212076, -2.522465683023197, -1.5030041426908416])
    #     obsResult = hiddenLayer.forwardProp()
    #     delta = outputLayer.runOutputLayer(obsResult)
    #     innerNodes, outerNodes = hiddenLayer.backpropagation(delta)
    #     observedTargets.append(outputLayer.obsResult)
    #     forRMSE.append(outputLayer.target - outputLayer.obsResult)
    #     if num % 100 == 0:
            
    #         print(num, RMSE(forRMSE))
    #         forRMSE = []
    #     num+=1
    # bestRMSE = RMSE(forRMSE)
    # t = testObservedNorm.normalize(observedTargets)
    # nnTargets = nTrg.denormalizeLyst(t)
    # s.storeData(bestRMSE, inputsList, actualTargetList,nnTargets, innerNodes, outerNodes, None)
    # #s.storeData(bestRMSE, inputsList, actualTargetList,observedTargets, innerNodes, outerNodes)
    # s.convertToCSV()
main()

