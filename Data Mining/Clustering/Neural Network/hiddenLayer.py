import math
"""

Handles the forward and backward propregration of the neural network

"""
class HiddenLayer:
    def __init__(self, learningRate, inputNodes,firstweights, secondWeights):
        self.learningRate = learningRate
        self.inputNodes = inputNodes
        self.firstweights = firstweights
        self.secondWeights = secondWeights

    def sigmoidFct(self, input):
        return 1 /(1 + math.exp(-input))

    def calcDerivative(self, input):
        deriv = self.sigmoidFct(input) * (1 - self.sigmoidFct(input))
        return deriv

    def forwardProp(self):
        nodeFirstWeightList = self.calculateFirstWeightNodeOutput()
        nodeSecondWeightList = self.calcSecondWeightNodeOutput(nodeFirstWeightList)
        return self.calcObservedResult(nodeSecondWeightList)
    
    def calculateFirstWeightNodeOutput(self):
        nodeOutputList = []
        for i in range(len(self.firstweights)):
            nodeOutput = self.inputNodes * self.firstweights[i] + 1
            nodeOutputList.append(self.sigmoidFct(nodeOutput))
        self.nodeOutputList = nodeOutputList
        return nodeOutputList
    
    def calcSecondWeightNodeOutput(self, nodeFirstWeightList):
        secondWeightNodes = []
        for i in range(len(nodeFirstWeightList)):
            secondWeightNodes.append(self.nodeOutputList[i] * self.secondWeights[i] + 1)#self.biasNodes[i])
        return secondWeightNodes
    
    def calcObservedResult(self, nodeSecondWeightList):
        return sum(nodeSecondWeightList)
    #############################################################

    def backpropagation(self, delta):
        
        errorTermList = self.calcErrorTerms(delta)
        newSecondWeightList = self.recalibrateSecondWeights(delta, errorTermList)
        newFirstWeightList = self.recalibrateFirstWeights(errorTermList)
        return newFirstWeightList, newSecondWeightList

    def calcErrorTerms(self, delta):
        errorTermList = []
        for i in range(len(self.secondWeights)):
            #errorTermList.append(self.secondWeights[i] * delta)
            errorTermList.append(delta * self.secondWeights[i] * self.calcDerivative(self.nodeOutputList[i]))
        return errorTermList
    
    def recalibrateSecondWeights(self, delta, errorTermList):
        newSecondWeightList = []
        for i in range(len(self.secondWeights)):
            newSecondWeightList.append(-self.learningRate * delta * self.nodeOutputList[i] + self.secondWeights[i])
        return newSecondWeightList

    def recalibrateFirstWeights(self, errorTermList):
        newFirstWeightList = []
        for i in range(len(self.firstweights)):
            newFirstWeightList.append(-self.learningRate * errorTermList[i] * self.inputNodes + self.firstweights[i])
        return newFirstWeightList