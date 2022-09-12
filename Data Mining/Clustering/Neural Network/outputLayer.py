import math
"""

Calculates the error between the actual doubled number and the resulting number from the neural network.

"""
class OutputLayer:
    def __init__(self, target):
        self.target = target
    
    def sigmoidFct(self, input):
        return 1 /(1 + math.exp(-input))

    def calcDerivative(self, input):
        deriv = self.sigmoidFct(input) * (1 - self.sigmoidFct(input))
        return deriv

    def runOutputLayer(self,obsResult):
        self.obsResult = obsResult
        return self.calcError(obsResult)

    def calcError(self, obsResult):
        #print(obsResult)
        delta = (-(self.target - obsResult) * self.calcDerivative(obsResult))
        return delta

    