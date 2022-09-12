"""

Normalizes the data.

"""


class Normalize:

    def normalize(self,input):
        minimum, maximum = self.findMaxAndMin(input)
        self.minimum = minimum
        self.maximum = maximum
        return self.calcNorm(minimum, maximum, input)

    def normalizeSingle(self,x):
        return ((x - self.minimum) / (self.maximum - self.minimum))

    def findMaxAndMin(self, input):
        minimum = 100
        maximum = -100
        for i in range(len(input)):
            if input[i] < minimum:
                minimum = input[i]
            if input[i] > maximum:
                maximum = input[i]
        return minimum, maximum
    
    def calcNorm(self, minimum, maximum, input):
        normInputs = self.findNormalizedValues(minimum, maximum, input)
        return normInputs

    def findNormalizedValues(self,minimum, maximum, inputs):
        normInputs =[]
        for i in range(len(inputs)):
            normInputs.append((inputs[i] - minimum) / (maximum - minimum))
        return normInputs

    def denormalize(self, input):
        return (input * (self.maximum - self.minimum) + self.minimum)

    def denormalizeLyst(self,input):
        for i in range(len(input)):
            input[i] = (input[i] * (self.maximum - self.minimum) + self.minimum)
        return input
        
