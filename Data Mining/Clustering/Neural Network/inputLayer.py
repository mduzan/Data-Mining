"""

Gets and sets the input.

"""


class InputLayer:
    def __init__(self, inputNodes):
        self.inputNodes = inputNodes
    
    def getInputNodes(self):
        return self.inputNodes

    def setInputNodes(self, input):
        self.inputNodes = input
