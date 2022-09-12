"""
This class stores the best set of data.
"""

class StoreBestData:
    def storeData(self,RMSE, inputNodes,targets,observedTargets,weight1, weight2, RMSEat100):
        self.RMSE = RMSE
        self.inputNodes = inputNodes
        self.targets = targets
        self.observedTargets = observedTargets
        self.weight1 = weight1
        self.weight2 = weight2
        self.RMSEat100 = RMSEat100
    
    def convertToCSV(self):
        self.convertActualInputNodesAndTargets()
        self.convertInputNodesAndObservedTargets()
        #self.printRMSE100()
        self.printRMSEandWeights()
    def printRMSE100(self):
        x = 0
        f = open("RMSEat100.csv", "w")
        for i in range(len(self.RMSEat100)):
            f.write(str(x)+","+str(self.RMSEat100[i]) + "\n")
            x += 100
        f.close()
    def convertActualInputNodesAndTargets(self):
        f = open("actualInputAndTarget.csv", "w")
        for i in range(len(self.inputNodes)):
            f.write(str(self.inputNodes[i])+","+str(self.targets[i]) + "\n")
        f.close()

    def convertInputNodesAndObservedTargets(self):
        f = open("observedInputAndTarget.csv", "w")
        for i in range(len(self.inputNodes)):
            f.write(str(self.inputNodes[i])+","+str(self.observedTargets[i]) + "\n")
        f.close()
    
    def printRMSEandWeights(self):
        print(self.RMSE)
        print(self.weight1)
        print(self.weight2)

