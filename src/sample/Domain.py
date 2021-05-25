class Domain():
    def __init__(self, name, clustList):
        self.name = name
        self.clustList = clustList

    def __repr__(self):
        return self.name

    def getName(self):
        return self.name

    def getClustList(self):
        return self.clustList

