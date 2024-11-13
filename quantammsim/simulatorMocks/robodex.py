import numpy as np

#input root
class simulationRun:
    def __init__(self, pool, startDate, endDate):
        self.id = pool
        self.startDate = startDate
        self.endDate = endDate
        
class LiquidityPool:
    def __init__(self, id, constituents, updateRule):
        self.id = id
        self.poolConstituents =  np.array(constituents)
        self.updateRule = updateRule

class UpdateRule:
    def __init__(self, name, updateRuleFactors):
        self.name = name
        self.updateRuleFactors =  np.array(updateRuleFactors)


class UpdateRuleFactor:
    def __init__(self, name, value):
        self.name = name
        self.value = value


class LiquidityPoolCoin:
    def __init__(self, name, marketValue, currentPrice, amount, weight):
        self.name = name
        self.marketValue = marketValue
        self.currentPrice = currentPrice
        self.amount = amount
        self.weight = weight

#outputs
class SimulationResultTimestep:
    def __init__(self, unix, coinsHeld , timeStepTotal):
        self.unix = unix
        self.coinsHeld = np.array(coinsHeld)#LiquidityPoolCoin
        self.timeStepTotal = timeStepTotal

