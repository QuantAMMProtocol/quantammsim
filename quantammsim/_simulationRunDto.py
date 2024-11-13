import json


# input root
class SimulationRunDto(object):
    def __init__(self, jsonDto):
        print("run const")
        self.pool = LiquidityPoolDto(jsonDto["pool"])
        print(jsonDto["startUnix"])
        print(jsonDto["endUnix"])
        self.startDate = jsonDto["startUnix"]
        self.endDate = jsonDto["endUnix"]
        self.startDateString = jsonDto["startDateString"]
        self.endDateString = jsonDto["endDateString"]


class LiquidityPoolDto(object):
    def __init__(self, poolDto):
        print("pool const")
        self.id = poolDto["id"]
        poolConstituents = list()
        for coin in poolDto["poolConstituents"]:
            poolConstituents.append(LiquidityPoolCoinDto(coin))
        self.poolConstituents = poolConstituents
        self.updateRule = UpdateRuleDto(poolDto["updateRule"])


class UpdateRuleDto(object):
    def __init__(self, ruleDto):
        print("rule const")
        self.name = ruleDto["name"]
        factors = list()
        for coin in ruleDto["updateRuleParameters"]:
            factors.append(UpdateRuleFactorDto(coin))
        self.updateRuleFactors = factors


class UpdateRuleFactorDto(object):
    def __init__(self, factorDto):
        self.name = factorDto["name"]
        self.value = factorDto["value"]


class LiquidityPoolCoinDto(object):
    def __init__(self, coinDto=None):
        if coinDto is None:
            return

        self.coinCode = coinDto["coinCode"]
        self.marketValue = coinDto["marketValue"]
        self.currentPrice = coinDto["currentPrice"]
        self.amount = coinDto["amount"]
        self.weight = coinDto["weight"]


# outputs
class SimulationResult(object):
    def __init__(self, resultTimeSteps):
        self.timeSteps = resultTimeSteps


class SimulationResultTimestepDto(object):
    def __init__(self, unix, coinsHeld, timeStepTotal):
        self.unix = unix
        self.coinsHeld = coinsHeld
        self.timeStepTotal = timeStepTotal


if __name__ == "__main__":
    print("module")
