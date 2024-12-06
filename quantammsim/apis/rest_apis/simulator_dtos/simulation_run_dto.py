

class LoadPriceHistoryRequestDto(object):
    def __init__(self, jsonDto):
        self.coinCode = jsonDto["coinCode"]


class TrainingParameterDto(object):
    def __init__(self, factorDto):
        self.name = factorDto["name"]
        self.value = factorDto["value"]


class FinancialAnalysisRequestDto(object):
    def __init__(self, jsonDto):
        self.startDateString = jsonDto["startDateString"]
        self.endDateString = jsonDto["endDateString"]
        self.tokens = jsonDto["tokens"]
        self.returns = jsonDto["returns"]
        self.benchmarks = jsonDto["benchmarks"]


# input root
class SimulationRunDto(object):
    def __init__(self, jsonDto):
        self.pool = LiquidityPoolDto(jsonDto["pool"])
        self.startDate = jsonDto["startUnix"]
        self.endDate = jsonDto["endUnix"]
        self.startDateString = jsonDto["startDateString"]
        self.endDateString = jsonDto["endDateString"]


class TrainingDto(object):
    def convert_to_run_fingerprint(self):
        optimisation_settings = dict()
        update_rule_parameters = dict()

        for urp in self.pool.updateRule.updateRuleFactors:
            update_rule_parameters[urp.name] = urp.value

        for opt in self.trainingParameters.trainingParameters:
            optimisation_settings[opt.name] = opt.value
        return {
            "filename_override": "override",
            "startDateUnix": self.startDate,
            "endDateUnix": self.endDate,
            "tokens": [
                constituent.coinCode for constituent in self.pool.poolConstituents
            ],
            "rule": self.pool.updateRule.name,
            "optimisation_settings": {
                "base_lr": float(optimisation_settings.get("base_lr")),
                "optimiser": optimisation_settings.get("optimiser"),
                "decay_lr_ratio": float(optimisation_settings.get("decay_lr_ratio")),
                "decay_lr_plateau": float(
                    optimisation_settings.get("decay_lr_plateau")
                ),
                "batch_size": int(optimisation_settings.get("batch_size")),
                "train_on_hessian_trace": optimisation_settings.get(
                    "train_on_hessian_trace"
                )
                == "True",
                "min_lr": float(optimisation_settings.get("min_lr")),
                "n_iterations": int(optimisation_settings.get("n_iterations")),
                "n_cycles": int(optimisation_settings.get("n_cycles")),
                "return_val": float(optimisation_settings.get("return_val")),
            },
            "initial_memory_length": update_rule_parameters.get("memory_length"),
            "initial_memory_length_delta": update_rule_parameters.get(
                "memory_length_delta"
            ),
            "initial_k": update_rule_parameters.get("k"),
            "bout_offset": 30 * 24 * 60 * 6,
            "initial_weights_logits": update_rule_parameters.get("weights_logits"),
            "initial_log_amplitude": update_rule_parameters.get("log_amplitude"),
            "initial_raw_width": update_rule_parameters.get("raw_width"),
            "initial_raw_exponents": update_rule_parameters.get("raw_exponents"),
            "subsidary_pools": [],
            "chunk_period": int(update_rule_parameters.get("chunk_period")),
            "weight_interpolation_period": int(
                update_rule_parameters.get("weight_interpolation_period")
            ),
            "initial_pool_value": update_rule_parameters.get("ipool_value"),
            "fees": update_rule_parameters.get("fees"),
            "use_alt_lamb": update_rule_parameters.get("memory_length_delta")
            is not None,
        }

    def __init__(self, jsonDto):
        self.trainingRunFilename = jsonDto["trainingRunFilename"]
        self.pool = LiquidityPoolDto(jsonDto["pool"])
        self.startDate = jsonDto["startUnix"]
        self.endDate = jsonDto["endUnix"]
        self.trainingParameters = TrainingParametersDto(jsonDto["trainingParameters"])


class TrainingParametersDto(object):
    def __init__(self, paramDto):
        params = list()
        for param in paramDto["trainingParameters"]:
            params.append(TrainingParameterDto(param))
        self.trainingParameters = params


class LiquidityPoolDto(object):
    def __init__(self, poolDto):
        self.id = poolDto["id"]
        poolConstituents = list()
        for coin in poolDto["poolConstituents"]:
            poolConstituents.append(LiquidityPoolCoinDto(coin))
        self.poolConstituents = poolConstituents
        self.updateRule = UpdateRuleDto(poolDto["updateRule"])


class UpdateRuleDto(object):
    def __init__(self, ruleDto):
        self.name = ruleDto["name"]
        factors = list()
        for coin in ruleDto["UpdateRuleParameters"]:
            factors.append(UpdateRuleFactorDto(coin))
        self.updateRuleFactors = factors


class UpdateRuleFactorDto(object):
    def __init__(self, factorDto):
        self.name = factorDto["name"]
        tokenValues = list()
        for tokenValue in factorDto["value"]:
            tokenValues.append(float(tokenValue))
        self.value = tokenValues


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
    def __init__(self, result):
        self.timeSteps = result["resultTimeSteps"]
        self.analysis = result["analysis"]


class FinancialAnalysisResult(object):
    def __init__(self, result):
        self.analysis = result


class SimulationResultTimestepDto(object):
    def __init__(self, unix, coinsHeld, timeStepTotal):
        self.unix = unix
        self.coinsHeld = coinsHeld
        self.timeStepTotal = timeStepTotal


