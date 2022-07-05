from scipy.stats import gamma, lognorm, pareto, genpareto


dataModuleToDefaultParamDict = {gamma: {"a": 2.5, "scale": 1.5},  # a is the shape parameter
                                # s: standard deviation, loc: mean
                                lognorm:  {"loc": 0, "s": 1},
                                pareto: {"b": 2, "scale": 20},  # b: shape
                                genpareto: {"c": 2}}  # c: shape


def RawDataGeneration(dataModule, paramDict, dataSize: int, random_state: int):
    return dataModule.rvs(
        size=dataSize, random_state=random_state, **paramDict)


def endPointGeneration(dataModule, percentage: float, paramDict):
    # To generate leftEndpoint and rightEndpoint for objective function 1_{lhs<=x<=rhs}.
    return dataModule.ppf(q=percentage, **paramDict)
