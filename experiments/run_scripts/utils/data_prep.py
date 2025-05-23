from scipy.stats import gamma, lognorm, pareto, genpareto

dataModuleToDefaultParamDict = {
    gamma: {"a": 0.5, "scale": 1},  # a: shape parameter
    lognorm:  {"loc": 0, "s": 1},     # s: standard deviation, loc: mean
    pareto: {"b": 1.5, "scale": 1},     # b: shape parameter
    genpareto: {"c": 2.17}}              # c: shape parameter 
# quantile point for 0.99 is [3.317, 10.24, 21.54]. 

# dataModuleToDefaultParamDict = {
#     gamma: {"a": 2.5, "scale": 1.5},  # a: shape parameter
#     lognorm:  {"loc": 0, "s": 1},     # s: standard deviation, loc: mean
#     pareto: {"b": 2, "scale": 5},     # b: shape parameter
#     genpareto: {"c": 2}}              # c: shape parameter


def RawDataGeneration(dataModule, paramDict, dataSize: int, random_state: int):
    return dataModule.rvs(
        size=dataSize, random_state=random_state, **paramDict)


def endPointGeneration(dataModule, percentage: float, paramDict):
    # To generate leftEndpoint and rightEndpoint for objective function 1_{lhs<=x<=rhs}.
    return dataModule.ppf(q=percentage, **paramDict)
