from scipy.stats import gamma, lognorm, pareto, genpareto

dataModuleToDefaultParamDict = {
    gamma: {"a": 1, "scale": 2},  # a: shape parameter
    lognorm:  {"loc": 0, "s": 2},     # s: standard deviation, loc: mean
    pareto: {"b": 1, "scale": 10},     # b: shape parameter
    genpareto: {"c": 2.17}}              # c: shape parameter 
# quantile point for 0.99 is [9.21034037197618, 104.86730070562277, 999.9999999999991, 11417.21105231628]. 

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
