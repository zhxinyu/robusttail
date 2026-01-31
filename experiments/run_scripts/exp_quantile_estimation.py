from multiprocessing import Pool
import quantile_estimation.quantileEstimationUnit as qe
from scipy.stats import gamma, lognorm, pareto
import pandas as pd
import numpy as np
import itertools
metaDataDict = {"dataSize": 500,
                "quantitleValue": 0.99,
                "thresholdPercentage": 0.7,
                "alpha": 0.05,
                "gEllipsoidalDimension": 3}

stringToDataModule = {"gamma": gamma,
                      "lognorm": lognorm,
                      "pareto": pareto}


def run(poolParam):
    dataDistribution, metaDataDict, random_state = poolParam
    metaDataDict = metaDataDict.copy()
    metaDataDict["random_state"] = random_state
    return qe.quantileEstimationPerRep(
        stringToDataModule[dataDistribution], **metaDataDict)


if __name__ == '__main__':
    randomSeed = 20220222
    dataDistributions = ['gamma', 'lognorm', 'pareto']
    thresholdPercentages = [0.60, 0.70, 0.80, 0.90, [0.60, 0.70, 0.80, 0.90]]
    # served as the target percentage the problem aims to estimate the quantile point from.
    quantitleValues = [0.99]
    dataSizes = [500]
    for dataDistribution, dataSize, quantitleValue, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, quantitleValues, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["quantitleValue"] = quantitleValue
        metaDataDict["thresholdPercentage"] = thresholdPercentage
        poolParamList = (dataDistribution, metaDataDict, randomSeed+0)
        try:
            results = run(poolParamList)
            print(results)
        except Exception as e:
            print(e)