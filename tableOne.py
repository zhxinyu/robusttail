from multiprocessing import Pool
import tailProbabilityPrediction as tailP
from scipy.stats import gamma, lognorm, pareto, genpareto
import pandas as pd
import numpy as np
import os
import itertools
FILE_DIR = "testResult"
metaDataDict = {"dataSize": 500,
                "percentageLHS": 0.99,
                "percentageRHS": 0.999,
                "thresholdPercentage": 0.7,
                "alpha": 0.05,
                "gEllipsoidalDimension": 3}

stringToDataModule = {"gamma": gamma,
                      "lognorm": lognorm,
                      "pareto": pareto,
                      "genpareto": genpareto}


def parallelRun(poolParam):
    dataDistribution, dataSize, percentageLHS, thresholdPercentage, random_state = poolParam
    metaDataDict["dataSize"] = dataSize
    metaDataDict["percentageLHS"] = percentageLHS
    metaDataDict["percentageRHS"] = percentageLHS+0.05
    metaDataDict["thresholdPercentage"] = thresholdPercentage

    metaDataDict["random_state"] = random_state
    val = tailP.tailProbabilityPredictionPerRep(
        stringToDataModule[dataDistribution], **metaDataDict)
    return val


if __name__ == '__main__':
    nExperimentReptition = 500
    randomSeed = 20220222
    dataDistributions = ['gamma', 'lognorm']
    thresholdPercentages = np.linspace(0.6, 0.85, 11)
    percentageLHSs = np.linspace(0.9, 0.99, 10)
    dataSizes = [500, 800, 1000, 1200, 1500, 1800, 2000, 2500, 3000]
    for dataDistribution, dataSize, percentageLHS, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, percentageLHSs, thresholdPercentages]):

        poolParamList = [(dataDistribution, dataSize, percentageLHS, thresholdPercentage, random_state+randomSeed)
                         for random_state in range(nExperimentReptition)]
        metaDataDict["dataSize"] = dataSize
        metaDataDict["percentageLHS"] = percentageLHS
        metaDataDict["percentageRHS"] = percentageLHS+0.005
        metaDataDict["thresholdPercentage"] = thresholdPercentage

        FILE_NAME = ["dataDistribution="+dataDistribution]
        FILE_NAME += [key+"="+str(metaDataDict[key])
                      for key in metaDataDict]
        FILE_NAME += ["randomSeed="+str(randomSeed)]
        FILE_NAME += ["nExperimentReptition="+str(nExperimentReptition)]
        FILE_NAME = '_'.join(FILE_NAME)+".csv"
        if os.path.exists(os.path.join(FILE_DIR, FILE_NAME)+".csv"):
            print("Note: Already exists! Write: " +
                  os.path.join(FILE_DIR, FILE_NAME))
        else:
            with Pool() as p:
                pd.DataFrame(np.asarray(p.map(parallelRun, poolParamList))
                             ).to_csv(os.path.join(FILE_DIR, FILE_NAME),
                                      header=["(0,KS)", "(1,KS)", "(2,KS)",
                                              "(0,CHI2)", "(1,CHI2)", "(2,CHI2)"],
                                      index=True,
                                      index_label="Experiment Repetition Index")
            print("Write: " +
                  os.path.join(FILE_DIR, FILE_NAME))
