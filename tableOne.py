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
    dataDistribution, metaDataDict, random_state = poolParam
    metaDataDict["random_state"] = random_state
    return tailP.tailProbabilityPredictionPerRep(
        stringToDataModule[dataDistribution], **metaDataDict)


if __name__ == '__main__':
    nExperimentReptition = 200
    randomSeed = 20220222
    dataDistributions = ['gamma', 'lognorm']
    thresholdPercentages = np.linspace(0.6, 0.85, 11)
    # served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    percentageLHSs = np.linspace(0.9, 0.99, 10)
    dataSizes = [500, 800]
    for dataDistribution, dataSize, percentageLHS, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, percentageLHSs, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["percentageLHS"] = percentageLHS
        metaDataDict["percentageRHS"] = percentageLHS+0.005
        metaDataDict["thresholdPercentage"] = thresholdPercentage
        assert "random_state" not in metaDataDict
        poolParamList = [(dataDistribution, metaDataDict, random_state+randomSeed)
                         for random_state in range(nExperimentReptition)]
        FILE_NAME = ["dataDistribution="+dataDistribution]
        FILE_NAME += [key+"="+str(metaDataDict[key])
                      for key in metaDataDict]
        FILE_NAME += ["randomSeed="+str(randomSeed)]
        FILE_NAME += ["nExperimentReptition="+str(nExperimentReptition)]
        FILE_NAME = '_'.join(FILE_NAME)+".csv"
        if os.path.exists(os.path.join(FILE_DIR, FILE_NAME)):
            print("Note: Already exists! Write: " +
                  os.path.join(FILE_DIR, FILE_NAME))
        else:
            with Pool() as p:
                df = pd.DataFrame(np.asarray(p.map(parallelRun, poolParamList))
                                  )
                print(df.mean(axis=0).values)
                df.to_csv(os.path.join(FILE_DIR, FILE_NAME),
                          header=["(0,KS)", "(1,KS)", "(2,KS)",
                                  "(0,CHI2)", "(1,CHI2)", "(2,CHI2)"],
                          index=True,
                          index_label="Experiment Repetition Index")
                del df

            print("Write: " +
                  os.path.join(FILE_DIR, FILE_NAME))
