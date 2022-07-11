from multiprocessing import Pool
import quantileEstimationUnit as qe
from scipy.stats import gamma, lognorm, pareto, genpareto
import pandas as pd
import numpy as np
import os
import itertools
FILE_DIR = "testResultSmall"
metaDataDict = {"dataSize": 500,
                "quantitleValue": 0.99,
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
    return qe.quantileEstimationnPerRep(
        stringToDataModule[dataDistribution], **metaDataDict)


if __name__ == '__main__':
    ## generate a folder `testResult` if it does not exist.
    if not os.path.isdir(FILE_DIR):
       os.mkdir(FILE_DIR) 
    nExperimentReptition = 10
    randomSeed = 20220222
    # trueValue = dpu.endPointGeneration(
    #     gamma, quantitleValue, dpu.dataModuleToDefaultParamDict[gamma])
    dataDistributions = ['gamma', 'lognorm']
    thresholdPercentages = np.linspace(0.6, 0.85, 11).tolist()
    # served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    quantitleValues = np.linspace(0.9, 0.99, 10).tolist()
    dataSizes = [500, 800]
    for dataDistribution, dataSize, quantitleValue, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, quantitleValues, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["quantitleValue"] = quantitleValue
        metaDataDict["thresholdPercentage"] = thresholdPercentage
        assert "random_state" not in metaDataDict
        poolParamList = [(dataDistribution, metaDataDict, random_state+randomSeed)
                         for random_state in range(nExperimentReptition)]
        FILE_NAME  = ["quantileEstimation"]
        FILE_NAME += ["dataDistribution="+dataDistribution]
        FILE_NAME += [key+"="+str(metaDataDict[key])
                      for key in metaDataDict]
        FILE_NAME += ["randomSeed="+str(randomSeed)]
        FILE_NAME += ["nExperimentReptition="+str(nExperimentReptition)]
        FILE_NAME = '_'.join(FILE_NAME)+".csv"
        if os.path.exists(os.path.join(FILE_DIR, FILE_NAME)):
            print("Note: Already exists! Write: " +
                  os.path.join(FILE_DIR, FILE_NAME))
            df = pd.read_csv(os.path.join(FILE_DIR, FILE_NAME), index_col="Experiment Repetition Index")
            print(df.mean(axis=0).values)
        else:
            print("Writing: " +
                os.path.join(FILE_DIR, FILE_NAME))
            try:
                with Pool() as p:
                    df = pd.DataFrame(np.asarray(p.map(parallelRun, poolParamList))
                                    )
                    print(df.mean(axis=0).values)
                    df.to_csv(os.path.join(FILE_DIR, FILE_NAME),
                            header=["(0,CHI2)", "(1,CHI2)", "(2,CHI2)"],
                            index=True,
                            index_label="Experiment Repetition Index")
                    del df
                print("Success!")
            except:
                print("Fail on "+os.path.join(FILE_DIR, FILE_NAME))
