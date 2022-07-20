from multiprocessing import Pool
import tailProbabilityEstimationUnit as tpe
from scipy.stats import gamma, lognorm, pareto, genpareto
import pandas as pd
import numpy as np
import os
import itertools
import sys
import traceback
FILE_DIR = "testResultSmall"
metaDataDict = {"dataSize": 500,
                "percentageLHS": 0.99,
                "percentageRHS": 0.995,
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
    return tpe.tailProbabilityEstimationPerRep(
        stringToDataModule[dataDistribution], **metaDataDict)


if __name__ == '__main__':
    ## generate a folder `testResult` if it does not exist.
    if not os.path.isdir(FILE_DIR):
       os.mkdir(FILE_DIR)

    nExperimentReptition = 10
    trueValue = 0.005
    randomSeed = 20220222
    dataDistributions = ['gamma', 'lognorm', 'pareto']
    ## as the min of the multi-threshold list, e.g. multi-threshold list: [0.6, 0.61, 0.62, 0.63, 0.64]
    thresholdPercentages = [0.6, 0.65, 0.70, 0.75, 0.8]
    ## served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    percentageLHSs = np.linspace(0.9, 0.99, 10).tolist()
    dataSizes = [500, 800]
    for dataDistribution, dataSize, percentageLHS, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, percentageLHSs, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["percentageLHS"] = percentageLHS
        metaDataDict["percentageRHS"] = percentageLHS+trueValue
        metaDataDict["thresholdPercentage"] = [thresholdPercentage +
                                               increment for increment in [0, 0.01, 0.02, 0.03, 0.04]]
        assert "random_state" not in metaDataDict
        poolParamList = [(dataDistribution, metaDataDict, random_state+randomSeed)
                         for random_state in range(nExperimentReptition)]
        FILE_NAME = ["tailProbabilityEstimation"]
        FILE_NAME += ["dataDistribution="+dataDistribution]
        FILE_NAME += [key+"="+str(metaDataDict[key])
                      for key in metaDataDict]
        FILE_NAME += ["randomSeed="+str(randomSeed)]
        FILE_NAME += ["nExperimentReptition="+str(nExperimentReptition)]
        FILE_NAME = '_'.join(FILE_NAME)+".csv"
        FILE_NAME = FILE_NAME.replace("00000000000001","").replace("0000000000001","")
        if os.path.exists(os.path.join(FILE_DIR, FILE_NAME)):
            print("Note: Already exists! Write: " +
                  os.path.join(FILE_DIR, FILE_NAME))
            df = pd.read_csv(os.path.join(FILE_DIR, FILE_NAME),
                             index_col="Experiment Repetition Index")
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
                              header=["(0,KS)", "(1,KS)", "(2,KS)",
                                      "(0,CHI2)", "(1,CHI2)", "(2,CHI2)"],
                              index=True,
                              index_label="Experiment Repetition Index")
                    del df
                print("Success!")
            except BaseException as ex:
                print("Fail on "+os.path.join(FILE_DIR, FILE_NAME))
                os.remove(os.path.join(FILE_DIR, FILE_NAME))

                # Get current system exception
                ex_type, ex_value, ex_traceback = sys.exc_info()

                # Extract unformatter stack traces as tuples
                trace_back = traceback.extract_tb(ex_traceback)

                # Format stacktrace
                stack_trace = list()

                for trace in trace_back:
                    stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                        trace[0], trace[1], trace[2], trace[3]))

                print("Exception type : %s " % ex_type.__name__)
                print("Exception message : %s" % ex_value)
                print("Stack trace : %s" % stack_trace)
