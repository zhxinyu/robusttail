from multiprocessing import Pool
import tailProbabilityEstimationUnit as tpe
from scipy.stats import gamma, lognorm, pareto, genpareto
import pandas as pd
import numpy as np
import os
import itertools
import sys
import traceback

FILE_DIR = "testResultRealData"
metaDataDict = {"alpha": 0.05,
                "gEllipsoidalDimension": 3}

stringToDataModule = {"gamma": gamma,
                      "lognorm": lognorm,
                      "pareto": pareto,
                      "genpareto": genpareto}

if __name__ == '__main__':
    ## generate a folder `testResult` if it does not exist.
    if not os.path.isdir(FILE_DIR):
       os.mkdir(FILE_DIR)
    randomSeed = 20220222
    thresholdPercentages = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    # served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    regions = ['FIJI_ISLANDS_REGION', 'HOKKAIDO_JAPAN_REGION', 'KURIL_ISLANDS', 'OFF_COAST_OF_NORTHERN_CA']
    for region in regions:
        inputData = np.loadtxt(f'./data_cmt/{region}.csv')
        for thresholdPercentage in thresholdPercentages:
            metaDataDict["thresholdPercentage"] = thresholdPercentage
            metaDataDict["random_state"] = randomSeed
            FILE_NAME = f"{region}_{thresholdPercentage:.2f}.csv"
            print("Writing: " +
                    os.path.join(FILE_DIR, FILE_NAME))
            try:                                          
                df = tpe.tailProbabilityEstimationWithRealData(inputData=inputData,
                                                            leftEndPointObjective=7.25, rightEndPointObjective=np.inf,
                                                            **metaDataDict)
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
