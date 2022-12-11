import numpy as np
import pandas as pd
import itertools
import os
from tableFiveOne import tableFiveOne


if __name__ == "__main__": 
    FILE_DIR = "testResultSmall"
    nExperimentReptition = 10
    randomSeed = 20220222
    trueValue = 0.005
    dataDistributions = ['gamma', 'lognorm', 'pareto']
    dataSizes = [500, 800]

    metaDataDict = {"dataSize": 500,
                    "percentageLHS": 0.99,
                    "percentageRHS": 0.995,
                    "thresholdPercentage": 0.7,
                    "alpha": 0.05,
                    "gEllipsoidalDimension": 3}


    thresholdPercentages = np.linspace(0.6, 0.85, 11).tolist()
    # served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    percentageLHSs = np.linspace(0.9, 0.99, 10).tolist()
    columnNames = ['dataDistribution','dataSize','percentageLHS', 'percentageRHS', "thresholdPercentage", "trueValue", "nrepIndex", "(0,KS)","(1,KS)","(2,KS)","(0,CHI2)","(1,CHI2)","(2,CHI2)"]
    cumDf1 = pd.DataFrame(columns=columnNames)
    for dataDistribution, dataSize, percentageLHS, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, percentageLHSs, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["percentageLHS"] = percentageLHS
        metaDataDict["percentageRHS"] = percentageLHS+trueValue
        metaDataDict["thresholdPercentage"] = thresholdPercentage
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
        df = pd.read_csv(os.path.join(FILE_DIR, FILE_NAME),
                        index_col="Experiment Repetition Index")
        df.reset_index(inplace=True)
        df.rename(columns={"Experiment Repetition Index":"nrepIndex"},inplace=True)
        df["dataDistribution"] = dataDistribution
        df["dataSize"] = dataSize
        df["percentageLHS"] = percentageLHS
        df["percentageRHS"] = percentageLHS+trueValue
        df["thresholdPercentage"] = thresholdPercentage
        df["trueValue"] = trueValue
        cumDf1 = cumDf1.append(df)
        
        
    go1 = cumDf1.groupby(by=['dataDistribution','dataSize','percentageLHS','thresholdPercentage'])

    targetColumns = ['(0,CHI2)','(1,CHI2)','(2,CHI2)','(0,KS)','(1,KS)','(2,KS)']
    keyChoice1 = ('gamma', 500, 0.99, 0.7)
    keyChoice2 = ('gamma', 800, 0.99, 0.7)
    subTableTitle1 = r"Data sample size = $500$."
    subTableLabel1 = "stb11_tpe_gamma"
    subTableTitle2 = r"Data sample size = $800$."
    subTableLabel2 = "stb12_tpe_gamma"
    tableTitle = r"Tail probablity estimation with Gamma data source."
    tableLabel = "stb1_tpe_gamma"
    print(tableFiveOne(go1, targetColumns, keyChoice1, keyChoice2, subTableTitle1, subTableLabel1, subTableTitle2, subTableLabel2, tableTitle, tableLabel))
    keyChoice1 = ('lognorm', 500, 0.99, 0.7)
    keyChoice2 = ('lognorm', 800, 0.99, 0.7)
    subTableTitle1 = r"Data sample size = $500$."
    subTableLabel1 = "stb11_tpe_lognormal"
    subTableTitle2 = r"Data sample size = $800$."
    subTableLabel2 = "stb12_tpe_lognormal"
    tableTitle = r"Tail probablity estimation with Lognormal data source."
    tableLabel = "tb1_tpe_lognormal"
    print(tableFiveOne(go1, targetColumns, keyChoice1, keyChoice2, subTableTitle1, subTableLabel1, subTableTitle2, subTableLabel2, tableTitle, tableLabel))
    keyChoice1 = ('pareto', 500, 0.99, 0.7)
    keyChoice2 = ('pareto', 800, 0.99, 0.7)
    subTableTitle1 = r"Data sample size = $500$."
    subTableLabel1 = "stb11_tpe_pareto"
    subTableTitle2 = r"Data sample size = $800$."
    subTableLabel2 = "stb12_tpe_pareto"
    tableTitle = r"Tail probablity estimation with Pareto data source."
    tableLabel = "tb1_tpe_pareto"
    print(tableFiveOne(go1, targetColumns, keyChoice1, keyChoice2, subTableTitle1, subTableLabel1, subTableTitle2, subTableLabel2, tableTitle, tableLabel))
