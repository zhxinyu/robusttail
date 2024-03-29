import numpy as np
import pandas as pd
import itertools
import os

import dataPreparationUtils as dpu
from scipy.stats import gamma, lognorm, pareto, genpareto


FILE_DIR = "testResultSmall-copy"
nExperimentReptition = 10
randomSeed = 20220222
trueValue = 0.005
dataDistributions = ['gamma', 'lognorm', 'pareto']
dataSizes = [500, 800]


def tpeWsingle():
    
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
    return cumDf1

def tpeWmultiple():
    
    metaDataDict = {"dataSize": 500,
                "percentageLHS": 0.99,
                "percentageRHS": 0.995,
                "thresholdPercentage": 0.7,
                "alpha": 0.05,
                "gEllipsoidalDimension": 3}


    thresholdPercentages = [0.6, 0.65, 0.70, 0.75, 0.8]
    # served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    percentageLHSs = np.linspace(0.9, 0.99, 10).tolist()
    dataSizes = [500, 800]
    columnNames = ['dataDistribution','dataSize','percentageLHS', 'percentageRHS', "thresholdPercentage", "trueValue", "nrepIndex", "(0,KS)","(1,KS)","(2,KS)","(0,CHI2)","(1,CHI2)","(2,CHI2)"]

    cumDf2 = pd.DataFrame(columns=columnNames)
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
        cumDf2 = cumDf2.append(df)    
    return cumDf2

def qeWsingle():
    stringToDataModule = {"gamma": gamma,
                          "lognorm": lognorm,
                          "pareto": pareto,
                          "genpareto": genpareto}
    metaDataDict = {"dataSize": 500,
                    "quantitleValue": 0.99,
                    "thresholdPercentage": 0.7,
                    "alpha": 0.05,
                    "gEllipsoidalDimension": 3}    

    thresholdPercentages = np.linspace(0.6, 0.85, 11).tolist()
    quantitleValues = np.linspace(0.9, 0.99, 10).tolist()
    dataSizes = [500, 800]
    columnNames = ['dataDistribution','dataSize','quantitleValue', "thresholdPercentage", "trueValue", "nrepIndex",
                   "(0,CHI2)","(1,CHI2)","(2,CHI2)"]

    cumDf3 = pd.DataFrame(columns=columnNames)
    for dataDistribution, dataSize, quantitleValue, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, quantitleValues, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["quantitleValue"] = quantitleValue
        metaDataDict["thresholdPercentage"] = thresholdPercentage
        assert "random_state" not in metaDataDict
        poolParamList = [(dataDistribution, metaDataDict, random_state+randomSeed)
                         for random_state in range(nExperimentReptition)]
        FILE_NAME = ["quantileEstimation"]
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
        df["quantitleValue"] = quantitleValue
        df["thresholdPercentage"] = thresholdPercentage
        trueValue = dpu.endPointGeneration(
            stringToDataModule[dataDistribution], quantitleValue, dpu.dataModuleToDefaultParamDict[stringToDataModule[dataDistribution]])        
        df["trueValue"] = trueValue

        cumDf3 = cumDf3.append(df)    
    return cumDf3


def qeWmultiple():
    
    stringToDataModule = {"gamma": gamma,
                          "lognorm": lognorm,
                          "pareto": pareto,
                          "genpareto": genpareto}
    metaDataDict = {"dataSize": 500,
                    "quantitleValue": 0.99,
                    "thresholdPercentage": 0.7,
                    "alpha": 0.05,
                    "gEllipsoidalDimension": 3}    

    
    thresholdPercentages = [0.6, 0.65, 0.70, 0.75, 0.8]
    quantitleValues = np.linspace(0.9, 0.99, 10).tolist()
    dataSizes = [500, 800]
    columnNames = ['dataDistribution','dataSize','quantitleValue', "thresholdPercentage", "trueValue", "nrepIndex",
                   "(0,CHI2)","(1,CHI2)","(2,CHI2)"]

    cumDf4 = pd.DataFrame(columns=columnNames)
    for dataDistribution, dataSize, quantitleValue, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, quantitleValues, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["quantitleValue"] = quantitleValue
        metaDataDict["thresholdPercentage"] = [thresholdPercentage +
                                               increment for increment in [0, 0.01, 0.02, 0.03, 0.04]]
        assert "random_state" not in metaDataDict
        poolParamList = [(dataDistribution, metaDataDict, random_state+randomSeed)
                         for random_state in range(nExperimentReptition)]
        FILE_NAME = ["quantileEstimation"]
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
        df["quantitleValue"] = quantitleValue
        df["thresholdPercentage"] = thresholdPercentage    
        trueValue = dpu.endPointGeneration(
            stringToDataModule[dataDistribution], quantitleValue, dpu.dataModuleToDefaultParamDict[stringToDataModule[dataDistribution]])    
        df["trueValue"] = trueValue

        cumDf4 = cumDf4.append(df)            
    return cumDf4