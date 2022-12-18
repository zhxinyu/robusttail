import numpy as np
import pandas as pd
import itertools
import os

import dataPreparationUtils as dpu
from scipy.stats import gamma, lognorm, pareto, genpareto


FILE_DIR = "small"
nExperimentReptition = 10
randomSeed = 20220222
trueValue = 0.005
dataDistributions = ['gamma', 'lognorm', 'pareto']
dataSizes = [500]


def tableOne():
    
    metaDataDict = {"dataSize": 500,
                "percentageLHS": 0.99,
                "percentageRHS": 0.995,
                "thresholdPercentage": 0.7,
                "alpha": 0.05,
                "gEllipsoidalDimension": 3}


    thresholdPercentages = [0.7]
    # served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    percentageLHSs = [0.99]
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

def tableTwo():
    stringToDataModule = {"gamma": gamma,
                          "lognorm": lognorm,
                          "pareto": pareto,
                          "genpareto": genpareto}
    metaDataDict = {"dataSize": 500,
                    "quantitleValue": 0.99,
                    "thresholdPercentage": 0.7,
                    "alpha": 0.05,
                    "gEllipsoidalDimension": 3}    

    thresholdPercentages = [0.70]
    quantitleValues = [0.99]
    dataSizes = [500]
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

def tableThreeOne():
    
    metaDataDict = {"dataSize": 500,
                "percentageLHS": 0.99,
                "percentageRHS": 0.995,
                "thresholdPercentage": 0.7,
                "alpha": 0.05,
                "gEllipsoidalDimension": 3}


    thresholdPercentages = [0.6, 0.7, 0.8, 0.9]
    # served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    percentageLHSs = [0.99]
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
        FILE_NAME = FILE_NAME.replace("00000000000001","").replace("0000000000001","").replace("0.8999999999999999","0.9")
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

def tableThreeTwo():
    
    metaDataDict = {"dataSize": 500,
                "percentageLHS": 0.99,
                "percentageRHS": 0.995,
                "thresholdPercentage": 0.7,
                "alpha": 0.05,
                "gEllipsoidalDimension": 3}


    thresholdPercentages = [0.6]
    # served as the lhsEndpoint in the objective function: 1_{lhs<=x<=rhs}.
    percentageLHSs = [0.99]
    dataSizes = [500]
    columnNames = ['dataDistribution','dataSize','percentageLHS', 'percentageRHS', "thresholdPercentage", "trueValue", "nrepIndex", "(0,KS)","(1,KS)","(2,KS)","(0,CHI2)","(1,CHI2)","(2,CHI2)"]

    cumDf2 = pd.DataFrame(columns=columnNames)
    for dataDistribution, dataSize, percentageLHS, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, percentageLHSs, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["percentageLHS"] = percentageLHS
        metaDataDict["percentageRHS"] = percentageLHS+trueValue
        metaDataDict["thresholdPercentage"] = [thresholdPercentage +
                                               increment for increment in [0, 0.1, 0.2, 0.3]]
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
        FILE_NAME = FILE_NAME.replace("00000000000001","").replace("0000000000001","").replace("0.8999999999999999","0.9")
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

def tableThreeThree():
    stringToDataModule = {"gamma": gamma,
                          "lognorm": lognorm,
                          "pareto": pareto,
                          "genpareto": genpareto}
    metaDataDict = {"dataSize": 500,
                    "quantitleValue": 0.99,
                    "thresholdPercentage": 0.7,
                    "alpha": 0.05,
                    "gEllipsoidalDimension": 3}    

    thresholdPercentages = [0.60, 0.70, 0.80, 0.90]
    quantitleValues = [0.99]
    dataSizes = [500]
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
        FILE_NAME = FILE_NAME.replace("00000000000001","").replace("0000000000001","").replace("0.8999999999999999","0.9")
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

def tableThreeFour():
    stringToDataModule = {"gamma": gamma,
                          "lognorm": lognorm,
                          "pareto": pareto,
                          "genpareto": genpareto}
    metaDataDict = {"dataSize": 500,
                    "quantitleValue": 0.99,
                    "thresholdPercentage": 0.7,
                    "alpha": 0.05,
                    "gEllipsoidalDimension": 3}    

    
    thresholdPercentages = [0.6]
    quantitleValues = [0.99]
    dataSizes = [500]
    columnNames = ['dataDistribution','dataSize','quantitleValue', "thresholdPercentage", "trueValue", "nrepIndex",
                   "(0,CHI2)","(1,CHI2)","(2,CHI2)"]

    cumDf4 = pd.DataFrame(columns=columnNames)
    for dataDistribution, dataSize, quantitleValue, thresholdPercentage in itertools.product(*[dataDistributions, dataSizes, quantitleValues, thresholdPercentages]):
        metaDataDict["dataSize"] = dataSize
        metaDataDict["quantitleValue"] = quantitleValue
        metaDataDict["thresholdPercentage"] = [thresholdPercentage +
                                               increment for increment in [0, 0.1, 0.2, 0.3]]
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
        FILE_NAME = FILE_NAME.replace("00000000000001","").replace("0000000000001","").replace("0.8999999999999999","0.9")
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

def tableFour():
    
    metaDataDict = {"dataSize": 500,
                "percentageLHS": 0.99,
                "percentageRHS": 0.995,
                "thresholdPercentage": 0.7,
                "alpha": 0.05,
                "gEllipsoidalDimension": 3}


    thresholdPercentages = [0.7]
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

