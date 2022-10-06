import tailProbabilityEstimationUnit as tpe
import quantileEstimationUnit as qe

from scipy.stats import gamma, lognorm, pareto, genpareto

stringToDataModule = {"gamma": gamma,
                      "lognorm": lognorm,
                      "pareto": pareto,
                      "genpareto": genpareto}

metaDataDict = {"dataSize": 500,
                "quantitleValue": 0.99,
                "thresholdPercentage": 0.7,
                "alpha": 0.05,
                "gEllipsoidalDimension": 3}

def tpeTest():
    dataSize = 500
    trueValue = 0.005
    percentageLHS = 0.9
    percentageRHS = percentageLHS+trueValue
    thresholdPercentage = 0.85
    alpha = 0.05
    random_state = 20220222
    gEllipsoidalDimension = 3
    randomStateIncrement = 99
    print(tpe.tailProbabilityEstimationPerRep(
        lognorm, percentageLHS, percentageRHS, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state+randomStateIncrement))

def qeTest(metaDataDict):    
    metaDataDict["dataSize"] = 500
    metaDataDict["quantitleValue"] = 0.99
    metaDataDict["thresholdPercentage"] = 0.85
    nExperimentReptition = 10
    randomSeed = 20220222
    dataDistribution = 'pareto'
    poolParamList = [(dataDistribution, metaDataDict, randomSeed+random_state)
                     for random_state in range(nExperimentReptition)]

    for i in range(len(poolParamList)):
        dataDistribution, metaDataDict, random_state = poolParamList[i]
        metaDataDict["random_state"] = random_state
        res = qe.quantileEstimationnPerRep(
            stringToDataModule[dataDistribution], **metaDataDict)
        print(res)
if __name__ == '__main__':
    qeTest(metaDataDict)