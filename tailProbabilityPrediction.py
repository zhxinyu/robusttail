import dataPreparationUtils as dputils
import tailProbabilityPredictionUnit as tailPUnit
import typing
from scipy.stats import gamma, lognorm, pareto, genpareto


def tailProbabilityPredictionPerRep(dataModule,
                                    percentageLHS: float, percentageRHS: float,
                                    dataSize: int, thresholdPercentage: float,
                                    gEllipsoidalDimension: int,
                                    alpha: float,
                                    random_state: int) -> typing.List[float]:
    leftEndPointObjective = dputils.endPointGeneration(
        dataModule, percentageLHS, dputils.dataModuleToDefaultParamDict[dataModule])
    rightEndPointObjective = dputils.endPointGeneration(
        dataModule, percentageRHS, dputils.dataModuleToDefaultParamDict[dataModule])
    inputData = dputils.RawDataGeneration(
        dataModule, dputils.dataModuleToDefaultParamDict[dataModule], dataSize, random_state)
    outputPerRep = [0]*6
    outputPerRep[0] = tailPUnit.OptimizationWithRectangularConstraint(0,
                                                                      inputData,
                                                                      thresholdPercentage,
                                                                      alpha,
                                                                      leftEndPointObjective, rightEndPointObjective,
                                                                      dataSize, 7*random_state+1)
    outputPerRep[1] = tailPUnit.OptimizationWithRectangularConstraint(1,
                                                                      inputData,
                                                                      thresholdPercentage,
                                                                      alpha,
                                                                      leftEndPointObjective, rightEndPointObjective,
                                                                      dataSize, 7*random_state+1)
    outputPerRep[2] = tailPUnit.OptimizationWithRectangularConstraint(2,
                                                                      inputData,
                                                                      thresholdPercentage,
                                                                      alpha,
                                                                      leftEndPointObjective, rightEndPointObjective,
                                                                      dataSize, 7*random_state+1)
    outputPerRep[3] = tailPUnit.OptimizationWithEllipsodialConstraint(0,
                                                                      inputData,
                                                                      thresholdPercentage,
                                                                      alpha,
                                                                      leftEndPointObjective, rightEndPointObjective,
                                                                      gEllipsoidalDimension,
                                                                      dataSize, 7*random_state+1)

    outputPerRep[4] = tailPUnit.OptimizationWithEllipsodialConstraint(1,
                                                                      inputData,
                                                                      thresholdPercentage,
                                                                      alpha,
                                                                      leftEndPointObjective, rightEndPointObjective,
                                                                      gEllipsoidalDimension,
                                                                      dataSize, 7*random_state+1)

    outputPerRep[5] = tailPUnit.OptimizationWithEllipsodialConstraint(2,
                                                                      inputData,
                                                                      thresholdPercentage,
                                                                      alpha,
                                                                      leftEndPointObjective, rightEndPointObjective,
                                                                      gEllipsoidalDimension,
                                                                      dataSize, 7*random_state+1)
    return outputPerRep


if __name__ == '__main__':
    dataSize = 500
    percentageLHS = 0.99
    percentageRHS = 0.999
    thresholdPercentage = 0.7
    alpha = 0.05
    trueValue = 0.009
    nExperimentRepetition = 500
    random_state = 20220222
    gEllipsoidalDimension = 3
    val = tailProbabilityPredictionPerRep(
        gamma, percentageLHS, percentageRHS, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state)
    print(val)
