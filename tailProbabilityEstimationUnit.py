import dataPreparationUtils as dpu
import optimizationUnit as ou
import typing
from scipy.stats import gamma, lognorm, pareto, genpareto


def tailProbabilityEstimationWithRectangularConstraintPerRep(dataModule,
                                                             percentageLHS: float, percentageRHS: float,
                                                             dataSize: int, thresholdPercentage: typing.Union[float, typing.List[float]],
                                                             alpha: float,
                                                             random_state: int) -> typing.List[float]:
    leftEndPointObjective = dpu.endPointGeneration(
        dataModule, percentageLHS, dpu.dataModuleToDefaultParamDict[dataModule])
    rightEndPointObjective = dpu.endPointGeneration(
        dataModule, percentageRHS, dpu.dataModuleToDefaultParamDict[dataModule])
    inputData = dpu.RawDataGeneration(
        dataModule, dpu.dataModuleToDefaultParamDict[dataModule], dataSize, random_state)
    outputPerRep = [0]*3
    outputPerRep[0] = ou.OptimizationWithRectangularConstraint(0,
                                                               inputData,
                                                               thresholdPercentage,
                                                               alpha,
                                                               leftEndPointObjective, rightEndPointObjective,
                                                               dataSize, 7*random_state+1)
    outputPerRep[1] = ou.OptimizationWithRectangularConstraint(1,
                                                               inputData,
                                                               thresholdPercentage,
                                                               alpha,
                                                               leftEndPointObjective, rightEndPointObjective,
                                                               dataSize, 7*random_state+1)
    outputPerRep[2] = ou.OptimizationWithRectangularConstraint(2,
                                                               inputData,
                                                               thresholdPercentage,
                                                               alpha,
                                                               leftEndPointObjective, rightEndPointObjective,
                                                               dataSize, 7*random_state+1)
    return outputPerRep


def tailProbabilityEstimationWithEllipsodialConstraintPerRep(dataModule,
                                                             percentageLHS: float, percentageRHS: float,
                                                             dataSize: int, thresholdPercentage: typing.Union[float, typing.List[float]],
                                                             gEllipsoidalDimension: int,
                                                             alpha: float,
                                                             random_state: int) -> typing.List[float]:
    leftEndPointObjective = dpu.endPointGeneration(
        dataModule, percentageLHS, dpu.dataModuleToDefaultParamDict[dataModule])
    rightEndPointObjective = dpu.endPointGeneration(
        dataModule, percentageRHS, dpu.dataModuleToDefaultParamDict[dataModule])
    inputData = dpu.RawDataGeneration(
        dataModule, dpu.dataModuleToDefaultParamDict[dataModule], dataSize, random_state)
    outputPerRep = [0]*3
    outputPerRep[0] = ou.OptimizationWithEllipsodialConstraint(0,
                                                               inputData,
                                                               thresholdPercentage,
                                                               alpha,
                                                               leftEndPointObjective, rightEndPointObjective,
                                                               gEllipsoidalDimension,
                                                               dataSize, 7*random_state+1)

    outputPerRep[1] = ou.OptimizationWithEllipsodialConstraint(1,
                                                               inputData,
                                                               thresholdPercentage,
                                                               alpha,
                                                               leftEndPointObjective, rightEndPointObjective,
                                                               gEllipsoidalDimension,
                                                               dataSize, 7*random_state+1)

    outputPerRep[2] = ou.OptimizationWithEllipsodialConstraint(2,
                                                               inputData,
                                                               thresholdPercentage,
                                                               alpha,
                                                               leftEndPointObjective, rightEndPointObjective,
                                                               gEllipsoidalDimension,
                                                               dataSize, 7*random_state+1)
    return outputPerRep


def tailProbabilityEstimationPerRep(dataModule,
                                    percentageLHS: float, percentageRHS: float,
                                    dataSize: int, thresholdPercentage: typing.Union[float, typing.List[float]],
                                    gEllipsoidalDimension: int,
                                    alpha: float,
                                    random_state: int) -> typing.List[float]:
    outputPerRep = tailProbabilityEstimationWithRectangularConstraintPerRep(dataModule,
                                                                            percentageLHS, percentageRHS,
                                                                            dataSize, thresholdPercentage,
                                                                            alpha,
                                                                            random_state)
    outputPerRep.extend(
        tailProbabilityEstimationWithEllipsodialConstraintPerRep(dataModule,
                                                                 percentageLHS, percentageRHS,
                                                                 dataSize, thresholdPercentage,
                                                                 gEllipsoidalDimension,
                                                                 alpha,
                                                                 random_state))
    return outputPerRep


if __name__ == '__main__':
    dataSize = 500
    trueValue = 0.005
    percentageLHS = 0.9
    percentageRHS = percentageLHS+trueValue
    thresholdPercentage = 0.7
    alpha = 0.05
    random_state = 20220222
    gEllipsoidalDimension = 3
    print(tailProbabilityEstimationPerRep(
        gamma, percentageLHS, percentageRHS, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state))
    thresholdPercentage = [0.65, 0.7, 0.75, 0.8]
    print(tailProbabilityEstimationPerRep(
        gamma, percentageLHS, percentageRHS, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state))
