import tailProbabilityEstimationUnit as tpe
from scipy.stats import gamma, lognorm, pareto, genpareto


if __name__ == '__main__':
    dataSize = 500
    trueValue = 0.005
    percentageLHS = 0.9
    percentageRHS = percentageLHS+trueValue
    thresholdPercentage = 0.85
    alpha = 0.05
    random_state = 20220222
    gEllipsoidalDimension = 3
    nExperimentReptition = 200
    randomStateIncrement = 99
    print(tpe.tailProbabilityEstimationPerRep(
        lognorm, percentageLHS, percentageRHS, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state+randomStateIncrement))
