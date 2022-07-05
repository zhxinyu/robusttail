from multiprocessing import Pool
import tailProbabilityPrediction as tailP
from scipy.stats import gamma, lognorm, pareto, genpareto

def parallelRun(random_state):
    dataSize = 500
    percentageLHS = 0.99
    percentageRHS = 0.999
    thresholdPercentage = 0.7
    alpha = 0.05
    gEllipsoidalDimension = 3
    val = tailP.tailProbabilityPredictionPerRep(
        gamma, percentageLHS, percentageRHS, dataSize, thresholdPercentage, gEllipsoidalDimension, alpha, random_state)
    return val
    
if __name__ == '__main__':
    nExperimentReptition = 8
    poolParamList = [random_state for random_state in range(nExperimentReptition)]
    with Pool() as p:
        print(p.map(parallelRun,poolParamList))
    