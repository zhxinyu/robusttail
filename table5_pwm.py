## Use Bayesian method to estimate CI.
import dataPreparationUtils as dpu
from scipy.stats import gamma, lognorm, pareto
import os
try:
	from rpy2.robjects.packages import importr
except:
    os.environ['R_HOME'] = os.path.join(os.environ['CONDA_PREFIX'], 'lib', 'R')
    from rpy2.robjects.packages import importr
    
import rpy2.robjects as ro
import pandas as pd
import numpy as np
from rpy2.robjects import numpy2ri
numpy2ri.activate()

importr('base')
utils = importr('utils')
try:
	importr('POT')
	importr('MASS')
	importr('eva')
except:
	utils.install_packages('POT', contribulr="https://cran.microsoft.com/")
	utils.install_packages('MASS', contribulr="https://cran.microsoft.com/")
	utils.install_packages('eva', contribulr="https://cran.microsoft.com/")
	importr('POT')
	importr('MASS')    
	importr('eva')
 
with open('gpdTIP_pwm.R','r') as f:
    POTUtilityinR = f.read()


if __name__ == '__main__':
    import argparse  
    FILE_DIR = f'large_{__file__}'
    if not os.path.exists(FILE_DIR):
        os.makedirs(FILE_DIR)
        
    parser = argparse.ArgumentParser(description='TIP estimation using probability weighted moment method.')
    parser.add_argument('lhs_st', type=float, help='start of LHS list in the objective function')
    parser.add_argument('ds', type=str, help='Data source for simulation')
    args = parser.parse_args()
    
    randomSeed = 20220222
    stringToDataModule = {"gamma": gamma,
						  "lognorm": lognorm,
						  "pareto": pareto}
    trueValue = 0.005
    # percentageLHSs = np.linspace(0.9, 0.99, 10).tolist()
    # dataSources = ['gamma','lognorm','pareto']
    percentageLHSs = [args.lhs_st]
    # percentageLHSs = np.arange(args.lhs_st, args.lhs_ed, 0.01)
    dataSources = [ args.ds ]
    metaDataDict = {"dataSize": 500}
    
    nrep = 200
    result = []
    columns = ["Data Source", 'nData',"percentageLHS", "Lower Bound","Upper Bound", "True Value", "Repetition Index"]
    for percentageLHS in percentageLHSs:
        percentageLHS = np.round(percentageLHS,2)
        percentageRHS = percentageLHS + trueValue
        for dataSource in dataSources:
            dataModule = stringToDataModule[dataSource]
            leftEndPointObjective = dpu.endPointGeneration(
                dataModule, percentageLHS, dpu.dataModuleToDefaultParamDict[dataModule])
            rightEndPointObjective = dpu.endPointGeneration(
                dataModule, percentageRHS, dpu.dataModuleToDefaultParamDict[dataModule])
            for nnrep in range(nrep):
                print(f"Working on {percentageLHS}_{dataSource}_{nnrep}")
                try:
                    metaDataDict['random_state'] = randomSeed+nnrep
                    inputData = dpu.RawDataGeneration(dataModule, 
                                                      dpu.dataModuleToDefaultParamDict[dataModule], 
                                                      metaDataDict['dataSize'], 
                                                      metaDataDict['random_state'])
                    POTApply = '''
lhs <- {:}

rhs <- {:}

data <- c({:})

out <- tryCatch(
    gpdTIP(data, lhs, rhs, conf=0.95), 
    error = function(e) e

)
if ("error" %in% class(out)) {{
  print(out)
}}

bbd <- if ("error" %in% class(out)) NA else{{
    out$CI
}}
bbd

                    '''.format(leftEndPointObjective, 
                               rightEndPointObjective, 
                               ', '.join([str(eachData)for eachData in inputData.tolist()]))
                    roResult = ro.r(POTUtilityinR+POTApply)
                    result.append([dataSource, metaDataDict['dataSize'], percentageLHS, roResult[0], roResult[1], trueValue, nnrep])
                    print(result[-1])
                except Exception as e: 
                    print(e)
                    result.append([dataSource, metaDataDict['dataSize'], 
                                percentageLHS, 0, 
                                0, 
                                trueValue, nnrep])
            print(f"Finish experiments on {percentageLHS}-{dataSource}")
            df = pd.DataFrame(data = result, columns = columns)
            df.to_csv(os.path.join(FILE_DIR, f'table5_{percentageLHS}_{dataSource}.csv'), header=columns, index = False)
    # df = pd.DataFrame(data = result, columns = columns)

    # df.to_csv(os.path.join(FILE_DIR, f'tableFive_bayesian_{args.lhs_st}_{args.ds}.csv'), header=columns, index = False)
