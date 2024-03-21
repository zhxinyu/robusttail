## Use Bayesian method to estimate CI.
from table5_central_import import *
 
with open('gpdTIP_pwm.R','r') as f:
    POTUtilityinR = f.read()


if __name__ == '__main__':
    import argparse  
    file_dir = f'large_{os.path.basename(__file__).replace('.py', '')}'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        
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
            df.to_csv(os.path.join(file_dir, f'table5_{percentageLHS}_{dataSource}.csv'), header=columns, index = False)
    # df = pd.DataFrame(data = result, columns = columns)

    # df.to_csv(os.path.join(file_dir, f'tableFive_bayesian_{args.lhs_st}_{args.ds}.csv'), header=columns, index = False)
