## Case 1: 
	# simulated n = 30 CLT
import numpy as np

from table5_central_import import *

data_size = 30

def runner(args):
    randomSeed = 20220222
    stringToDataModule = {"gamma":   gamma,
                          "lognorm": lognorm,
                          "pareto":  pareto}
    trueValue = 0.005
    metaDataDict = {"dataSize": data_size}
    dataSources = [ args.ds ]
    method = args.method
    if method == 'pot':
        # We have two sources of failures: a). unable to solve fitGPD
        #                                  b). unable to obtain the cov matrix
        with open('gpdTIP_pot.R','r') as f:
            RCodeLib = f.read()
    if method == 'pot_bt':
        with open('gpdTIP_pot_bt.R','r') as f:
            RCodeLib = f.read()    
    elif method == 'pl':
        with open('gpdTIP_pl.R','r') as f:
            RCodeLib = f.read()
    elif method == 'bayesian':
        with open('gpdTIP_bayesian.R','r') as f:
            RCodeLib = f.read()
    elif method == 'pwm':
        with open('gpdTIP_pwm.R','r') as f:
            RCodeLib = f.read()
    else:
        raise NotImplementedError()
    
    file_dir = f'results/n{data_size}_table5_{method}'
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    
    nrep = 200
    result = []
    columns = ["Data Source", 'nData',"percentile", "Lower Bound","Upper Bound", "True Value", "Repetition Index"]
    percentageLHS = 0.99
    percentageRHS = percentageLHS + trueValue
    # percentiles = [0.6, 0.7]
    # for percentile in percentiles:
    percentile = 0 # placeholder
    for dataSource in dataSources:
        dataModule = stringToDataModule[dataSource]
        leftEndPointObjective = dpu.endPointGeneration(
            dataModule, percentageLHS, dpu.dataModuleToDefaultParamDict[dataModule])
        rightEndPointObjective = dpu.endPointGeneration(
            dataModule, percentageRHS, dpu.dataModuleToDefaultParamDict[dataModule])
        for nnrep in range(nrep):
            print(f"Working on {dataSource}_{nnrep}")
            try:
                metaDataDict['random_state'] = randomSeed+nnrep
                RCodeApply = f'''
lhs <- {leftEndPointObjective}

rhs <- {rightEndPointObjective}

data <-  c(t(read.csv("./n{data_size}/{dataSource}/default/randomseed={randomSeed+nnrep}.csv", header=FALSE)))
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
                    '''
                roResult = ro.r(RCodeLib+RCodeApply)
                if len(roResult)==1 or (not isinstance(roResult[0], float) and not isinstance(roResult[1], float)):
                    print("something wrongs with the limited amount of data")
                    result.append([dataSource, metaDataDict['dataSize'], percentile, 0, 0, trueValue, nnrep])
                else:
                    result.append([dataSource, metaDataDict['dataSize'], percentile, roResult[0], roResult[1], trueValue, nnrep])
                print(result[-1])
            except Exception as e: 
                print(e)
                result.append([dataSource, metaDataDict['dataSize'], 
                            percentile, 0, 
                            0, 
                            trueValue, nnrep])
            # assert False
            print(f"Finish experiments on {dataSource}")
            df = pd.DataFrame(data = result, columns = columns)
            df.to_csv(os.path.join(file_dir, f'table5_{dataSource}.csv'), header=columns, index = False)

if __name__ == '__main__':

    """Usage
    
        ```
        #!/bin/bash 
        for lhs in 0.9 0.95 0.99
        do
            echo "Running" ${lhs} gamma pl
            python table5_ab.py ${lhs} gamma pl &
            echo "Running" ${lhs} lognorm pl
            python table5_ab.py ${lhs} lognorm pl &
            echo "Running" ${lhs} pareto pl
            python table5_ab.py ${lhs} pareto pl &
        done
        wait 
        echo "done all processes."
        ``` 
    """
    parser = argparse.ArgumentParser(description='TIP estimation with user-specific methods.')
    parser.add_argument('ds', type=str, help='Data source for simulation')
    parser.add_argument('method', type=str, help='choose the method for tail probability estimation: pot, pl, bayesian and pwm')
    args = parser.parse_args()
    
    runner(args)
    